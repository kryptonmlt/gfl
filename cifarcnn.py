import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os

# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt
import cifar10s as cifar10
from cifar10s import img_size, num_channels, num_classes
import helper

cifar10.maybe_download_and_extract()
class_names = cifar10.load_class_names()
print(class_names)

# 50,000 images/labels
images_train, cls_train, labels_train = cifar10.load_training_data()

# 10,000 images/labels
images_test, cls_test, labels_test = cifar10.load_test_data()

print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))

img_size_cropped = 24


def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.
    if training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image.
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)
    return image


def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)

    return images


def main_network(images, training):
    # Wrap the input images as a Pretty Tensor object.
    x_pretty = pt.wrap(images)

    # Pretty Tensor uses special numbers to distinguish between
    # the training and testing phases.
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer

    # Create the convolutional neural network using Pretty Tensor.
    # It is very similar to the previous tutorials, except
    # the use of so-called batch-normalization in the first layer.
    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty. \
            conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True). \
            max_pool(kernel=2, stride=2). \
            conv2d(kernel=5, depth=64, name='layer_conv2'). \
            max_pool(kernel=2, stride=2). \
            flatten(). \
            fully_connected(size=256, name='layer_fc1'). \
            fully_connected(size=128, name='layer_fc2'). \
            softmax_classifier(num_classes=num_classes, labels=y_true)

    return y_pred, loss


def create_network(training):
    # Wrap the neural network in the scope named 'network'.
    # Create new variables during training, and re-use during testing.
    with tf.variable_scope('network', reuse=not training):
        # Just rename the input placeholder variable for convenience.
        images = x

        # Create TensorFlow graph for pre-processing.
        images = pre_process(images=images, training=training)

        # Create TensorFlow graph for the main processing.
        y_pred, loss = main_network(images=images, training=training)

    return y_pred, loss


def random_batch():
    # Number of images in the training-set.
    num_images = len(images_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch


def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = random_batch()

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        # We also want to retrieve the global_step counter.
        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        # Print status to screen every 100 iterations (and last).
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            summary, batch_acc = session.run([merged, accuracy],
                                             feed_dict=feed_dict_train)
            train_writer.add_summary(summary, i)
            # Print status.
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

        # Save a checkpoint to disk every 1000 iterations (and last).
        if (i_global % 1000 == 0) or (i == num_iterations - 1):
            # Save all variables of the TensorFlow graph to a
            # checkpoint. Append the global_step counter
            # to the filename so we save the last several checkpoints.
            saver.save(session,
                       save_path=save_path,
                       global_step=global_step)
            helper.print_test_accuracy(session, images_test, labels_test, cls_test,
                                       x, y_true, y_pred_cls, num_classes, class_names,
                                       show_example_errors=False, show_confusion_matrix=False)
            print("Saved checkpoint.")

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# Get the first images from the test-set.
# images = images_test[0:9]
# Get the true classes for those images.
# cls_true = cls_test[0:9]
# Plot the images and labels using our helper-function above.
# plot_images(images=images, cls_true=cls_true, smooth=True)

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1)

distorted_images = pre_process(images=x, training=True)

# Create Neural Network for Training Phase
global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

_, loss = create_network(training=True)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

# Create Neural Network for Test Phase / Inference
y_pred, _ = create_network(training=False)

y_pred_cls = tf.argmax(y_pred, dimension=1)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

tf.summary.scalar('accuracy', accuracy)

# weights_conv1 = helper.get_weights_variable(layer_name='layer_conv1')
# weights_conv2 = helper.get_weights_variable(layer_name='layer_conv2')
# output_conv1 = helper.get_layer_output(layer_name='layer_conv1')
# output_conv2 = helper.get_layer_output(layer_name='layer_conv2')

session = tf.Session()

save_dir = 'checkpoints/'
train_sum_dir = 'checkpoints/train_summaries/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'cifar10_cnn')

try:
    print("Trying to restore last checkpoint ...")

    # Use TensorFlow to find the latest checkpoint - if any.
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

    # Try and load the data in the checkpoint.
    saver.restore(session, save_path=last_chk_path)

    # If we get to this point, the checkpoint was successfully loaded.
    print("Restored checkpoint from:", last_chk_path)
except:
    # If the above failed for some reason, simply
    # initialize all the variables for the TensorFlow graph.
    print("Failed to restore checkpoint. Initializing variables instead.")
    session.run(tf.global_variables_initializer())

train_batch_size = 64

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(train_sum_dir, session.graph)

optimize(num_iterations=100000)

helper.print_test_accuracy(session, images_test, labels_test, cls_test,
                           x, y_true, y_pred_cls, num_classes, class_names,
                           show_example_errors=True, show_confusion_matrix=True)
