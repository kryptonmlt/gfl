import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
import prettytensor as pt
import cifar10s as cifar10
from cifar10s import img_size, num_channels, num_classes
import helper

save_dir = 'cifar_devices_checkpoints/'
train_sum_dir = save_dir + 'train_summaries/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'cifar10_cnn')

global_epoch = 1
local_epoch = 100
train_batch_size = 50
clients = 10
training_per_client = 500
testing_per_client = 100
train_file = 'train.txt'
test_file = 'test.txt'

img_size_cropped = 24


def update_device_data(file, i_t_d, c_t_d, l_t_d, i_t_g, c_t_g, l_t_g):
    with open(file) as f:
        d = 0
        for line in f:
            if d == clients:
                break
            images_labels = line.split(':')
            imgs_ids = list(map(int, images_labels[0].split(',')))
            lbls_ids = list(map(int, images_labels[1].split(',')))

            for i in range(len(imgs_ids)):
                i_t_d[d].append(i_t_g[lbls_ids[i]][imgs_ids[i]])
                c_t_d[d].append(c_t_g[lbls_ids[i]][imgs_ids[i]])
                l_t_d[d].append(l_t_g[lbls_ids[i]][imgs_ids[i]])

            d += 1


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


def main_network(images, training, y_true):
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


def create_network(device_id, training, input_image, y_true):
    # Wrap the neural network in the scope named 'network'.
    # Create new variables during training, and re-use during testing.
    with tf.variable_scope(str(device_id) + 'network', reuse=not training):
        # Create TensorFlow graph for pre-processing.
        images = pre_process(images=input_image, training=training)

        # Create TensorFlow graph for the main processing.
        y_pred, loss = main_network(images=images, training=training, y_true=y_true)

    return y_pred, loss


def random_batch(images_train, labels_train):
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


def get_maximum_batch(images_train):
    return int(len(images_train) / train_batch_size)


def get_batch(batch, images_train, labels_train):
    # Create a random index.

    start = batch * train_batch_size;
    last = (batch + 1) * train_batch_size;
    # Use the random index to select random images and labels.
    x_batch = images_train[start:last]
    y_batch = labels_train[start:last]
    if isinstance(images_train, np.ndarray):
        x_batch = images_train[start:last, :, :, :]
        y_batch = labels_train[start:last, :]
    else:
        x_batch = images_train[start:last]
        y_batch = labels_train[start:last]

    return x_batch, y_batch


def optimize(device_id, num_iterations, images_train, labels_train, images_test, labels_test, cls_test, y_pred_cls, class_names,
             session, merged, train_writer, local_step, accuracy, optimizer, x, y_true):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):
        max_batch = get_maximum_batch(images_train)
        for b in range(max_batch):
            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            # x_batch, y_true_batch = random_batch(images_train, labels_train)
            x_batch, y_true_batch = get_batch(b, images_train, labels_train)

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            # We also want to retrieve the global_step counter.
            i_local, _ = session.run([local_step, optimizer],
                                      feed_dict=feed_dict_train)

        # Print status to screen every 100 iterations (and last).
        if  (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            summary, batch_acc = session.run([merged, accuracy],
                                             feed_dict=feed_dict_train)
            train_writer.add_summary(summary, i)
            # Print status.
            msg = "Device: {0} Local Step: {1:>6}, Training Batch Accuracy: {2:>6.1%}"
            print(msg.format(device_id, i_local, batch_acc))

        if  (i == num_iterations - 1):
            helper.print_test_accuracy(session, images_test, labels_test, cls_test,
                                       x, y_true, y_pred_cls, num_classes, class_names,
                                       show_example_errors=False, show_confusion_matrix=False)

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def main():
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

    # split into a list per label
    images_train_group = [[] for n in range(10)]
    cls_train_group = [[] for n in range(10)]
    labels_train_group = [[] for n in range(10)]
    images_test_group = [[] for n in range(10)]
    cls_test_group = [[] for n in range(10)]
    labels_test_group = [[] for n in range(10)]

    for i in range(50000):
        cur_label = cls_train[i]
        images_train_group[cur_label].append(images_train[i])
        labels_train_group[cur_label].append(labels_train[i])
        cls_train_group[cur_label].append(cls_train[i])

    for i in range(10000):
        cur_label = cls_test[i]
        images_test_group[cur_label].append(images_test[i])
        labels_test_group[cur_label].append(labels_test[i])
        cls_test_group[cur_label].append(cls_test[i])

    # split into a list per device
    images_train_device = [[] for n in range(clients)]
    cls_train_device = [[] for n in range(clients)]
    labels_train_device = [[] for n in range(clients)]
    images_test_device = [[] for n in range(clients)]
    cls_test_device = [[] for n in range(clients)]
    labels_test_device = [[] for n in range(clients)]

    update_device_data(train_file,
                       images_train_device, cls_train_device, labels_train_device,
                       images_train_group, cls_train_group, labels_train_group)
    update_device_data(test_file,
                       images_test_device, cls_test_device, labels_test_device,
                       images_test_group, cls_test_group, labels_test_group)
    print(images_train_device[0][0].shape)
    print(images_train.shape)
    # prepare input placeholders in tensorflow
    x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

    y_true_cls = tf.argmax(y_true, dimension=1)

    distorted_images = pre_process(images=x, training=True)

    # Create Neural Network for Training Phase
    losses = []
    optimizers = []
    y_preds = []
    y_pred_clss = []
    correct_predictions = []
    accuracies = []
    local_steps = []
    for d in range(clients):
        _, loss = create_network(d, training=True, input_image=x, y_true=y_true)
        local_step = tf.Variable(initial_value=0, name='local_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=local_step)
        # Create Neural Network for Test Phase / Inference
        y_pred, _ = create_network(d, training=False, input_image=x, y_true=y_true)
        y_pred_cls = tf.argmax(y_pred, dimension=1)
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar(str(d) + 'accuracy', accuracy)
        losses.append(loss)
        optimizers.append(optimizer)
        y_preds.append(y_pred)
        y_pred_clss.append(y_pred_cls)
        correct_predictions.append(correct_prediction)
        accuracies.append(accuracy)
        local_steps.append(local_step)

    saver = tf.train.Saver()
    session = tf.Session()

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

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(train_sum_dir, session.graph)

    for global_step in range(global_epoch):
        for d in range(clients):
            optimize(d, local_epoch, images_train_device[d], labels_train_device[d], images_test_device[d],
                     labels_test_device[d], cls_test_device[d], y_pred_clss[d], class_names, session,
                     merged, train_writer, local_steps[d], accuracies[d], optimizers[d], x, y_true)
            # helper.print_test_accuracy(session, images_test_device[d], labels_test_device[d], cls_test_device[d],
            #                           x, y_true, y_pred_clss[d], num_classes, class_names,
            #                           show_example_errors=True, show_confusion_matrix=True)
        saver.save(session, save_path=save_path, global_step=global_step)
        global_step += 1


if __name__ == "__main__":
    main()
