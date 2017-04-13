import numpy as np
from random import randint

labels = 10
train_images_max = 5000
test_images_max = 1000
train_images_device = 500
test_images_device = 100
devices = 100

train_images = [np.arange(train_images_max) for n in range(labels)]
test_images = [np.arange(test_images_max) for n in range(labels)]

train_images_device_actual = [[] for n in range(devices)]
train_labels_device_actual = [[] for n in range(devices)]
test_images_device_actual = [[] for n in range(devices)]
test_labels_device_actual = [[] for n in range(devices)]

for device in range(devices):
    for ti in range(train_images_device):

        # retrieve image
        label_chosen = randint(0, labels-1)
        while len(train_images[label_chosen]) == 0:
            label_chosen = randint(0, labels-1)
        location = randint(0, len(train_images[label_chosen])-1)
        imageChosen = train_images[label_chosen][location]

        # add image to device
        train_images_device_actual[device].append(imageChosen)
        train_labels_device_actual[device].append(label_chosen)

        # delete image from list
        train_images[label_chosen] = np.delete(train_images[label_chosen], [location])

    for ti in range(test_images_device):
        # retrieve image
        label_chosen = randint(0, labels-1)
        while len(test_images[label_chosen]) == 0:
            label_chosen = randint(0, labels-1)
        location = randint(0, len(test_images[label_chosen])-1)
        imageChosen = test_images[label_chosen][location]

        # add image to device
        test_images_device_actual[device].append(imageChosen)
        test_labels_device_actual[device].append(label_chosen)

        # delete image from list
        test_images[label_chosen] = np.delete(test_images[label_chosen], [location])

trainFile = open('train.txt', 'w')
testFile = open('test.txt', 'w')

for i in range(len(train_images_device_actual)):
    trainFile.write(",".join(str(x) for x in train_images_device_actual[i]))
    trainFile.write(":")
    trainFile.write(",".join(str(x) for x in train_labels_device_actual[i]))
    trainFile.write("\n")

for i in range(len(test_images_device_actual)):
    testFile.write(",".join(str(x) for x in test_images_device_actual[i]))
    testFile.write(":")
    testFile.write(",".join(str(x) for x in test_labels_device_actual[i]))
    testFile.write("\n")

trainFile.close()
testFile.close()
