import csv
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.layers import Lambda, Cropping2D, Conv2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from utils import *

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('bs', 256, "The batch size.")
flags.DEFINE_integer('epochs', 10, "The Epochs.")
flags.DEFINE_string('data_dir', "../../data/all/", "train data dir")
flags.DEFINE_string('model_save', 'model.h5', "model save file")
flags.DEFINE_float('dout', 0.5, 'Drop-out')
flags.DEFINE_float('lrate', 0.001, 'Learning-rate')


# read dataset descriptor
samples = []
with open(FLAGS.data_dir + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


# split the dataset with same random state to expect same split
train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2, random_state=43)


# simple read data from description( from csv file )
def read_data(data_dir, sample_desc):
    image = load_image(data_dir + sample_desc[0].split('/')[-1])
    image = correct_ground_true(image)
    angle = float(sample_desc[3])
    return preprocess(image), angle


# data augment, from the emulator, the left and right has 0.2 deviation.
def augment(data_dir, sample_desc):
    # random selection for 3 camera
    choice = np.random.choice(3)
    if choice == 0:  # left camera
        image = load_image(data_dir + sample_desc[1].split('/')[-1])
        angle = float(sample_desc[3]) + 0.2
    elif choice == 1:  # right camera
        image = load_image(data_dir + sample_desc[2].split('/')[-1])
        angle = float(sample_desc[3]) - 0.2
    else:  # center camera
        image = load_image(data_dir + sample_desc[0].split('/')[-1])
        angle = float(sample_desc[3])

    # ground true correction
    image = correct_ground_true(image, horizontal_bias=0)

    # random flip the image
    mirror = np.random.choice(2)
    if mirror:
        image = cv2.flip(image, 1)
        angle = -angle

    return preprocess(image), angle


def generator(samples, batch_size=FLAGS.bs, is_training=False):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                if is_training:
                    feature_image, angle = augment(FLAGS.data_dir + 'IMG/', batch_sample)
                else:
                    feature_image, angle = read_data(FLAGS.data_dir + 'IMG/', batch_sample)

                images.append(feature_image)
                angles.append(angle)

            X_train = np.array(images)
            Y_train = np.array(angles)
            yield shuffle(X_train, Y_train)


# make generator for each dataset
train_generator = generator(train_samples, batch_size=FLAGS.bs, is_training=True)
validation_generator = generator(validation_samples, batch_size=FLAGS.bs)


# Use a fixed value to determine the model input before loading the image.
# row, col, ch = 66, 200, 3  # Trimmed image shape
row, col, ch = 160, 320, 3  # original image shape


# Build a model
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(row, col, ch)))
# Crop the input image to fit the network
model.add(Cropping2D(cropping=((74, 20), (60, 60))))
model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Dropout(FLAGS.dout))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer=Adam(lr=FLAGS.lrate))

history_object = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_samples)/FLAGS.bs,
    epochs=FLAGS.epochs,
    validation_data=validation_generator, validation_steps=len(validation_samples)/FLAGS.bs)

model.save(FLAGS.model_save)

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

