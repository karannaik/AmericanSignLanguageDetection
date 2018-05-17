import numpy as np
import pickle
import cv2, os
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import pprint
import matplotlib.pyplot as plt
from keras.layers import Dropout, Flatten, Dense, Input
from keras.applications import VGG16
from keras.models import Sequential, Model

K.set_image_dim_ordering('tf')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_image_size():
    # img = cv2.imread('gestures/0/100.jpg', 0)#change
    img = cv2.imread('gesturess/0/0 (1).jpg', 0)#change
    return img.shape


def get_num_of_classes():
    return len(os.listdir('gesturess/'))#change


image_x, image_y = get_image_size()


def cnn_model():
    num_of_classes = get_num_of_classes()

    input_tensor = Input(shape=(3, 200, 200))
    # base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

    # base_model.layers.pop()

    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='sigmoid'))

    #This is CNN2 model implemented. We need to uncomment this to use second model mentioned in the paper
    # model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='sigmoid'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    #Uncomment below model to use second model
    # model.add(Conv2D(64, (5, 5), activation='sigmoid'))
    model.add(Conv2D(64, (5, 5), activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes, activation='softmax'))

    # model_new = Model(input=base_model.input, output=model(base_model.output))

    pprint.pprint(model.layers)
    print(model.summary())

    sgd = optimizers.SGD(lr=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    filepath = "cnn_model_keras2.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # checkpoint2 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint1]
    return model, callbacks_list

def train():
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    with open("test_images", "rb") as f:
        test_images = np.array(pickle.load(f))
    with open("test_labels", "rb") as f:
        test_labels = np.array(pickle.load(f), dtype=np.int32)

    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))#change
    test_images = np.reshape(test_images, (test_images.shape[0], image_x, image_y, 1))#change
    train_labels = np_utils.to_categorical(train_labels)
    test_labels = np_utils.to_categorical(test_labels)

    model, callbacks_list = cnn_model()
    history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=50, batch_size=32,
              callbacks=callbacks_list)
    scores = model.evaluate(test_images, test_labels, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

    model.save('cnn_model_keras2.h5')

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

train()
K.clear_session();
