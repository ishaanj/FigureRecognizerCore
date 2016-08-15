from keras.layers import Input, Dropout, Flatten, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.utils.visualize_util import plot
from keras.layers.convolutional import Convolution2D, AveragePooling2D

ip = Input(shape=(3, 128, 128))

x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(ip)
x = Convolution2D(64, 3, 3, activation='relu', subsample=(2, 2), border_mode='same')(x)

x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
x = Convolution2D(128, 3, 3, activation='relu', subsample=(2, 2))(x)

x = Convolution2D(196, 3, 3, activation='relu', subsample=(2, 2), border_mode='same')(x)
x = Convolution2D(196, 3, 3, activation='relu', subsample=(2, 2), border_mode='same')(x)

x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)

x = AveragePooling2D((8,8))(x)

x = Dropout(0.5)(x)
x = Flatten()(x)

out = Dense(7, activation='softmax')(x)

model = Model(ip, out)

model.summary()
plot(model, 'mini-vgg.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['acc'])
print('Compiling model.')

trainPath = r"C:\Shape Dataset\train"
testPath = r"C:\Shape Dataset\test"

batchSize = 128
nbEpoch = 100

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(trainPath, target_size=(128, 128), batch_size=batchSize, seed=1)
print('Loaded training data.')

datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = datagen.flow_from_directory(testPath, target_size=(128, 128), batch_size=batchSize, seed=1, )
print('Loaded testing data.')
print("Precompiling...")

model.load_weights('Model Weights.h5')

#model.fit_generator(train_generator, samples_per_epoch=350000, nb_epoch=nbEpoch,
#                    callbacks=[ModelCheckpoint('Model Weights.h5', monitor='val_acc', save_best_only=True, save_weights_only=True)],
#                    validation_data=test_generator, nb_val_samples=70000, nb_worker=4)

scores = model.evaluate_generator(test_generator, 70000)
print(scores)