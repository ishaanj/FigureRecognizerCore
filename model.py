from inception import create_inception_resnet_v2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint

trainPath = r"C:\Shape Dataset\train"
testPath = r"C:\Shape Dataset\test"

batchSize = 1024
nbEpoch = 100

datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(trainPath, target_size=(224, 224), batch_size=batchSize, seed=1)
print('Loaded training data')

datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = datagen.flow_from_directory(testPath, target_size=(224, 224), batch_size=batchSize, seed=1)
print('Loaded testing data')

ip = Input(shape=(3, 224, 224))
op = create_inception_resnet_v2(ip, nb_output=7)

model = Model(ip, op)

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics='acc')
print('Compiling model.')

model.fit_generator(train_generator, samples_per_epoch=350000, nb_epoch=nbEpoch,
                    callbacks=[ModelCheckpoint('Model Weights.h5', monitor='val_acc', save_best_only=True, save_weights_only=True)],
                    validation_data=test_generator, nb_val_samples=70000, nb_worker=4)

scores = model.evaluate_generator(test_generator, 70000, )