import os

from keras.layers import Convolution2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
import matplotlib as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
size = 128

classifier = Sequential()

classifier.add(Convolution2D(27, (3, 3), input_shape = (size, size, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(27, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 144, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5), activation = 'relu'))
classifier.add(Dropout(0.30))


classifier.add(Dense(units = 96, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5), activation = 'relu'))
classifier.add(Dropout(0.30))





classifier.add(Dense(units = 96, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5), activation = 'relu'))
classifier.add(Dropout(0.30))





classifier.add(Dense(units =27, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.summary()







from keras.preprocessing.image import ImageDataGenerator

Training_Dataset_Generator = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

Testing_Dataset_Generator = ImageDataGenerator(rescale = 1./255)


Training_Dataset = Training_Dataset_Generator.flow_from_directory('Dataset/Training Data', target_size = (size, size),batch_size = 1, color_mode = 'grayscale', class_mode = 'categorical')

Testing_Dataset = Testing_Dataset_Generator.flow_from_directory('Dataset/Test Data', target_size = (size, size), batch_size = 1, color_mode = 'grayscale', class_mode = 'categorical')

classifier.fit_generator(Training_Dataset, steps_per_epoch  = 22127, epochs = 20, validation_data = Testing_Dataset, validation_steps = 200, verbose =1)


Trained_model = classifier.to_json()
with open("Saved_Model.json", "w") as json_file:
    json_file.write(Trained_model)
print("Model Saved")
classifier.save_weights('Saved_Weights.h5')
print("Weights Saved")
#print(history.history['val_accuracy', 'val_loss'])
