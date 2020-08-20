#Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout,SpatialDropout2D,Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(200, (3, 3), input_shape=(128,128,1) , activation = 'relu'))
#adding droupout 2d
classifier.add(SpatialDropout2D(0.5))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#------------------------------------------------
# Step 1 - Convolution
classifier.add(Conv2D(100, (3, 3), input_shape=(128,128,1) , activation = 'relu'))
#adding droupout 2d
classifier.add(SpatialDropout2D(0.5))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dropout(0.5))
classifier.add(Dense(units =50, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units =20, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss ='binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory("Dataset/training_set",
                                                 target_size = (128, 128),
                                                 color_mode = 'grayscale',
                                                 batch_size = 15,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory("Dataset/test_set",
                                            target_size = (128, 128),
                                            color_mode = 'grayscale',
                                            batch_size = 15,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs =2,
                         validation_data = test_set,
                         validation_steps = 400)

classifier.save('Model.h5')

'''
# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
model = load_model('Model.h5')
test_image = image.load_img('pic210.jpg', target_size = (128, 128), grayscale=True)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print(result)
model.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
'''