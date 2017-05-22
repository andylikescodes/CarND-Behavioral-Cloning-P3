import csv
import numpy as np
import cv2



lines = []
with open('../simdata/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

import sklearn

def generator(samples, batch_size=32):
    
    n_samples = len(samples)
    

    while 1:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            
            for line in batch_samples:
                center_path = line[0]
                left_path = line[1]
                right_path = line[2]
                filename_center = center_path.split('/')[-1]	
                filename_left = left_path.split('/')[-1]
                filename_right = right_path.split('/')[-1]
                img_center_path = '../simdata/IMG/' + filename_center	
                img_left_path = '../simdata/IMG/' + filename_left
                img_right_path = '../simdata/IMG/' + filename_right
                
                steer_corr = 0.1
                steer_center = float(line[3])
                steer_left = steer_center+steer_corr
                steer_right = steer_center-steer_corr
                
                image_center = cv2.imread(img_center_path)
                image_left = cv2.imread(img_left_path)
                image_right = cv2.imread(img_right_path)
                images.append(image_center)
                images.append(image_left)
                images.append(image_right)
                
                measurements.append(steer_center)
                measurements.append(steer_left)
                measurements.append(steer_right)

                image_center_flipped = np.fliplr(image_center)
                image_left_flipped = np.fliplr(image_left)
                image_right_flipped = np.fliplr(image_right)

                images.append(image_center_flipped)	
                images.append(image_left_flipped)
                images.append(image_right_flipped)

                measurements.append(-steer_center)
                measurements.append(-steer_left)
                measurements.append(-steer_right)

            X_train = np.array(images)
            y_train = np.array(measurements)

            yield sklearn.utils.shuffle(X_train, y_train)


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Convolution2D, Dropout, MaxPooling2D

model = Sequential()

model.add(Cropping2D(cropping=((50,25),(0,0)), input_shape = (160,320,3)))
model.add(Lambda(lambda x: x/255 - 0.5))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=(len(train_samples)*6), validation_data=validation_generator, nb_val_samples=(len(validation_samples)*6), nb_epoch=3)


model.save('model.h5')
