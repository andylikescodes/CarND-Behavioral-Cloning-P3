import csv
import numpy as np
import cv2



lines = []
with open('../simdata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    steer_corr = 0.2
    for line in reader:	
        center_path = line[0]
        left_path = line[1]
        right_path = line[2]
        filename_center = center_path.split('/')[-1]	
        filename_left = left_path.split('/')[-1]
        filename_right = right_path.split('/')[-1]

        steer_center = float(line[3])
        steer_left = steer_center+steer_corr
        steer_right = steer_center-steer_corr
        
        lines.append([filename_center, steer_center, False])		
        lines.append([filename_left, steer_left, False])
        lines.append([filename_right, steer_right, False])
        lines.append([filename_center, -steer_center, True])
        lines.append([filename_left, -steer_left, True])
        lines.append([filename_right, -steer_right, True])

samples = lines[1:]
		
		
		

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
                path = line[0]
                img_path = '../simdata/IMG/' + path
                image = cv2.imread(img_path)
                if line[2] == False:
                    images.append(image)
                else:
                    image_flipped = np.fliplr(image)
                    images.append(image_flipped)				 
                
                steer = float(line[1])
                measurements.append(steer)
            
            X_train = np.array(images)
            y_train = np.array(measurements)

            yield sklearn.utils.shuffle(X_train, y_train)



def nvdia_net(train_generator, train_samples, validation_generator, validation_samples, n_epoch):

    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Cropping2D, Lambda, Convolution2D, Dropout, MaxPooling2D

    model = Sequential()

    model.add(Cropping2D(cropping=((25,25),(0,0)), input_shape = (160,320,3)))
    model.add(Lambda(lambda x: x/255 - 0.5))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=n_epoch)


    model.save('model.h5')


def lenet(train_generator, train_samples, validation_generator, validation_samples, n_epoch):
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Cropping2D, Lambda, Convolution2D, Dropout,     MaxPooling2D
    
    model = Sequential()
    model.add(Cropping2D(cropping=((25,25),(0,0)), input_shape = (160,320,3)))
    model.add(Lambda(lambda x: x/255 - 0.5))
    model.add(Convolution2D(6,5,5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Convolution2D(6,5,5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(84))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=n_epoch)
    model.save('model.h5')

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
n_epoch=3

nvdia_net(train_generator, train_samples, validation_generator, validation_samples, n_epoch)

