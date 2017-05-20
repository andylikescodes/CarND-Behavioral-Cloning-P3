import csv
import numpy as np
import cv2

lines = []
with open('../simdata/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []

for line in lines[1:]:
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

from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Convolution2D, Dropout, MaxPooling2D

model = Sequential()

model.add(Cropping2D(cropping=((75,25),(0,0)), input_shape = (160,320,3)))
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
model.fit(X_train, y_train, nb_epoch=5, validation_split=0.2, shuffle=True)


model.save('model.h5')
