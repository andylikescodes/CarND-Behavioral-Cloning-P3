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
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = '../simdata/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Dropout, MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape = (160,320,3)))
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
model.fit(X_train, y_train, nb_epoch=3, validation_split=0.2, shuffle=True)


model.save('model.h5')
