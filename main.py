import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras._tf_keras.keras import layers, models, datasets

(trainingImages, trainingLabels), (testingImages, testingLabels) = datasets.cifar10.load_data()
trainingImages, testingImages = trainingImages / 255, testingImages / 255

classNames = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(trainingImages[i], cmap=plt.cm.binary)
    plt.xlabel(classNames[trainingLabels[i][0]])

plt.show()

trainingImages = trainingImages[:20000]
trainingLabels = trainingLabels[:20000]
testingImages = testingImages[:4000]
testingLabels = testingLabels[:4000]

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(trainingImages, trainingLabels, epochs=10, validation_data=(testingImages, testingLabels))

loss, accuracy = model.evaluate(testingImages, testingLabels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.keras')