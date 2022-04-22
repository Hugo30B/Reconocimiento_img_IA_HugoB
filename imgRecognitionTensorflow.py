import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as grafica
from tensorflow.keras import datasets, layers, models
import ssl

ssl._create_default_https_context = ssl._create_unverified_context # Crear certificado SSL por defecto para poder acceder a los datos

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

training_images, testing_images = training_images / 255, testing_images / 255

opciones = ['Avión', 'Coche', 'Pájaro', 'Gato', 'Reno', 'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']


# esto es para reducir el tamaño de conjunto de pruebas
#'''
training_images = training_images[:20000]
training_labels = training_labels[:20000]
training_images = training_images[:4000]
training_labels = training_labels[:4000]

# Creación y entrenamiento de la red neuronal ... esto se debe hacer una vez

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation = 'relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(training_images, training_labels, epochs = 10, validation_data=(testing_images,testing_labels))

loss, accuracy = model.evaluate(testing_images,testing_labels)


#cambiar el estilo del gráfico y guardarlo en el disco

print(f"Loss: {loss}")
print(f"Accuracy : {accuracy*100}%")

model.save('modelo_entrenado.model')


pd.DataFrame(history.history).plot()
grafica.show()
#'''


#'''
model = models.load_model('modelo_entrenado.model')

img = cv.imread("car4.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

grafica.imshow(img, cmap=grafica.cm.binary)

prediction = model.predict(np.array([img]) / 255)
indice = np.argmax(prediction)
print(f'La predicción es {opciones[indice]}')
#'''
