#Import stuff
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

#Loading the data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

#Giving Name to classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Example of the dataset
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(True)
plt.show()

#Reshaping the dataset
train_images = train_images / 255.0

test_images = test_images / 255.0

#Showing the data in the dataset
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#creating the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

#Compiling the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#Training the model
model.fit(train_images, train_labels, epochs=10)

#Evaluating the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#Printing accuracy
print('\nTest accuracy:', test_acc)

#Making the prediction model
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
#Predicting....
predictions = probability_model.predict(test_images)

predicted_output = np.argmax(predictions[0])
actual_value = test_labels[0]

check = predicted_output == actual_value

if check==False :
  print("Prediction was right")
else:
  print("Prediction was wrong")
