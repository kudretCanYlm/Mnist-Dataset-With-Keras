from keras import layers, models, activations, optimizers, losses, metrics
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

model = models.Sequential()

model.add(layers.Conv2D(
    32, (3, 3), activation=activations.relu))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="relu"))



def to_one_hot(labels,dimension=46):
    results=np.zeros((len(labels),dimension))
    for i,label in enumerate(labels):
        results[i,label]=1.
    return results


    

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float32")/255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32")/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer="rmsprop",
              loss='categorical_crossentropy', metrics=metrics.accuracy)

model.fit(train_images, train_labels, epochs=5, batch_size=400)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test loss: ", test_loss)
print("test acc: ", test_acc)
