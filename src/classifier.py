import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense,Flatten,Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from callbacks import CustomCallback

# apply callback
custom_callback = CustomCallback() # to be used to reduce overfitting
# load data
data = tf.keras.datasets.fashion_mnist
# split data
(X_train, y_train), (X_test, y_test) = data.load_data()
# normalize data
X_train = X_train/255.0
X_test = X_test/255.0
# build a sequential model with convolutions
model = Sequential([
                    Conv2D(64, (3,3),activation=tf.nn.relu,input_shape=(28,28,1)),
                    MaxPooling2D(2,2),
                    Conv2D(64, (3,3),activation=tf.nn.relu),
                    MaxPooling2D(2,2),
                    Flatten(), # numpy-one-dim array
                    Dense(512, activation=tf.nn.relu), # 512 neurons 
                    Dense(64, activation=tf.nn.relu), # 64 neurons
                    Dense(10, activation=tf.nn.softmax), # 10 neurons
                ]
            )
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
print("training process")
model.fit(X_train, y_train, epochs=5, callbacks=[custom_callback])
# evaluate model on test data
print("evaluation process")
test_loss = model.evaluate(X_test, y_test)





