# starting with tensorflow 

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# define input and output/target/response variables
Xs = np.array([-1, 2, 3, -2, 0, 4, 5], dtype=float)
ys = np.array([-3, 3, 5, -5, -1, 7, 9], dtype=float)

# create a sequential model
model = Sequential([Dense(units=1, input_shape=[1])])
model.compile(optimizer="sgd", loss="mean_squared_error")
# train the model
model.fit(Xs,ys,epochs=500)
# predict 
predictions = model.predict([-7])
print(predictions)






