#Assignment 6-Part 2
import tensorflow as TFObj
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_val, y_val) = mnist.load_data()
x_train = x_train.astype('float32') / 255
y_train = to_categorical(y_train)
A=TFObj.keras.models.Sequential()

A.add(TFObj.keras.layers.Dense(units=6,activation='relu'))

A.add(TFObj.keras.layers.Dense(units=1,activation='sigmoid'))

A.compile(optimizer='SGD', loss=BinaryCrossentropy(), metrics=['accuracy'])

A.fit(x_val,y_val,epochs=5)