import tensorflow as tf
import timeit

import numpy as np
import matplotlib.pyplot as plt

class SequentialModel(tf.keras.Model):
    
  def __init__(self, **kwargs):
      
    super(SequentialModel, self).__init__(**kwargs)
    self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
    self.dense_1 = tf.keras.layers.Dense(64, activation="relu")
    self.dropout = tf.keras.layers.Dropout(0.2)
    self.dense_2 = tf.keras.layers.Dense(64)

  def f(self, x):
      
    x = self.flatten(x)
    x = self.dense_1(x)
    x1 = self.dropout(x)
    x2 = self.dense_2(x1)
    
    return x, x1, x2

  def call(self, x):
      
    x0,x1,x2 = self.f(x)
    
    return x1+x2

a = []
b = []
t = []

for i in range(3):
    x = tf.range(i+1, dtype=tf.float32)
    a.append(x)
    t.append(x)
    
for i in range(3):
    x = (i+1)*tf.ones(i+1, dtype=tf.float32)
    b.append(x)
    t.append(x)
    
x = np.ones(20000)
y = np.ones(20000)
z = np.ones(20000)

x1 = np.reshape(x, (-1,1))
y1 = np.reshape(y, (-1,1))
z1 = np.reshape(z, (-1,1))

@tf.function
def concat(x,y,z):
    return( tf.concat([x,y,z], axis=1) )
@tf.function
def stack(x,y,z):
    return( tf.stack([x,y,z], axis=1) )

# number = 10000
#
# print("Concat time:", timeit.timeit(lambda: concat(x1,y1,z1), number=number)/number)
# print("Stack time:", timeit.timeit(lambda: stack(x, y, z), number=number)/number)

in_dim = 128
n = 10000
i = 0.85*n

# wcut = tf.constant(1.2)*i/n + tf.constant(0.05)
# s    = 20*(tf.linspace(1.0, 0.0, in_dim) + wcut - 1.0)
#
#
# a = tf.where(s > 1.0, 1.0, s)
# a = tf.where(a < 0.0, 0.0, a)
#
# a = a/tf.reduce_mean(a)

wcut = tf.constant(1.1)*i/n + tf.constant(0.1)
s    = 20*(tf.linspace(1.0, 0.0, in_dim) + wcut - 1.0)

a = tf.where(s > 1.0, 1.0, s)
a = tf.where(a < 0.0, 0.0, a)

a = a/tf.reduce_mean(a)
        
plt.plot(a, 'o-')
plt.ylim(0,None)
# plt.plot(s, 'x--')
plt.show()

# input_data = tf.random.uniform([60, 28, 28])
#
# x,y,z = input_data

# print(x,y,z)

# eager_model = SequentialModel()
# graph_model = tf.function(eager_model)
#
# print("Eager time:", timeit.timeit(lambda: eager_model(input_data), number=10000))
# print("Graph time:", timeit.timeit(lambda: graph_model(input_data), number=10000))