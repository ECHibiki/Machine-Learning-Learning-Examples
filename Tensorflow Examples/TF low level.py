from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
'''

#Central to TF is tensors. Primitive values shaped into an array. Rank is dimensions. Shape is tuple of ints specifying arrays length
#numpy arrays represent tensors

#The TF core is the computational graph and running it in a session

# Graphs have Operations representing nodes, and tensors as edges

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b

print(a)
print(b)
print(total)



#The above outputs the computational graph, each with a unique name. Not values.


# evaluation requires creating a tf.Session object.

sess = tf.Session()
print(sess.run(total))

print(sess.run({'ab':(a, b), 'total':total}))

vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))

# A ML graph needs variable result. Placeholders are designed to hold future values
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))

#Datasets are however the preffered way of working with models

my_data = [
    [0, 1,],
    [2, 3,],
    [4, 5,],
    [6, 7,],
]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

while True:
  try:
    print(sess.run(next_item))
  except tf.errors.OutOfRangeError:
    break

#If statefull the itterator may need to initialized

r = tf.random_normal([10,3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

sess.run(iterator.initializer)
while True:
  try:
    print(sess.run(next_row))
  except tf.errors.OutOfRangeError:
    break

#Trainable models need values to to be modified in the graph to reach now outputs with same inputs.
# Layers are used and package variables and opperations together.
# A denseely-connected layer applies an opitonal activation on the output to all functions inputs

x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)

#Initializing the layers resulting variables
init = tf.global_variables_initializer()
sess.run(init)

#Now we can evaluate the linear model's output tensors as any otherself.

print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]}))

# Condensed removing access to the linear model layer
x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.layers.dense(x, units=1)
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))

# Feature columns are easiest done with tf.feature_column.input_layer and only accepts dense columnsself.
# Viewing requires a wrapper of indicator_column

features = {
    'sales' : [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening'])
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)

#Feature columns have an internal state like layers and require initializationself.
# Categorical columns use lookup tables requiring a different intiialization, tf.tables_initializer

var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))

# once sess initializes. Run
print(sess.run(inputs))


# Training
# Some arbritrary inputs
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32, name="C1")
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32, name="C2")

#The training model with one outputs
linear_model = tf.layers.Dense(units=1, name="L1")
y_pred = linear_model(x)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(y_pred))

# Loss to train
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
print(sess.run(loss))

#Optimizers test the loss
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#itterative training
for i in range(100):
  _, loss_value = sess.run((train, loss))
  print(loss_value)

'''



#Completed:

#The input values
x = tf.constant([[4], [3], [2], [1]], dtype=tf.float32, name="X")
#The comparison values
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32, name="Y_t")
#A dense LM
linear_model = tf.layers.Dense(units=1, name="Dense_LM")
'''Dense layer y_pred that takes a batch of input vectors,'''
#assinged to y_predictions
y_pred = linear_model(x)
#With loss operations based on y_true labels and predictions dictated by the model
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
'''y_pred produces a single output and MSE judges it'''

#Set up the basic trainer to minimize loss
optimizer = tf.train.GradientDescentOptimizer(0.01, name="gdo")
train = optimizer.minimize(loss)

#set internal states
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Train n times outputting a blank and the loss value.
#Put the training function and loss function into the run function.
for i in range(10000):
  _, loss_value = sess.run((train, loss))
  print(loss_value)

#Run the variables through the prediction model
print(sess.run(y_pred))





#TensorBoard is a way to viusalize the graphself.
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

# Go into the directory and type `tensorboard --logdir .` to view your graph


input()
