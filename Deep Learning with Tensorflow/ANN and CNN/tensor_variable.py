# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:27:07 2020

@author: elton
"""

import tensorflow as tf

#create a tensor variable with zeroes filled with default datatype float32
a_tensor=tf.Variable(tf.zeros([2, 2, 2]))

#Create a 0-D array or scalar variable with data type tf.int32
a_scalar=tf.Variable(200, tf.int32)

#Create a 1-D array or vector with data type tf.int32
an_initialized_vector=tf.Variable([1, 3, 5, 7, 9, 11], tf.int32)

#Create a 2-D array or matrix with default data type which is tf.float32
an_initialized_matrix = tf.Variable([ [2, 4], [5, 25] ])

#Get the tensor's rank and shape
rank = tf.rank(a_tensor)
shape = tf.shape(a_tensor)

#Create a constant initialized with a fixed value.
a_constant_tensor = tf.constant(123.100)
print(a_constant_tensor)
tf.print(a_constant_tensor)




