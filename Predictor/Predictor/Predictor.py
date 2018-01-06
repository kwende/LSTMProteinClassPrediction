import tensorflow as tf
import numpy as np
import data_reader

# parse the training data. 
ret = data_reader.read_annotated_fasta('training.txt')

# constants
classVectorSize = len(ret[0].ClassVector)
batch_size = 3
cell1Size = 215
cell2Size = 128

# create placeholders for the input/output data. 
x = tf.placeholder(dtype=tf.float32, shape=[None, batch_size, classVectorSize])
y = tf.placeholder(dtype=tf.float32, shape=[classVectorSize])

# create my lstm cells
cell1 = tf.contrib.rnn.BasicLSTMCell(cell1Size)
cell2 = tf.contrib.rnn.BasicLSTMCell(cell2Size)
multiLayerCell = tf.contrib.rnn.MultiRNNCell(cells=[cell1, cell2])

# this is the operation that works over the lstm cells. 
_, state = tf.nn.dynamic_rnn(dtype=tf.float32, cell=multiLayerCell, inputs=x)
lstmOutput = state[-1][-1]
outputShape = lstmOutput.get_shape().as_list()

# create fully connected layer which produces the one-hot vector output. 
weights = tf.Variable(tf.truncated_normal(shape=[outputShape[-1], classVectorSize], stddev=.1))
biases = tf.Variable(tf.constant(value=.1, shape=[classVectorSize]))
lastLayer = tf.matmul(lstmOutput, weights) + biases