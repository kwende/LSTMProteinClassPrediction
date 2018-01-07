import tensorflow as tf
import numpy as np
import data_reader
import batch_builder

# parse the training data.
ret = data_reader.read_annotated_fasta('training.txt')

# constants
classVectorSize = len(ret[0].ClassVector)
vocabSize = len(ret[0].SequenceVector[0])
batch_size = 3
cell1Size = 215
cell2Size = 128
trainingIterations = 50000

# create placeholders for the input/output data.
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 700, vocabSize])
y = tf.placeholder(dtype=tf.float32, shape=[batch_size, classVectorSize])

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

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=lastLayer, labels=y))
optmizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as session:

    tf.global_variables_initializer().run(session=session)

    for i in range(0, trainingIterations):

        batch = batch_builder.next_batch(batch_size, 
                                         [r.SequenceVector for r in ret], 
                                         [r.ClassVector for r in ret])

        o, l, w = session.run([optmizer, lastLayer, weights], feed_dict={x : batch[0], y : batch[1]})
        
        print(str(np.argmax(l, 1)) + "vs" + str(np.argmax(batch[1], 1)))

