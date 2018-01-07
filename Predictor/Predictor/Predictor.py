import tensorflow as tf
import numpy as np
import data_reader
import batch_builder

# parse the training data.
training = data_reader.read_annotated_fasta('training.txt')
testing = data_reader.read_annotated_fasta('testing.txt')

# constants
classVectorSize = len(training[0].ClassVector)
vocabSize = len(training[0].SequenceVector[0])
batch_size = 25
cell1Size = 215
cell2Size = 128
trainingIterations = 50000

# create placeholders for the input/output data.
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1024, vocabSize])
y = tf.placeholder(dtype=tf.float32, shape=[batch_size, classVectorSize])

# create my lstm cells
sequenceLength = tf.placeholder(dtype=tf.float32, shape=[batch_size])
cell1 = tf.contrib.rnn.BasicLSTMCell(cell1Size)
cell2 = tf.contrib.rnn.BasicLSTMCell(cell2Size)
multiLayerCell = tf.contrib.rnn.MultiRNNCell(cells=[cell1, cell2])

# this is the operation that works over the lstm cells.
#initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
_, state = tf.nn.dynamic_rnn(dtype=tf.float32, 
                             cell=multiLayerCell, 
                             inputs=x, 
                             sequence_length=sequenceLength)
lstmOutput = state[-1][-1]
outputShape = lstmOutput.get_shape().as_list()

# create fully connected layer which produces the one-hot vector output.
weights = tf.Variable(tf.truncated_normal(shape=[outputShape[-1], classVectorSize], stddev=.1))
biases = tf.Variable(tf.constant(value=.1, shape=[classVectorSize]))
lastLayer = tf.matmul(lstmOutput, weights) + biases

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=lastLayer, labels=y))
optmizer = tf.train.AdamOptimizer(0.0003).minimize(cost)

with tf.Session() as session:

    tf.global_variables_initializer().run(session=session)

    for i in range(0, trainingIterations):

        batch = batch_builder.next_batch(batch_size, 
                                         [(r.SequenceVector, r.SequenceLength) for r in training], 
                                         [r.ClassVector for r in training])

        xVals = [l[0] for l in batch[0]]
        xValLengths = [l[1] for l in batch[0]]
        o, l, w = session.run([optmizer, lastLayer, weights], feed_dict={x : xVals, y : batch[1], sequenceLength : xValLengths})
        
        test_batch = batch_builder.next_batch(batch_size, 
                                         [(r.SequenceVector, r.SequenceLength) for r in testing], 
                                         [r.ClassVector for r in testing])
        xVals = [l[0] for l in test_batch[0]]
        xValLengths = [l[1] for l in test_batch[0]]

        l = session.run([lastLayer], feed_dict={x : xVals, y : test_batch[1], sequenceLength : xValLengths})

        same = 0
        for c in range(0, len(l[0])):
            if np.argmax(l[0][c]) == np.argmax(test_batch[1][c]):
                same = same + 1

        with open("results.csv", "a") as f:
            f.write(str(same) + '\n')

        print(str(same / len(l[0])))

