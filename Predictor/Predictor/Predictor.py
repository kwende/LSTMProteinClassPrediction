import tensorflow as tf
import numpy as np
import data_reader

ret = data_reader.read_annotated_fasta('training.txt')

batch_size = 2
lstmCell1HiddenLayerSize = 215
lstmCell2HiddenLayerSize = 128

x = tf.placeholder(dtype=tf.float32, shape=[batch_size, None])