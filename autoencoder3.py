""" Auto Encoder Example.

Build a 2 layers auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# make this a data class
class DATA():
    def __init__(self):
        self.epochs          = 0
        self.data            = None
        self.data_rows       = 0
        self.train_rows      = 0
        self.test_rows       = 0
        self.train_batch_idx = 0
        self.test_batch_idx  = 0
    def read_data_sets(self, fn):
        if self.data:
            return
        full_data = pd.read_csv(fn, sep=";")
        full_data = full_data.dropna()
        print( full_data.shape )
        self.data = full_data.iloc[ :, 3:403]
        self.data_rows = int(self.data.shape[0])
        self.train = self.data.iloc[ 0:60000, : ]
        print( self.train.shape )
        self.train_rows = int(self.train.shape[0])
        self.test  = self.data.iloc[ 60000:61000, : ]
        print( self.test.shape )
        self.test_rows = int(self.test.shape[0])
    def train_next_batch(self, n):
        if self.train_batch_idx + n > self.train_rows:
            n = self.train_rows - self.train_batch_idx
        start = self.train_batch_idx
        end   = self.train_batch_idx + n
        self.train_batch_idx = end
        #print( start, end )
        return self.train.iloc[ start:end, : ].values, []
    def test_next_batch(self, n):
        if self.test_batch_idx + n > self.test_rows:
            n = self.test_rows - self.test_batch_idx
        start = self.test_batch_idx
        end   = self.test_batch_idx + n
        self.test_batch_idx = end
        #print( start, end )
        return self.test.iloc[ start:end, : ].values, []

# Import MNIST data
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
print( mnist )
'''
mydata = DATA()
mydata.read_data_sets("multi_data.csv")

# Training Parameters
learning_rate =     0.001
num_steps     =  3000
batch_size    =   256

display_step     =  500
examples_to_show =   10

# Network Parameters
num_hidden_1 = 256 #256 # 1st layer num features
num_hidden_2 = 128 #128 # 2nd layer num features 
num_hidden_3 = 64  #(the latent dim)
num_input    = 20*20 #784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

def encoder(x):
    with tf.variable_scope('encoder', reuse=False):
        with tf.variable_scope('layer_1', reuse=False):
            w1 = tf.Variable(tf.random_normal([num_input, num_hidden_1]), name="w1")
            b1 = tf.Variable(tf.random_normal([num_hidden_1]), name="b1")
            # Encoder Hidden layer with sigmoid activation #1
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w1), b1))
            tf.summary.histogram("histogram-w1", w1)
            tf.summary.histogram("histogram-b1", b1)

        with tf.variable_scope('layer_2', reuse=False):
            w2 = tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2]), name="w2")
            b2 = tf.Variable(tf.random_normal([num_hidden_2]), name="b2")
            # Encoder Hidden layer with sigmoid activation #2
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2))
            tf.summary.histogram("histogram-w2", w2)
            tf.summary.histogram("histogram-b2", b2)
        
        with tf.variable_scope('layer_3', reuse=False):
            w2 = tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3]), name="w2")
            b2 = tf.Variable(tf.random_normal([num_hidden_3]), name="b2")
            # Encoder Hidden layer with sigmoid activation #2
            layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, w2), b2))
            tf.summary.histogram("histogram-w2", w2)
            tf.summary.histogram("histogram-b2", b2)    
            return layer_3

# Building the decoder
def decoder(x):
    with tf.variable_scope('decoder', reuse=False):
        with tf.variable_scope('layer_1', reuse=False):
            w1 = tf.Variable(tf.random_normal([num_hidden_3, num_hidden_2]), name="w1")
            b1 = tf.Variable(tf.random_normal([num_hidden_2]), name="b1")
            # Decoder Hidden layer with sigmoid activation #1
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w1), b1))
            tf.summary.histogram("histogram-w1", w1)
            tf.summary.histogram("histogram-b1", b1)

        with tf.variable_scope('layer_2', reuse=False):
            w1 = tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1]), name="w1")
            b1 = tf.Variable(tf.random_normal([num_hidden_1]), name="b1")
            # Decoder Hidden layer with sigmoid activation #1
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w1), b1))
            tf.summary.histogram("histogram-w1", w1)
            tf.summary.histogram("histogram-b1", b1)
            
        with tf.variable_scope('layer_3', reuse=False):
            w2 = tf.Variable(tf.random_normal([num_hidden_1, num_input]), name="w2")
            b2 = tf.Variable(tf.random_normal([num_input]), name="2")
            # Decoder Hidden layer with sigmoid activation #2
            layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, w2), b2))
            tf.summary.histogram("histogram-w2", w2) 
            tf.summary.histogram("histogram-b2", b2)           
            return layer_3
        
# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss      = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
tf.summary.scalar("loss", loss)
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

check = tf.add_check_numerics_ops()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    #print( tf.trainable_variables() )
    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder'):
        print( i.name )   # i.name if you want just a name
        #if i.name == "encoder/layer_1/w1:0":
        #    print( i.eval() )

    
    # Save Graph for TensorBoard representation
    # from https://github.com/Octadero/rada/blob/master/Autoencoder/autoencoder.py
    summary_writer = tf.summary.FileWriter("autoencoder_logdir/", graph=sess.graph)
    summaries      = tf.summary.merge_all()
    saver          = tf.train.Saver(tf.global_variables())
    
    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        #batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x, _ = mydata.train_next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        #c, _, l = sess.run([check, optimizer, loss], feed_dict={X: batch_x})
        _, l, summ = sess.run([optimizer, loss, summaries], feed_dict={X: batch_x})
        #_, l, yp = sess.run([optimizer, loss, y_pred], feed_dict={X: batch_x})#pjb
        # sess.run([optimizer,loss], feed_dict={X:x,Y:y})

        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))
            summary_writer.add_summary(summ, global_step=i)
            #saver.save(sess, "autoencoder_logdir/" + 'model.ckpt', i)

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig  = np.empty((20 * n, 20 * n))
    canvas_recon = np.empty((20 * n, 20 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mydata.test_next_batch(n)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Display original images
        for j in range(n):
            #print( batch_x[j] )
            y_true = batch_x[j]
            y_pred = g[j]
            rec_error = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
            print( rec_error.eval() )
            # Draw the original digits
            canvas_orig[i * 20:(i + 1) * 20, j * 20:(j + 1) * 20] = \
                batch_x[j].reshape([20, 20])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 20:(i + 1) * 20, j * 20:(j + 1) * 20] = \
                g[j].reshape([20, 20])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper")#, cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper") #, cmap="gray")
    plt.show()
