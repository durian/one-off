""" 
DNN

Better with signoid instead of relu? and original loss funxion?

"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument( '-b', "--batch_size", type=int, default=16, help='Batch size' )
parser.add_argument( '-s', "--steps", type=int, default=1000, help='Number of steps to train' )
parser.add_argument( '-L', "--log_dir", type=str, default="dnn8_logdir", help='Log and model directory, with /' )
parser.add_argument( '-d', "--training_data", type=str, default="multi_data.csv", help='Training data' )
parser.add_argument( '-S', "--save_data", action='store_true', default=False, help='Save train and test sets' )
parser.add_argument( '-r', "--ratio", type=float, default="0.25", help='Ratio of test data from all data' )
parser.add_argument( '-l', "--learning_rate", type=float, default="0.001", help='Learning rate' )
parser.add_argument( '-R', "--random_state", type=int, default=42, help='Random state' )
parser.add_argument( '-H', "--hidden_layers", type=str, default="512,256,128", help='Network architecture' )
args = parser.parse_args()

if not args.log_dir[-1] == "/":
    args.log_dir += "/"
    
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
        self.num_input  = None
        self.num_output = None
    def read_data_sets(self, fn):
        # save these, and have a "restore datasets" option?
        if self.data:
            return
        full_data = pd.read_csv(fn, sep=";")
        full_data = full_data.dropna()
        #full_data = full_data.loc[(full_data!=0).any(axis=1)]
        print( "Full data shape", full_data.shape )
        #
        # X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
        #
        self.data      = full_data.iloc[ :, 3:403]  # X
        self.labels    = full_data.iloc[ :, 404]    # y
        self.num_input = self.data.shape[1]
        #
        if True:
            self.values = self.labels.unique()
            print( "Unique values", self.values )
            self.num_output = len( self.values )
            # Labels to integers
            self.label_encoder   = LabelEncoder()
            self.integer_encoded = self.label_encoder.fit_transform( self.labels )
            print( self.integer_encoded )
            # and integers to onehot
            self.onehot_encoder  = OneHotEncoder(sparse=False)
            self.integer_encoded = self.integer_encoded.reshape( len(self.integer_encoded), 1 )
            self.onehot_encoded  = self.onehot_encoder.fit_transform( self.integer_encoded )
            print( "Onehot shape", self.onehot_encoded.shape )
            self.labels = self.onehot_encoded # now numpy array!
        else:
            self.labels = self.labels.values # now numpy array!
            self.num_output = 1
        #
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( self.data.values,
                                                                                 self.labels,
                                                                                 test_size=args.ratio,
                                                                                 random_state=args.random_state)
        print( "X_train", self.X_train[0, 0:10] )
        print( "y_train", self.y_train[0] )
        print( self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape )
        #
        if args.save_data:
            print( "Saving data in dnn_X/y..." )
            np.savetxt("dnn8_X_train.csv", self.X_train, delimiter=";", fmt="%d")
            np.savetxt("dnn8_y_train.csv", self.y_train, delimiter=";", fmt="%d")
            np.savetxt("dnn8_X_test.csv",  self.X_test,  delimiter=";", fmt="%d")
            np.savetxt("dnn8_y_test.csv",  self.y_test,  delimiter=";", fmt="%d")
        #
        self.data_rows = int(self.data.shape[0])
        self.test_rows = int(self.labels.shape[0])
        #sys.exit(1)
    def train_next_batch(self, n):
        # maybe use: from sklearn.model_selection import ShuffleSplit
        if self.train_batch_idx + n > self.train_rows:
            n = self.train_rows - self.train_batch_idx #PJB FIXME wrong kind of reset
        start = self.train_batch_idx
        end   = self.train_batch_idx + n
        self.train_batch_idx = end
        return shuffle( self.X_train[ start:end], self.y_train[ start:end ], random_state=args.random_state )
        #return self.X_train[ start:end], self.y_train[ start:end ]
    def test_next_batch(self, n):
        if n == 0:
            self.test_batch_idx = 0
            return self.X_test, self.y_test
        if self.test_batch_idx + n > self.test_rows:
            n = self.test_rows - self.test_batch_idx #PJB FIXME wrong kind of reset
        start = self.test_batch_idx
        end   = self.test_batch_idx + n
        self.test_batch_idx = end
        return shuffle( self.X_test[ start:end ], self.y_test[ start:end ], random_state=args.random_state )
        #return self.X_test[ start:end ], self.y_test[ start:end ] 

mydata = DATA()
mydata.read_data_sets( args.training_data )

# Training Parameters
learning_rate =  args.learning_rate
num_steps     =  args.steps
batch_size    =  args.batch_size
display_step  =  1000

# Network Parameters
num_input    = mydata.num_input
num_output   = mydata.num_output #no... one hot...?

# tf Graph input 
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_output], name="Y")

# Define the layers
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape= (None, num_input))

regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

hl_prev = num_input
output_prev = X
hidden = [ int(x) for x in args.hidden_layers.split(",")]
for i, hl in enumerate(hidden):
    i_str = "{:02d}".format(i)
    print( i_str, hl_prev )
    with tf.variable_scope('layer_'+i_str):
        weights = tf.get_variable('weights'+i_str, shape=[hl_prev, hl],
                                  initializer = tf.contrib.layers.xavier_initializer(),
                                  #regularizer=regularizer
        )
        biases = tf.get_variable('bias'+i_str, shape=[hl], initializer = tf.zeros_initializer())
        layer_output = tf.nn.relu(tf.matmul(output_prev, weights) + biases)
        layer_output = tf.nn.dropout(layer_output, 0.8)
        output_prev = layer_output
        hl_prev = hl
        
'''
# https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
# Layer 2 with BN, using Tensorflows built-in BN function
w2_BN = tf.Variable(w2_initial)
z2_BN = tf.matmul(l1_BN,w2_BN)
batch_mean2, batch_var2 = tf.nn.moments(z2_BN,[0])
scale2 = tf.Variable(tf.ones([100]))
beta2 = tf.Variable(tf.zeros([100]))
BN2 = tf.nn.batch_normalization(z2_BN,batch_mean2,batch_var2,beta2,scale2,epsilon)
l2_BN = tf.nn.sigmoid(BN2)
'''

with tf.variable_scope('output'):
    weights = tf.get_variable('weights_out', shape=[hl, num_output],
                              initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('bias_out', shape=[num_output], initializer = tf.zeros_initializer())
    prediction =  tf.matmul(layer_output, weights) + biases
 
with tf.variable_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = prediction))

global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.variable_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)
 
with tf.variable_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(Y, axis=1), tf.argmax(prediction, axis=1) )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
# Logging results
with tf.variable_scope("logging"):
    tf.summary.scalar('current_cost', cost)
    tf.summary.scalar('current_accuracy', accuracy)
    #tf.summary.scalar('global_step', global_step)
    #foo = tf.get_default_graph().get_tensor_by_name("layer_1/weights1:0")
    #tf.summary.histogram("histogram-w1", foo) #layer_1/weights1:0)
    #tf.summary.histogram("histogram-b1", bias1)
    summary = tf.summary.merge_all()


check = tf.add_check_numerics_ops()
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
# config=tf.ConfigProto(log_device_placement=True)
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    #print( tf.trainable_variables() )
    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        print( i.name, i.shape )
        if "weights" in i.name:
            tf.summary.histogram(i.name, i) 
    summary = tf.summary.merge_all()
    
    # Save Graph for TensorBoard representation
    # from https://github.com/Octadero/rada/blob/master/Autoencoder/autoencoder.py
    training_writer = tf.summary.FileWriter(args.log_dir+"train", graph=sess.graph)
    testing_writer  = tf.summary.FileWriter(args.log_dir+"test", graph=sess.graph)
    saver           = tf.train.Saver(tf.global_variables())

    try:
        print( tf.train.latest_checkpoint(args.log_dir, "latest") )
        saver.restore(sess, tf.train.latest_checkpoint(args.log_dir, "latest"))
        print( "RESTORED" )
    except ValueError:
        print( "NO RESTORE" )
        pass

    global_step = tf.train.get_or_create_global_step().eval()
    print( "GLOBAL STEP", global_step )
    #tf.train.get_global_step().eval()
    
    # Training
    time_start = time.perf_counter()
    for i in range(global_step, num_steps+global_step):
        # Prepare Data
        # Get the next batch of data
        #batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x, batch_y = mydata.train_next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        #c, _, l = sess.run([check, optimizer, loss], feed_dict={X: batch_x})

        #_, l, summ = sess.run([optimizer, loss, summaries], feed_dict={X: batch_x, Y: batch_y})
        sess.run(optimizer, feed_dict={X:batch_x, Y:batch_y})

        
        # Every 5 training steps, log our progress
        if i % display_step == 0:
            batch_test_x, batch_test_y = mydata.test_next_batch(0)
            
            training_cost, training_summary = sess.run([cost, summary], feed_dict={X: batch_x, Y: batch_y})
            testing_cost, testing_summary = sess.run([cost, summary], feed_dict={X: batch_test_x, Y: batch_test_y})
            
            #accuracy
            train_accuracy = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
            test_accuracy = sess.run(accuracy, feed_dict={X: batch_test_x, Y: batch_test_y})
            #roc = sess.run(roc_score, feed_dict={X: batch_test_x, Y: batch_test_y})
            #roc_score = tf.metrics.auc( tf.argmax(Y, axis=1), tf.argmax(prediction, axis=1) )
            
            time_now = time.perf_counter()
            time_diff = time_now - time_start
            time_start = time_now
            print( i, "{:.2f}".format(time_diff), training_cost, testing_cost, train_accuracy, test_accuracy )
            
            training_writer.add_summary(training_summary, i)
            testing_writer.add_summary(testing_summary, i)
            
            saver.save(sess, args.log_dir+"my_test_model", global_step=i, latest_filename="latest")
            
    final_train_accuracy = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
    final_test_accuracy  = sess.run(accuracy, feed_dict={X: batch_test_x, Y: batch_test_y})
 
    print( "Final Training Accuracy: {:.2f}".format(final_train_accuracy) )
    print( "Final Testing Accuracy:  {:.2f}".format(final_test_accuracy) )
 
    # Testing, again, with output
    import warnings
    warnings.filterwarnings("ignore")
    num_correct = 0
    batch_x, batch_y = mydata.test_next_batch(-1)

    g = sess.run( prediction, feed_dict={X: batch_x, Y: batch_y} )

    #https://stackoverflow.com/questions/50544347/tensorflow-sess-run-returns-same-output-label# ?

    ground_truth = [ np.argmax(y) for y in batch_y ]  # Only if onehot encoded...
    y_pred       = g
    predicted    = [ np.argmax(p) for p in y_pred ]
    q = confusion_matrix( ground_truth, predicted )
    print(q)
    '''
    colsums = [0,0,0,0,0]
    for l in q:
        print( l, sum(l) )
        colsums += l
    print( colsums )
    '''
    '''
    import matplotlib.pyplot as plt
    plt.imshow(q)
    plt.show(block=True)
    '''
    sys.exit(1)
    
    # Display
    for j in range( len(batch_x) ):
        #print( batch_x[j] )
        y_true = batch_y[j]
        ground_truth  = np.argmax( y_true )  # Only if onehot encoded...
        inverted_true = mydata.label_encoder.inverse_transform([np.argmax(y_true)])
        inverted_true = inverted_true[0]
        y_pred = g[j]
        predicted = np.argmax(y_pred)
        rec_error = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        inverted = mydata.label_encoder.inverse_transform([np.argmax(y_pred)])[0]
        #y_pred_str = [ "{:5.2f}".format(x) for x in y_pred ]
        foo = " {:8.2f}" * len(y_pred)
        y_pred_str = foo.format(*y_pred)
        marker = " "
        if ground_truth == predicted:
            num_correct += 1
            marker = "*"
        #print( y_true_str )
        print( "{:2d}".format(j), y_true, "|", ground_truth, "|",
               y_pred_str, "|", predicted, "|",
               "{:8.2f}".format(rec_error.eval()), marker )
    print( num_correct )
