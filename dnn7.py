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
import time

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
        if self.data:
            return
        full_data = pd.read_csv(fn, sep=";")
        full_data = full_data.dropna()
        print( full_data.shape )
        #
        self.num_input  = 400
        self.data = full_data.iloc[ :, 3:403]
        #self.data = full_data.iloc[ :, 5:405]
        self.data_rows = int(self.data.shape[0])
        self.train = self.data.iloc[ 0:60000, : ]
        print( self.train.shape )
        self.train_rows = int(self.train.shape[0])
        self.test  = self.data.iloc[ 60000:61000, : ]
        print( self.test.shape )
        self.test_rows = int(self.test.shape[0])
        #
        # labels...
        self.labels = full_data.iloc[ :, 404]
        #self.labels = full_data.iloc[ :, 406]
        #print( self.labels.head() )
        # encode, one hot, etc?
        values = self.labels.unique()
        print( values )
        self.num_output = len( values )
        #
        print(values)
        # Labels to integers
        self.label_encoder   = LabelEncoder()
        self.integer_encoded = self.label_encoder.fit_transform( self.labels )
        print( self.integer_encoded )
        # and integers to onehot
        onehot_encoder  = OneHotEncoder(sparse=False)
        self.integer_encoded = self.integer_encoded.reshape( len(self.integer_encoded), 1 )
        self.onehot_encoded = onehot_encoder.fit_transform( self.integer_encoded )
        #print( self.onehot_encoded[0:20] )
        print( "onehot shape", self.onehot_encoded.shape )
        self.train_labels = self.onehot_encoded[ 0:60000 ]
        self.test_labels  = self.onehot_encoded[ 60000:61000 ]
        # invert examples
        #inverted = self.label_encoder.inverse_transform([np.argmax(self.onehot_encoded[0, :])])
        #print(inverted)
        #inverted = self.label_encoder.inverse_transform([1])
        #print(inverted)
        #sys.exit(1)
    def train_next_batch(self, n):
        if self.train_batch_idx + n > self.train_rows:
            n = self.train_rows - self.train_batch_idx
        start = self.train_batch_idx
        end   = self.train_batch_idx + n
        self.train_batch_idx = end
        #print( start, end )
        #return self.train.iloc[ start:end, : ].values, []
        return self.train.iloc[ start:end, : ].values, self.train_labels[ start:end ]
    def test_next_batch(self, n):
        if n == 0:
            self.test_batch_idx = 0
            return self.test.values, self.test_labels
        if self.test_batch_idx + n > self.test_rows:
            n = self.test_rows - self.test_batch_idx #PJB FIXME wrong kind of reset
        start = self.test_batch_idx
        end   = self.test_batch_idx + n
        self.test_batch_idx = end
        #print( start, end )
        return self.test.iloc[ start:end, : ].values, self.test_labels[ start:end ] 

mydata = DATA()
mydata.read_data_sets("multi_data.csv") #("delta_10000.csv") #("multi_data.csv")

# Training Parameters
learning_rate =     0.0001
num_steps     =  4000
batch_size    =     1
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

#with tf.variable_scope('layer_1'):
#    weights = tf.get_variable('weights1', shape=[num_input, num_hidden_1],
#                              initializer = tf.contrib.layers.xavier_initializer())
#    biases = tf.get_variable('bias1', shape=[num_hidden_1], initializer = tf.zeros_initializer())
#    layer_1_output =  tf.nn.relu(tf.matmul(X, weights) +  biases) 

hl_prev = num_input
output_prev = X
for i, hl in enumerate([2048, 1024, 512, 256]):
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
    training_writer = tf.summary.FileWriter("dnn_logdir/train", graph=sess.graph)
    testing_writer  = tf.summary.FileWriter("dnn_logdir/test", graph=sess.graph)
    saver           = tf.train.Saver(tf.global_variables())

    try:
        print( tf.train.latest_checkpoint("dnn_logdir/", "latest") )
        saver.restore(sess, tf.train.latest_checkpoint("dnn_logdir/", "latest"))
        print( "RESTORED" )
    except ValueError:
        print( "NO RESTORE" )
        pass

    print( "GLOBAL STEP", tf.train.get_global_step() )
    global_step =  tf.train.get_or_create_global_step().eval()
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
            
            saver.save(sess, 'dnn_logdir/my_test_model', global_step=i,
                       latest_filename='latest')
            
    final_train_accuracy = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
    final_test_accuracy  = sess.run(accuracy, feed_dict={X: batch_test_x, Y: batch_test_y})
 
    print( "Final Training Accuracy: {:.2f}".format(final_train_accuracy) )
    print( "Final Testing Accuracy:  {:.2f}".format(final_test_accuracy) )
 
    # Testing
    n = 10
    import warnings
    warnings.filterwarnings("ignore")
    num_correct = 0
    for i in range(n):
        batch_x, batch_y = mydata.test_next_batch(1)

        # eval
        g = sess.run( prediction, feed_dict={X: batch_x, Y: batch_y} )

        #https://stackoverflow.com/questions/50544347/tensorflow-sess-run-returns-same-output-label# ?
        
        # Display
        for j in range(1):
            #print( batch_x[j] )
            y_true = batch_y[j]
            ground_truth  = np.argmax( y_true )
            inverted_true = mydata.label_encoder.inverse_transform([np.argmax(y_true)])
            inverted_true = inverted_true[0]
            y_pred = g[j]
            predicted = np.argmax(y_pred)
            rec_error = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
            inverted = mydata.label_encoder.inverse_transform([np.argmax(y_pred)])[0]
            #y_pred_str = [ "{:5.2f}".format(x) for x in y_pred ]
            foo = " {:6.2f}" * len(y_pred)
            y_pred_str = foo.format(*y_pred)
            marker = " "
            if ground_truth == predicted:
                num_correct += 1
                marker = "*"
            #print( y_true_str )
            print( "{:2d}".format(i), y_true, "|", ground_truth, "|",
                   y_pred_str, "|", predicted, "|",
                   "{:5.2f}".format(rec_error.eval()), marker )
    print( num_correct )
