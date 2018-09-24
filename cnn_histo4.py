#!/usr/bin/env python3
#
# (c) PJB 2018
# CNN from scratch
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import tensorflow as tf
print( tf.__version__ )

import sys, time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import datetime
now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M:%S"))

tf.logging.set_verbosity(tf.logging.INFO)

# -----------------------------------------------------------
# CNN model
# -----------------------------------------------------------

def cnn_model_fn(features, labels, mode, params):
  """Model function for CNN."""

  print( params )
  
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 20, 20, 1]) #20x20 histograms
  print( "input_layer", input_layer )
  
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  print( "conv1", conv1 )
  # Tensor("conv2d/Relu:0", shape=(100, 20, 20, 32), dtype=float32)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  print( "pool1", pool1 )
  # Tensor("max_pooling2d/MaxPool:0", shape=(100, 10, 10, 32), dtype=float32)
  
  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  print( "conv2", conv2 )
  # Tensor("conv2d_2/Relu:0", shape=(100, 10, 10, 64), dtype=float32)
  
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  print( "pool2", pool2, np.prod(pool2.shape[1:]) )
  # Tensor("max_pooling2d_2/MaxPool:0", shape=(100, 5, 5, 64), dtype=float32)
  
  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 64])
  print( "pool2_flat", pool2_flat )
  # Tensor("Reshape_1:0", shape=(100, 1600), dtype=float32)

  dense1 = tf.layers.dense(inputs=pool2_flat, units=2048, activation=tf.nn.relu)
  print( "dense1", dense1 )

  dropout1 = tf.layers.dropout(
      inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  print( "dropout1", dropout1 )

  dense2 = tf.layers.dense(inputs=dropout1, units=2048, activation=tf.nn.relu)
  print( "dense2", dense2 )

  dropout2 = tf.layers.dropout(
      inputs=dense2, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
  print( "dropout2", dropout2 )


  # Logits Layer
  logits = tf.layers.dense(inputs=dropout2, units=2)
  print( "logits", logits )
  # Tensor("dense_2/BiasAdd:0", shape=(100, 2), dtype=float32)


  predictions = {
    # Generate predictions (for PREDICT and EVAL mode)
    "classes": tf.argmax(input=logits, axis=1),
    # Add softmax_tensor to the graph. It is used for PREDICT and by the logging_hook.
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    #"pool2": pool2,
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
    "accuracy":  tf.metrics.accuracy(  labels=labels, predictions=predictions["classes"] ),
    "precision": tf.metrics.precision( labels=labels, predictions=predictions["classes"] ),
    "recall":    tf.metrics.recall(    labels=labels, predictions=predictions["classes"] ),
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

   
# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main(unused_argv):
  # Load training and eval data

  global sess
  
  with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print( sess )

    # All data
    #df = pd.read_csv( "multi_data.csv", sep=";")
    df = pd.read_csv( "mirrored.csv", sep=";")

    the_label = "All_Fault_in_3_months"

    # Change labels to ints. This creates a integer mapping to string values:
    lbls, uniqs = pd.factorize(df[the_label])
    print( lbls, uniqs )
    #df[the_label] = lbls # Put integers instead of the label strings
    ##print( df.head() )

    #df_pos   = df_train[ ( df_train[the_label] == 1 ) ] # positive example of break down
    #df_neg   = df_train[ ( df_train[the_label] == 0 ) ]
    #df_all   = df_pos[0:8000].append( df_neg[0:8000] )
    #print( "Pos/neg/all", df_pos.shape, df_neg.shape, df_all.shape )

    df_train     = df[ ( df["PARTITIONNING"] == "1_Training" ) ]
    train_data   = df_train.iloc[ :,  3:403]
    train_labels = df_train.loc[ :, the_label]

    if FLAGS.balanced:
      df_pos       = df_train[ ( df_train[the_label] == 1 ) ] # positive example of break down
      df_neg       = df_train[ ( df_train[the_label] == 0 ) ]
      df_balanced  = df_pos[0:6000].append( df_neg[0:6000] ) # or twice as many healthy ones?
      train_data   = df_balanced.iloc[ :,  3:403]
      train_labels = df_balanced.loc[ :, the_label]

    df_test     = df[ ( df["PARTITIONNING"] == "2_Testing" ) ]
    test_data   = df_test.iloc[:, 3:403]
    test_labels = df_test.loc[:, the_label] 

    df_validate     = df[ ( df["PARTITIONNING"] == "3_Validation" ) ]
    validate_data   = df_validate.iloc[ :, 3:403]
    validate_labels = df_validate.loc[ :, the_label] 

    print( train_data.shape )
    print( test_data.shape )
    print( validate_data.shape )
    
    with open( "progress_h4.txt", 'a' ) as f:
      f.write( "\nSTART:\n" )
      if FLAGS.balanced:
        f.write( "Balanced data\n" )
        f.write( "pos/neg: "+str(df_pos.shape)+"/"+str(df_neg.shape)+"\n" )
      f.write( "train:    "+str(train_data.shape)+"\n" )
      f.write( "test:     "+str(test_data.shape)+"\n" )
      f.write( "validate: "+str(validate_data.shape)+"\n" )
      for key, value in FLAGS.__flags.items():
        pass #doesnt work on 1.4
      #  if key not in ["h", "help", "helpfull", "helpshort"]:
      #    f.write( key+"="+str(value.value)+"\n" )
      #f.write( "\nLabels:\n" )
      #f.write( "Train:\n"+str(train_data[1].value_counts())+"\n" )
      #f.write( "Test:\n"+str(test_data[1].value_counts())+"\n" )
      #f.write( "Validate:\n"+str(validate_data[1].value_counts())+"\n" )
      f.write( "\n" )

    # Make numpy arrays, correct type
    train_data = train_data.values.astype(np.float32)
    print( train_data )
    train_labels = train_labels.values #.astype(np.int)
    print( train_data )
    #
    eval_data = test_data.values.astype(np.float32)
    print( eval_data )
    eval_labels = test_labels.values
    print( eval_labels )
    #
    validate_data = validate_data.values.astype(np.float32)
    print( validate_data )
    validate_labels = validate_labels.values
    print( validate_labels )

    cnn_train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=FLAGS.batch_size,
      num_epochs=None,
      shuffle=True
    )
    cnn_eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )
    cnn_validate_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": validate_data},
        y=validate_labels,
        num_epochs=1,
        shuffle=False
    )

    my_checkpointing_config = tf.estimator.RunConfig(
      keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
      log_step_count_steps=1000,
      save_summary_steps=10000,
      save_checkpoints_steps=10000
    )

    # Create the Estimator
    cnn_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn,
      model_dir=FLAGS.model_dir,
      params={"my_session":"none"}
    )

    # Set up logging for predictions
    tensors_to_log = {
      "probabilities": "softmax_tensor",
      #"dense_1": "dense_1/kernel:0",
      #"dense_2": "dense_2/BiasAdd:0",
      #"pool2flat": "Reshape_1:0",
      #"output": "dense_2/BiasAdd:0",
    }
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=10000)

    # -----------------------------------------------------------
    # Train the model
    # -----------------------------------------------------------

    # Dataframe with predictions for each iteration
    res_fn = time.strftime("results_"+"%Y%m%d_%H%M%S"+".csv")
    res = pd.DataFrame()
    res["ground"] = test_labels
    with open( "progress_h4.txt", 'a' ) as f:
      f.write( "CSV in: "+res_fn+"\n" )
    res.to_csv(res_fn, sep=";" )

    for i in range(0, int(FLAGS.iterations)):
      with open( "progress_h4.txt", 'a' ) as f:
        f.write( "Iteration:"+str(i)+"\n" )

        print( "------> Iteration", i )
        now = datetime.datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        f.write( now.strftime("%Y-%m-%d %H:%M:%S")+"\n" )
        f.flush()

      cnn_classifier.train(
        input_fn=cnn_train_input_fn,
        steps=FLAGS.max_steps,
        hooks=[logging_hook]
      )

      #tf.get_default_graph()
      #kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1/kernel') #[0]
      #print( kernel )
      #sys.exit(1)

      eval_results = cnn_classifier.evaluate(
        input_fn=cnn_eval_input_fn
      )
      print(eval_results)
      with open( "progress_h4.txt", 'a' ) as f:
        f.write( "Evaluation (testset):\n" )
        f.write( str(eval_results)+"\n" )


      pred_results = cnn_classifier.predict(
        input_fn=cnn_eval_input_fn
      )
      print(pred_results)

      tot  = 0
      corr = 0
      pred_list = []
      label_list = []
      with open( "progress_h4.txt", 'a' ) as f:
        for ground, x in zip(test_labels, pred_results):
          #print( str(x) )
          the_class = int(x["classes"])
          if tot < 10:
            print( tot, ground, the_class, x["probabilities"] )
            f.write( str(tot)+" "+str(ground)+" "+str(the_class)+" "+str(x["probabilities"])+"\n" )
          if ground == the_class:
            corr += 1
          tot += 1
          pred_list.append( the_class )
          label_list.append( ground )
        print( tot, corr, corr/tot*100 )
        f.write( str(tot)+" "+str(corr)+" "+str(corr/tot*100)+"\n" )

        cm = confusion_matrix( label_list, pred_list )
        #cm = [[round(val/tot, 2) for val in sublst] for sublst in cm]
        #cm = [[round(val, 2) for val in sublst] for sublst in cm]
        out = ' '.join(['{:5d}'.format(num) for num in uniqs.values])
        print( " ", out )
        f.write( "  "+out+"\n" )
        for u, row in zip(uniqs, cm):
          #out = ' '.join(['{:4.2f}'.format(num) for num in row])
          out = ' '.join(['{:5d}'.format(num) for num in row])
          print( u, out )
          f.write( str(u)+" "+out+"\n" )
        #f.write( str(confusion_matrix( label_list, pred_list ) )+"\n\n" )
        f.flush()

      res[ "I"+str(i) ] = pred_list
      print( res.head() )

      '''
      validate_results = cnn_classifier.evaluate(
        input_fn=cnn_validate_input_fn
      )
      print(validate_results)
      with open( "progress_h4.txt", 'a' ) as f:
        f.write( "\nValidation:\n" )
        f.write( str(validate_results)+"\n\n" )
      '''
      
      print( res.head(4) )
      print( res.tail(4) )
      # write evey iteration
      res.to_csv(res_fn, sep=";" )

    
# -----------------------------------------------------------
# Entry
# -----------------------------------------------------------

#sess = tf.Session()
#print( sess )

if __name__ == "__main__":
  flags = tf.app.flags
  FLAGS = flags.FLAGS

  flags.DEFINE_integer('batch_size',      100,      'Batch size.')
  flags.DEFINE_float(  'learning_rate',     0.0001, 'Initial learning rate.')
  flags.DEFINE_integer('max_steps',     10000,      'Number of steps to run trainer.')
  flags.DEFINE_integer('iterations',        1,      'Number of iterations.')
  flags.DEFINE_string( 'model_dir', "histo4model",   'Model directory.')
  flags.DEFINE_bool( 'balanced', False,   'Equal number of pos/neg examples in training.')

  
  try:
    tf.app.run()
  except AssertionError:
    pass

  # for x in 0 1 2 3 4 5 6 7 8 9;do echo $x;python3 ground6.py;done
  
'''
petber@hh.se@stockholm:~/tf/health/cnn$ nvidia-smi
Fri Sep 14 10:19:00 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.130                Driver Version: 384.130                   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:01:00.0  On |                  N/A |
| 39%   69C    P2   133W / 250W |  10821MiB / 11170MiB |     52%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 108...  Off  | 00000000:02:00.0 Off |                  N/A |
| 24%   41C    P8    16W / 250W |  10622MiB / 11172MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1078      G   /usr/lib/xorg/Xorg                           314MiB |
|    0      2508      G   compiz                                       212MiB |
|    0     10570      G   python                                         2MiB |
|    0     15030      G   /opt/google/chrome/chro                       58MiB |
|    0     27781      C   python3                                    10219MiB |
|    1     27781      C   python3                                    10611MiB |
+-----------------------------------------------------------------------------+

stockholm:
INFO:tensorflow:step = 166501, loss = 0.09626102   (0.337 sec)
INFO:tensorflow:step = 166601, loss = 0.0013853312 (0.334 sec)
INFO:tensorflow:step = 166701, loss = 0.0034902946 (0.335 sec)

bidaf:
INFO:tensorflow:loss = 0.2074248, step = 1001      (3.793 sec)
INFO:tensorflow:loss = 0.01580182, step = 1101     (3.828 sec)
INFO:tensorflow:loss = 0.5218127, step = 1201      (3.834 sec)
'''

