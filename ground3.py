#!/usr/bin/env python3
#
# (c) PJB 2018
# DNN from scratch
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import tensorflow as tf
print( tf.__version__)

import sys, time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)

def dnn_model_fn(features, labels, mode, params):
  """Model function for DNN."""
  # Input Layer
  input_layer = tf.feature_column.input_layer(features, params['feature_columns'])
  
  print( "------->", mode )
  
  hidden0 = tf.layers.dense(input_layer, 12, tf.nn.relu)
  hidden1 = tf.layers.dense(hidden0, 6, tf.nn.relu)
  dropout = tf.layers.dropout(inputs=hidden1, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)
  logits  = tf.layers.dense(dropout, 2, tf.nn.relu) #output, 2 classes

  predictions = {
    # Generate predictions (for PREDICT and EVAL mode)
    "classes": tf.argmax(input=logits, axis=1, name="classes"),
    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    # `logging_hook`.
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    "logits":logits
  }

  print( "predictions =", predictions )
    
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  #tf.summary.scalar("random", np.random.randint(0,10))
  #tf.summary.scalar("logit0", logits[0,0])
  
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
    train_op  = optimizer.minimize(
      loss=loss,
      global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  
  # Add evaluation metrics (for EVAL mode)
  # These end up in tensorboard
  eval_metric_ops = {
    "accuracy":  tf.metrics.accuracy(labels=labels, predictions=predictions["classes"]),
    "precision": tf.metrics.precision(labels=labels, predictions=predictions["classes"]),
    "recall":    tf.metrics.recall(labels=labels, predictions=predictions["classes"]),
    "FN":        tf.metrics.false_negatives(labels=labels, predictions=predictions["classes"]),
    "FP":        tf.metrics.false_positives(labels=labels, predictions=predictions["classes"]),
    "TN":        tf.metrics.true_negatives(labels=labels, predictions=predictions["classes"]),
    "TP":        tf.metrics.true_positives(labels=labels, predictions=predictions["classes"])
    # https://stackoverflow.com/questions/45643809/custom-eval-metric-ops-in-estimator-in-tensorflow
    #"ME":        foofun( labels=labels, predictions=predictions["classes"] )
  }
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  # Load training and eval data

  BATCHSIZE = 3
  
  df = pd.read_csv( "h100.csv", sep=";")
  
  # making fake data using numpy
  #train_data = ( df.iloc[ 0:90,  4:404], df.iloc[ 0:90,  404:405] )
  #test_data  = ( df.iloc[90:100, 4:404], df.iloc[90:100, 404:405] )
  train_data = ( df.iloc[ 0:10,  4:14], df.iloc[ 0:10,  404:405] ) # last is labels
  test_data  = ( df.iloc[90:100, 4:14], df.iloc[90:100, 404:405] )
  
  my_feature_columns = []
  for key in train_data[0].keys():
    print( "Feature column", key )
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

  def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    inputs = (dict(features), labels) # to convert dataframe to suitable format
    dataset = tf.data.Dataset.from_tensor_slices(inputs).shuffle(buffer_size=20).batch(BATCHSIZE).repeat()
    
    # Build the Iterator, and return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

  def eval_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    inputs = (dict(features), labels) # note batch 1, and steps 10/batch 1 in eval, to get all items
    dataset = tf.data.Dataset.from_tensor_slices(inputs).batch(1) #.repeat()
    
    # Build the Iterator, and return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

  my_checkpointing_config = tf.estimator.RunConfig(
    #save_checkpoints_secs = 1*60,  # Save checkpoints every minute (conflicts with save_checkpoints_steps)
    keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
    log_step_count_steps=10000,
    save_summary_steps=1000,
    save_checkpoints_steps=10000
  )

  # Create the Estimator
  dnn_classifier = tf.estimator.Estimator(
    model_fn=dnn_model_fn,
    params={'feature_columns': my_feature_columns},
    model_dir="./ground3model",
    config=my_checkpointing_config)

  # Set up logging for predictions, classes give a class (0, 1 or 2), the probs are the 3 outputs
  tensors_to_log = {"classes":"classes"} # or {"probabilities": "softmax_tensor"}
  logging_hook   = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1000)
  
  # Train the model
  
  dnn_classifier.train(
    #input_fn=train_input_fn,
    input_fn=lambda:train_input_fn(train_data[0], train_data[1], BATCHSIZE),
    steps=FLAGS.max_steps,
    hooks=[logging_hook])

  eval_results = dnn_classifier.evaluate(
    #input_fn=eval_input_fn
    input_fn=lambda:eval_input_fn(test_data[0], test_data[1], 1),
    steps=10
  )
  print(eval_results)
  try:
    f1 = 2 * (eval_results["precision"] * eval_results["recall"]) / (eval_results["precision"] + eval_results["recall"])
    print( "f1", f1 )
  except:
    pass

if __name__ == "__main__":
  flags = tf.app.flags
  FLAGS = flags.FLAGS

  flags.DEFINE_integer('batch_size', 10, 'Batch size.')
  flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
  flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
  
  tf.app.run()

  # for x in 0 1 2 3 4 5 6 7 8 9;do echo $x;python3 ground1.py --max_steps 10000;done
  
