
# python3 dnn_model_loop.py --hidden_units=200x200 --choosen_label=ENGINE_TYPE

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pandas
import tensorflow
import argparse
from os import listdir
from os.path import isfile, join
import datetime
import sys

import dataloader



parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--choosen_label', default="T_CHASSIS", type=str, help='the label to train and evaluate')
parser.add_argument('--hidden_units', default="200x200", type=str, help='Number of hidden units')

"""
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')
parser.add_argument('--hidden_units', default=[10,10], type=string, help='layout for hidden layers')
parser.add_argument('--nr_epochs', default=None, type=int, help='number of epochs')
parser.add_argument('--choosen_label', default=T_CHASSIS, type=string, help='the label to train and evaluate')
parser.add_argument('--label_path', default=Labels/, type=string, help='where one labels file is located')
parser.add_argument('--data_path', default=Data_original/, type=string, help='path to data source files or compressed file')
parser.add_argument('--compressed', default=True, type=boolean, help='if true structured data will be used, false means data source files and a structured file will be produced')
parser.add_argument('--max_nr_nan', default=0, type=int, help='number of nan per row for exclusion')
parser.add_argument('--fixed_sdelection', default=True, type=boolean, help='If true selection is done by truck_date')
"""

args = parser.parse_args()

def main(argv):

        #args = parser.parse_args(argv[1:]) # argv[1:] argv
        #parser.print_help()
        #print(args)
        #sys.exit()
        
        batch_size = args.batch_size # 100
        #print('Batch_size: ' + str(batch_size))
        train_steps = 10000 # 1000 #LOOPED LATER ON
        nr_epochs = None
        hu = [int(x) for x in args.hidden_units.split("x")]
        hidden_units = hu #[200, 200] # [10, 10] [400, 400] [400, 400, 400, 400]
        choosen_label = args.choosen_label #'ENGINE_TYPE' # 'T_CHASSIS' 'COUNTRY' 'ENGINE_TYPE' 'BRAND_TYPE'
        max_nr_nan = 0
        fixed_selection = True

        my_id="l"+choosen_label+"_s"+str(train_steps)+"_h"+args.hidden_units

        label_path = 'Labels/'
        data_path = 'Data_original/' # 'Data_original/' 'Testdata/'
        structured_data_path = 'Compressed/' # 'Compressed_valid_chassis' Compressed/Compressed_single/
        
        #sys.exit()
        
        
        # Label_mapping holds key value pairs where key is the label and value its integer representation
        label_mapping = dataloader.get_valid_labels(label_path, choosen_label) # Labels from labels file only
        
        
        #Get three structured separate dataframes from data sources
        #trainframe, testframe, validationframe = dataloader.loadData(data_path, False, label_mapping, max_nr_nan, fixed_selection)
        trainframe, testframe, validationframe = dataloader.loadData(structured_data_path, True, label_mapping, max_nr_nan, fixed_selection)
        
        # Train model data
        trainset, labels_training, label_mapping, int_labels_train = \
                dataloader.get_model_data(trainframe, label_mapping, choosen_label)
        
        # Test model data
        testset, labels_test, label_mapping, int_labels_test = \
                dataloader.get_model_data(testframe, label_mapping, choosen_label)
        
        # Validate model data
        validationset, labels_validate, label_mapping, int_labels_validate = \
                dataloader.get_model_data(validationframe, label_mapping, choosen_label)

        my_checkpointing_config = tensorflow.estimator.RunConfig(
                #save_checkpoints_secs = 1*60,  # Save checkpoints every minute (conflicts with save_checkpoints_steps)
                keep_checkpoint_max = 4,       # Retain the 4 most recent checkpoints.
                log_step_count_steps=1000,
                save_summary_steps=1000,
                save_checkpoints_steps=1000
        )       

        ### Model training
        my_feature_columns = []
        for key in trainset.keys():
                my_feature_columns.append(tensorflow.feature_column.numeric_column(key=key))

        # The model must choose between x classes.
        print('Number of unique labels, n_classes: ' + str(len(label_mapping)))
        #print('Number of unique trucks, n_classes: ' + str(int_labels.size))
        
        # optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=0.1) ?
        # optimizer = tensorflow.train.AdagradOptimizer(learning_rate=0.1) ?
        # optimizer = tensorflow.train.AdagradDAOptimizer(learning_rate=0.1, global_step= ?) global_step=train_steps?   
        # optimizer = tensorflow.train.AdamOptimizer(learning_rate=0.1) ?
        #opt = tensorflow.train.GradientDescentOptimizer(learning_rate=0.0001)
        my_id=my_id+"_oDFLT_a" #"_oGDO_lr0.0001"

        resultfile = open("Results/"+my_id+"_model_results.txt", "w")
        
        resultfile.write('\nModel training: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n\n')
        resultfile.write('Layer setting: ' + str(hidden_units) + '\n')
        resultfile.write('Train steps: ' + str(train_steps) + '\n')
        resultfile.write('Number epochs: ' + str(nr_epochs) + '\n')
        resultfile.write('Batchsize: ' + str(batch_size) + '\n')
        resultfile.write('Choosen label: ' + choosen_label + '\n')
        resultfile.write('Max_nr_nan: ' + str(max_nr_nan) + '\n')
        resultfile.write('Fixed_selection: ' + str(fixed_selection) + '\n')
        resultfile.flush()

        classifier = tensorflow.estimator.DNNClassifier(
                feature_columns=my_feature_columns,
                hidden_units=hidden_units,
                n_classes=len(label_mapping),
                model_dir="./models/"+my_id,
                config=my_checkpointing_config ) 
                #optimizer=opt)
        #classifier = tensorflow.estimator.DNNClassifier \
        #               (feature_columns=my_feature_columns,hidden_units=hidden_units,n_classes=len(label_mapping))
        #classifier = tensorflow.estimator.DNNClassifier \
        #       (feature_columns=my_feature_columns,hidden_units=hidden_units,n_classes=len(label_mapping), model_dir='Volvo_model')
        
        ### Train the Model.
        # PJB added loop
        for i in range(0,25):
                print('\nModel training\n\n\n')
                #resultfile.write('\nModel training: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n\n\n')
                classifier.train(input_fn=lambda:dataloader.train_input_fn(trainset, int_labels_train, batch_size, nr_epochs), steps=train_steps)

                ### Test the model
                print('\nModel testing\n\n\n')
                resultfile.write('\nModel testing\n\n\n')
                # Evaluate the model.
                eval_result = classifier.evaluate(input_fn=lambda:dataloader.eval_input_fn(testset, int_labels_test, batch_size))
                print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
                resultfile.write('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

                ### Evaluate the model
                print('\nModel evaluation\n\n\n')
                resultfile.write('\nModel evaluation\n\n\n')
                expected = list(label_mapping.keys())
                predictions = classifier.predict(input_fn=lambda:dataloader.eval_input_fn(validationset, labels=None, batch_size=batch_size))
                template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

                predictfile = open("Results/"+my_id+"_predictions.txt", "a")

                for pred_dict, expec in zip(predictions, expected):
                        class_id = pred_dict['class_ids'][0]
                        probability = pred_dict['probabilities'][class_id]
                        #print(template.format(expected[class_id], 100 * probability, expec))
                        resultfile.write('\n')
                        resultfile.write(template.format(expected[class_id], 100 * probability, expec))

                        if str(expected[class_id]) == str(expec):
                                predictfile.write('Loop:'+str(i)+' Percent: ' + str(100 * probability) + '  ' + choosen_label + ': ' + str(expec) + '\n')
                predictfile.close()
        resultfile.write('\n******************************\n')
        resultfile.close()

        
if __name__ == '__main__':
    tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
    tensorflow.app.run(main) # So far only a dummy arguments...
        
        
        
        
        
        
