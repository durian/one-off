
# python3 dnn_model_loop.py --hidden_units=200x200 --chosen_label=ENGINE_TYPE

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
parser.add_argument('--chosen_label', default="T_CHASSIS", type=str, help='the label to train and evaluate')
parser.add_argument('--hidden_units', default="200x200", type=str, help='Number of hidden units')
parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate')
parser.add_argument('--optimiser', default="DFLT", type=str, help='Optimiser, Adam or GDO')
parser.add_argument('--suffix', default="", type=str, help='Model dir suffix')
parser.add_argument("--test", action='store_true', default=False, help='Exit before training' )
parser.add_argument("--dropout", default=None, type=float, help='Hidden layer dropout (0-1)' )
parser.add_argument('--iterations', default=30, type=int, help='Number of interations of steps')
parser.add_argument('--steps', default=10000, type=int, help='Number steps per iteration')

"""
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')
parser.add_argument('--hidden_units', default=[10,10], type=string, help='layout for hidden layers')
parser.add_argument('--nr_epochs', default=None, type=int, help='number of epochs')
parser.add_argument('--chosen_label', default=T_CHASSIS, type=string, help='the label to train and evaluate')
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
        train_steps = args.steps #10000 # 1000 #LOOPED LATER ON
        nr_epochs = None
        hu = [int(x) for x in args.hidden_units.split("x")]
        hidden_units = hu #[200, 200] # [10, 10] [400, 400] [400, 400, 400, 400]
        chosen_label = args.chosen_label #'ENGINE_TYPE' # 'T_CHASSIS' 'COUNTRY' 'ENGINE_TYPE' 'BRAND_TYPE'
        max_nr_nan = 0
        fixed_selection = True

        my_id="l"+chosen_label+"_s"+str(train_steps)+"_h"+args.hidden_units

        label_path = 'Labels/'
        data_path = 'Data_original/' # 'Data_original/' 'Testdata/'
        structured_data_path = 'Compressed/' # 'Compressed_valid_chassis' Compressed/Compressed_single/
        
        # Label_mapping holds key value pairs where key is the label and value its integer representation
        label_mapping = dataloader.get_valid_labels(label_path, chosen_label) # Labels from labels file only

        #Get three structured separate dataframes from data sources
        #trainframe, testframe, validationframe = dataloader.loadData(data_path, False, label_mapping, max_nr_nan, fixed_selection)
        trainframe, testframe, validationframe = dataloader.loadData(structured_data_path, True, label_mapping, max_nr_nan, fixed_selection)
        
        # Train model data
        trainset, labels_training, label_mapping, int_labels_train = \
                dataloader.get_model_data(trainframe, label_mapping, chosen_label)
        
        # Test model data
        testset, labels_test, label_mapping, int_labels_test = \
                dataloader.get_model_data(testframe, label_mapping, chosen_label)
        
        # Validate model data
        validationset, labels_validate, label_mapping, int_labels_validate = \
                dataloader.get_model_data(validationframe, label_mapping, chosen_label)

        if args.test:
                sys.exit(2)
                
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
        if args.optimiser == "Adam":
                opt = tensorflow.train.AdamOptimizer(learning_rate=args.learning_rate)
                my_id=my_id+"_o"+args.optimiser
                my_id=my_id+"_lr"+str(args.learning_rate)
        elif args.optimiser == "GDO":
                opt = tensorflow.train.GradientDescentOptimizer(learning_rate=learning_rate)
                my_id=my_id+"_o"+args.optimiser
                my_id=my_id+"_lr"+str(args.learning_rate)
        else:
                opt = None #adagrad, whoch doesn not have a learning rate
                my_id=my_id+"_o"+args.optimiser

        if args.dropout:
                my_id = my_id + "_do" + str(args.dropout)
                
        if args.suffix != "":
                my_id = my_id + "_" + args.suffix

        # Save data files
        #         validationset, labels_validate, label_mapping, int_labels_validate = \
        trainset.to_csv( "train_"+my_id+".csv", index=False )
        labels_training.to_csv( "labels_train_"+my_id+".csv", index=False )
        testset.to_csv( "test_"+my_id+".csv", index=False )
        labels_test.to_csv( "labels_test_"+my_id+".csv", index=False )
        validationset.to_csv( "val_"+my_id+".csv", index=False )
        labels_validate.to_csv( "labels_val_"+my_id+".csv", index=False )

        resultfile = open("Results/"+my_id+"_model_results.txt", "w")
        
        resultfile.write('\nModel training: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n\n')
        resultfile.write('Layer setting: ' + str(hidden_units) + '\n')
        resultfile.write('Train steps: ' + str(train_steps) + '\n')
        resultfile.write('Number epochs: ' + str(nr_epochs) + '\n')
        resultfile.write('Batchsize: ' + str(batch_size) + '\n')
        resultfile.write('Chosen label: ' + chosen_label + '\n')
        resultfile.write('Max_nr_nan: ' + str(max_nr_nan) + '\n')
        resultfile.write('Fixed_selection: ' + str(fixed_selection) + '\n')
        resultfile.flush()

        if opt:
                classifier = tensorflow.estimator.DNNClassifier(
                        feature_columns=my_feature_columns,
                        hidden_units=hidden_units,
                        n_classes=len(label_mapping),
                        model_dir="./models/"+my_id,
                        config=my_checkpointing_config,
                        dropout=args.dropout,
                        optimizer=opt)
        else:
                classifier = tensorflow.estimator.DNNClassifier(
                        feature_columns=my_feature_columns,
                        hidden_units=hidden_units,
                        n_classes=len(label_mapping),
                        model_dir="./models/"+my_id,
                        dropout=args.dropout,
                        config=my_checkpointing_config)
                
        
        ### Train the Model.
        # PJB added loop
        for i in range(0,args.iterations):
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
                #
                expected = list(int_labels_validate)  #list(label_mapping.keys())
                predictions = classifier.predict(input_fn=lambda:dataloader.eval_input_fn(validationset,
                                                                                          labels=expected,
                                                                                          batch_size=batch_size))
                template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
                predictfile = open("Results/"+my_id+"_predictions.txt", "a")

                corr = 0 # count correct predictions
                for pred_dict, expec in zip(predictions, expected):
                        template = ('Prediction is "{}" ({:5.1f}%), expected "{}"')
                        class_id = pred_dict["class_ids"][0]
                        probability = pred_dict['probabilities'][class_id]
                        print( template.format(class_id, 100 * probability, expec) )
                        resultfile.write(template.format(class_id, 100 * probability, expec))
                        # indices should be reverse mapped to correct label
                        predictfile.write("Loop {}: Prediction is {} ({:.1f}%), expected {}\n".format(i, class_id, 100 * probability, expec) )
                        if class_id == expec:
                                corr += 1
                print( "CORRECT", corr, "/", len(expected) )
                resultfile.write( "Correct: "+str(corr)+"\n" )
                predictfile.write( "Loop {}: Correct {}/{} ({:.2f}%)\n".format(i, corr, len(expected), (corr*100)/len(expected)) )
                predictfile.close()
                resultfile.write('\n******************************\n')
        resultfile.close()

        
if __name__ == '__main__':
    tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
    tensorflow.app.run(main) # So far only a dummy arguments...
        
        
        
        
        
        
