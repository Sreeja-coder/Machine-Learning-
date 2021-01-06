#### important #####
please run the commands in requirements.txt before you run any other commands
my code uses 
 nltk  - there in cade
 word2number  - not there in cade
 wordnet  - there in cade 
 stop-words  - there in cade
 sklearn libraries  - there in cade

##### all my cross validation are commented out and I have used the hyper-parameters that i get from cross validation directly because cade kept killing my process #####


#### Submission 1 Average perceptron ####

command to run : python3 avg_perceptron_tfidf_submission1.py
this uses tfidf dataset please keep data and .py file in same folder



#### Submission 2 Margin perceptron ####

command to run : python3 margin_perceptron_bow.py
this uses bow dataset please keep data and .py file in same folder


#### Submission 2 ID3 ####
please run the commands in requirements.txt before you run any other commands
commands to run : python3 pre_processing_misc.py    ## pre-processes the files and creates two new .csv files misc-train.csv and misc-test.csv
		  python3 ID3_misc.py   
this uses glove to get labels and misc to train and evalaute the model


#### Submission 3 SVM ####

command to run : python3 implementing_svm
this uses bow dataset please keep data and .py files in same folder


#### Submission 3 Random Forest using python sklearn package ####
please run the commands in requirements.txt before you run any other commands
command to run: Random_forest_misc_final.ipynb
this uses glove to get labels and misc to train and evalaute the model,please keep data and .py files in same folder 


#### Submission 3 Ensemble SVM-Perceptron ####

command to run : perceptron_svm_ensemble_tfidf.ipynb
this uses tfidf dataset, please keep data and .py files in same folder 

command to run : perceptron_svm_ensmble.ipynb
this uses glove to get labels and misc to train and evalaute the model,please keep data and .py files in same folder


#### Submission 3 Ensemble Perceptron-ID3 ####

please run the commands in requirements.txt before you run any other commands
command to run : perceptron_tree_ensemble_working.ipynb
this uses glove to get labels and misc to train and evalaute the model,please keep data and .py files in same folder