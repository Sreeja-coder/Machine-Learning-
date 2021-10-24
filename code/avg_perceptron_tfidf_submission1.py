import pandas as pd
import importlib
import numpy as np
import math
import csv

def batch_perceptron(data,hp,ep):
    # initialize weights and bias term randomly between -0.01 and 0.01
    np.random.seed(7)
    update = 0
    av_w = np.zeros(shape=data.shape[1]-1)
    av_b = 0
    w = np.random.uniform(-0.01, 0.01, size=data.shape[1] - 1)
    b = np.random.uniform(-0.01, 0.01)
    dict_epoch_acc = {}
    learning_rate = hp
    for i in range(ep):
        accuracy = 0
        np.random.shuffle(data)
        for r in range(len(data)):
            ground_truth = data[r, 0]
            sample = data[r, 1:]
            if np.dot(np.transpose(w), sample) + b <= 0:
                prediction = -1
            else:
                prediction = 1
            if int(ground_truth) != int(prediction):
                update += 1
                w = w + learning_rate * ground_truth * sample
                b = b + learning_rate * ground_truth
            else:
                accuracy += 1
            av_w = av_w + w
            av_b = av_b + b
        dict_epoch_acc[i] = av_w, av_b, (accuracy / len(data))
    return  dict_epoch_acc,update


def evaluate(we, bi, test_data):
    accuracy = 0
    for r in range(len(test_data)):
        ground_truth = test_data.iloc[r][0]
        sample = list(test_data.iloc[r])
        sample.pop(0)
        prediction = -1 if np.dot(np.transpose(we), sample) + bi <= 0 else 1
        if prediction == ground_truth:
            accuracy = accuracy + 1
    return (accuracy / len(test_data)) * 100

def cal_acc_eval(we, bi, test_data):
    accuracy = 0
    predictions = {}
    for r in range(len(test_data)):
        ground_truth = test_data.iloc[r][0]
        sample = list(test_data.iloc[r])
        sample.pop(0)
        prediction = -1 if np.dot(np.transpose(we), sample) + bi <= 0 else 1
        if prediction == -1:
            predictions[r] = 0
        else:
            predictions[r] = 1
        # if prediction == ground_truth:
        #     accuracy = accuracy + 1
    return predictions


def cal_max(dict):
    acc = 0
    we_training = []
    bias_training = 0
    for key, value in dict.items():
        if acc < value[2]:
            acc = value[2]
            we_training = value[0]
            bias_training = value[1]
    return we_training,bias_training,acc



def read_split_data(filename):
    file = open(filename, "r").read()
    list_rows = []
    file = file.split("\n")
    for row in range(0, len(file) - 1):
        dict_rows = {}
        splits = file[row].split(" ")
        for s in range(len(splits)):
            if s == 0:
                if int(splits[s]) == 1:
                    dict_rows[s] =  int(splits[0])
                else:
                    dict_rows[s] = -1
            else:
                index,val = [float(e) for e in splits[s].split(':')]
                dict_rows[index] = val
        list_rows.append(dict_rows)
    df =  pd.DataFrame.from_dict(list_rows)
    df = df.fillna(0)
    return df

def crossvalidation(f1,f2,f3,f4,f5):

    best_h = 0
    max_acc = 0
    hyper_paramter = [0.1,1,0.01]
    for h in hyper_paramter:
        acc = 0
        #run for f1 as test:
        frames = [f2, f3, f4, f5]
        train =pd.concat(frames)
        d, u = batch_perceptron(train.to_numpy(), h, 10)
        w1,b1,a1 = cal_max(d)
        acc = acc + evaluate(w1,b1,f1)
        #run for f2 as test
        frames = [f1, f3, f4, f5]
        train = pd.concat(frames)
        d, u = batch_perceptron(train.to_numpy(), h, 10)
        w1, b1, a1 = cal_max(d)
        acc = acc + evaluate(w1, b1, f2)
        #run for f3 as test
        frames = [f2, f1, f4, f5]
        train = pd.concat(frames)
        d, u = batch_perceptron(train.to_numpy(), h, 10)
        w1, b1, a1 = cal_max(d)
        acc = acc + evaluate(w1, b1, f3)
        #run for f4 as test
        frames = [f2, f1, f3, f5]
        train = pd.concat(frames)
        d, u = batch_perceptron(train.to_numpy(), h, 10)
        w1, b1, a1 = cal_max(d)
        acc = acc + evaluate(w1, b1, f4)
        #run for f5 as test
        frames = [f2, f1, f4, f3]
        train = pd.concat(frames)
        d, u = batch_perceptron(train.to_numpy(), h, 10)
        w1, b1, a1 = cal_max(d)
        acc = acc + evaluate(w1, b1, f5)
        if max_acc < acc/5:
            max_acc = acc/5
            best_h = h
    return best_h



df_train = read_split_data('tfidf.train.libsvm')
df_test = read_split_data('tfidf.test.libsvm')
df_eval = read_split_data('tfidf.eval.anon.libsvm')

print("running average perceptron on tfidf")
cols_test = list(df_test.columns)
for i in range(0,10001):
    if i not in cols_test:
        df_test[i] = [0.0 for i in range(len(df_test))]


cols_eval = list(df_eval.columns)
for i in range(0,10001):
    if i not in cols_eval:
        df_eval[i] = [0.0 for i in range(len(df_eval))]



df_test = df_test.reindex(sorted(df_test.columns), axis = 1)
df_train = df_train.reindex(sorted(df_train.columns), axis = 1)
df_eval =  df_eval.reindex(sorted(df_eval.columns),axis = 1)

# please uncomment cross validation if we want to , it takes long time hence commented it out
# fold1 = df_train.iloc[5000:7500,:]
# fold2 = df_train.iloc[7500:10000,:]
# fold3 = df_train.iloc[10000:12500,:]
# fold4 = df_train.iloc[12500:15000,:]
# fold5 = df_train.iloc[15000:17500,:]
# print("calling corss validation to find best [0.1,1,0.01]")
# best_lr = crossvalidation(fold1,fold2,fold3,fold4,fold5)
# #
# training_model = df_train.iloc[:5000,:]
# print("best lr",best_lr)
# dict_training_per,u = batch_perceptron(df_train.to_numpy(), best_lr, 10)
dict_training_per,u = batch_perceptron(df_train.to_numpy(), 0.01, 10)
w,b,a = cal_max(dict_training_per)
# print("shape",df_train.shape)
print("accuracy on training",a)

print("accuracy on testing ",evaluate(w, b, df_test))
#running evaluation
print("calculating accuracy on eval")
p = cal_acc_eval(w,b,df_eval)
with open('avg_perceptron_tfidf.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["example_id", "label"])
    for i in range(len(p)):
        writer.writerow([i, p[i]])
