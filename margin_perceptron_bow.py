import importlib
import pandas as pd
import numpy as np
import math
import csv



def margin_perceptron(data, hp,m, ep):
    # initialize weights and bias term randomly between -0.01 and 0.01
    # initialize weights and bias term randomly between -0.01 and 0.01
    np.random.seed(7)
    updates = 0
    w = np.random.uniform(-0.01, 0.01, size=data.shape[1] - 1)
    dict_epoch_acc = {}
    learning_rate = hp
    t = 0
    b = np.random.uniform(-0.01, 0.01)
    # epoch is 20 as per question
    for i in range(ep):
        #np.random.seed(7)
        #learning_rate = learning_rate/(i+1)
        accuracy = 0
        np.random.shuffle(data)
        for r in range(len(data)):
            ground_truth = data[r, 0]
            sample = data[r, 1:]

            if ground_truth*(np.dot(np.transpose(w), sample) + b) <= m:
                updates +=1
                w = w + learning_rate * ground_truth * sample
                b = b + learning_rate * ground_truth
            else:
                accuracy += 1



        learning_rate = learning_rate/(i+1)
        # print("epoch",i)
        # print("accuracy",accuracy/len(data))
        dict_epoch_acc[i] = w, b, (accuracy / len(data))
    return dict_epoch_acc,updates

def cal_max(d):
    max_w = []
    max_accuracy = 0
    bias = 0
    for key, value in d.items():
        if value[2] > max_accuracy:
            max_accuracy = value[2]
            max_w = value[0]
            bias = value[1]
    return max_w,bias,max_accuracy

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
        prediction = -1 if np.dot(np.transpose(we), sample) + bi <= 1 else 1
        if prediction == -1:
            predictions[r] = 0
        else:
            predictions[r] = 1
        # if prediction == ground_truth:
        #     accuracy = accuracy + 1
    return predictions


def cross_validation(f1,f2,f3,f4,f5):
    margin = [1, 0.1, 0.01]
    learning_rate = [1, 0.1, 0.01]
    DATA_DIR = "data/"
    file_names = ["fold1.csv", "fold2.csv", "fold3.csv", "fold4.csv", "fold5.csv"]
    dict_margin_lr = {}
    for m in margin:
        for lr in learning_rate:
            acc = 0
            # run for f1 as test:
            frames = [f2, f3, f4, f5]
            train = pd.concat(frames)
            d, u = margin_perceptron(train.to_numpy(), lr,m, 10)
            w1, b1, a1 = cal_max(d)
            acc = acc + evaluate(w1, b1, f1)
            # run for f2 as test
            frames = [f1, f3, f4, f5]
            train = pd.concat(frames)
            d, u = margin_perceptron(train.to_numpy(), lr,m, 10)
            w1, b1, a1 = cal_max(d)
            acc = acc + evaluate(w1, b1, f2)
            # run for f3 as test
            frames = [f2, f1, f4, f5]
            train = pd.concat(frames)
            d, u = margin_perceptron(train.to_numpy(), lr,m, 10)
            w1, b1, a1 = cal_max(d)
            acc = acc + evaluate(w1, b1, f3)
            # run for f4 as test
            frames = [f2, f1, f3, f5]
            train = pd.concat(frames)
            d, u = margin_perceptron(train.to_numpy(), lr,m, 10)
            w1, b1, a1 = cal_max(d)
            acc = acc + evaluate(w1, b1, f4)
            # run for f5 as test
            frames = [f2, f1, f4, f3]
            train = pd.concat(frames)
            d, u = margin_perceptron(train.to_numpy(), lr,m, 10)
            w1, b1, a1 = cal_max(d)
            acc = acc + evaluate(w1, b1, f5)
            dict_margin_lr[(m, lr)] = acc/5
    #print(dict_margin_lr.keys())
    return dict_margin_lr





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





print("Starting reading files for margin using bow")

df_train = read_split_data('bow.train.libsvm')

df_test = read_split_data('bow.test.libsvm')

df_eval = read_split_data('bow.eval.anon.libsvm')


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

##### cross validation ######
fold1 = df_train.iloc[5000:7500,:]
fold2 = df_train.iloc[7500:10000,:]
fold3 = df_train.iloc[10000:12500,:]
fold4 = df_train.iloc[12500:15000,:]
fold5 = df_train.iloc[15000:17500,:]
#### kindly un comment it to run the cross validation part #####
# hyper_p_dict= cross_validation(fold1,fold2,fold3,fold4,fold5)
# best_hyper_p = 0
# best_acc_hp = 0
# for key, value in hyper_p_dict.items():
#     #print("key",key)
#     #print("value",value)
#     if best_acc_hp < value:
#         best_acc_hp = value
#         best_hyper_p = key
# print("----Margin perceptron----- ")
# best_margin = best_hyper_p[0]
# best_lr = best_hyper_p[1]
# print("cross validation Best hyper paramter value (margin,leraning rate) ", best_hyper_p)




df_numpy = df_train.to_numpy()

safe_max = np.abs(df_numpy).max(axis=0)
safe_max[safe_max==0] = 1

df_numpy_normed =  df_numpy / safe_max
dict, u = margin_perceptron(df_numpy_normed, 0.01, 1, 10)  # i get this after running cross validation
# dict, u = margin_perceptron(df_numpy_normed, best_lr, best_margin, 20)

acc = 0
we_training = []
x_axis = []
y_axis = []
bias_training = 0
for key, value in dict.items():
    x_axis.append(key)
    y_axis.append(value[2] * 100)
    if acc < value[2]:
        acc = value[2]
        we_training = value[0]
        bias_training = value[1]

# print("accuracy", acc * 100)
# print("shape",df_train.shape)
# print("shape",df_test.shape)
print("training accuracy",evaluate(we_training, bias_training, df_train))



print("testing accuracy",evaluate(we_training, bias_training, df_test))
print("calculating accuracy on eval")
p = cal_acc_eval(we_training, bias_training, df_eval)
with open('margin_perceptron_bow.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["example_id", "label"])
    for i in range(len(p)):
        writer.writerow([i, p[i]])






