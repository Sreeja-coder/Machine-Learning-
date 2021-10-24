import pandas as pd
import importlib
import numpy as np
import math
import csv
from random import seed


global threshold
def svm(data,lr,C,ep):
    #initialize weights and bias term

    lr_0 = lr
    loss = 0.0
    dict_epoch_accuracy = {}
    dict_epoch_loss = {}
    prev_loss =  float("inf")
    seed(30)
    # print(data.shape[1])
    w = np.random.uniform(-0.01,0.01,size=data.shape[1])
    # for i in range(len(data.columns)):
    #     w.append(uniform(-0.01,0.01))
    # print(len(w))
    #for epoch as per threshold
    for i in range(ep):
        # print("for epoch",i)

        lr = lr_0/(1+i)
        np.random.shuffle(data)
        # data =  data.sample(frac=1,random_state=1)
        for row in range(len(data)):
            y = data[row,0]
            x = data[row,1:]
            # x.pop(0)
            x = np.append(x,1)

            # print("len of x",len(x))
            if np.dot((np.transpose(w)),x)*y < 1:
                w = ((1-lr)*w) + ((lr*C*y)*x)
            else:
                w = (1-lr)*w

        dict_epoch_accuracy[i] = i,evaluate(w,data),w
        # print("accuracy",accuracy)
            ## calculate the loss/value of the objective function
        loss = (1/2)*np.dot(np.transpose(w),w)
        l = 0
        # loss = (1/2)*np.dot(np.transpose(w),w)
        for row in range(len(data)):
            y1 = data[row,0]
            x1 = data[row,1:]
            x1 = np.append(x1, 1)
            # loss = round(loss + max(0,(1- (y1 * np.dot(np.transpose(w),x1)))),4)
            l =  l + (max(0, (1 - (y1 * np.dot(np.transpose(w), x1)))))
        loss = loss + C*l
        # loss =  (loss/len(data))* C +  (1/2)*np.dot(np.transpose(w),w)
        # loss = loss
        # print("loss",loss)
        # print("loss",loss)
        dict_epoch_loss[i] = loss
        # print(loss)
        if abs(prev_loss - loss)  < 1:
            # print("threshold",i)
            global threshold
            threshold = i
            # print("threshold epoch",i)
            break
        prev_loss = loss

    return dict_epoch_accuracy,dict_epoch_loss
#
# def cal_loss(dict_w):
#     for key,value in dic



def svm_cross_validation(data,lr,C,ep):
    #initialize weights and bias term
    w = []
    lr_0 = lr
    loss = 0.0
    dict_epoch_accuracy = {}
    dict_epoch_loss = {}
    prev_loss =  float("inf")
    # seed(7)
    # print(data.shape[1])
    w = np.random.uniform(-0.01,0.01,size=data.shape[1])
    # for i in range(len(data.columns)):
    #     w.append(uniform(-0.01,0.01))
    # print(len(w))
    #for epoch as per threshold
    for i in range(ep):
        # print("for epoch",i)
        accuracy = 0
        lr = lr_0/(1+i)
        np.random.shuffle(data)
        # data =  data.sample(frac=1,random_state=1)
        for row in range(len(data)):
            y = data[row,0]
            x = data[row,1:]
            # x.pop(0)
            x = np.append(x,1)

            # print("len of x",len(x))
            if np.dot((np.transpose(w)),x)*y <= 1:
                w = ((1-lr)*w) + ((lr*C*y)*x)
            else:
                w = (1-lr)*w
                accuracy = accuracy + 1
        dict_epoch_accuracy[i] = i,evaluate(w,data),w
    return dict_epoch_accuracy




def cal_max(d):
    max_w = []
    max_accuracy = -1
    epoch = 0
    for key, value in d.items():
        if value[1] > max_accuracy:
            max_accuracy = value[1]
            max_w = value[2]
            epoch = value[0]
        # if max_accuracy == 0:
    # print(len(max_w))
    return  epoch, max_accuracy,max_w


def evaluate(we, test_data):
    accuracy = 0
    for r in range(len(test_data)):
        y = test_data[r,0]
        x = test_data[r,1:]
        # sample.pop(0)
        x = np.append(x,1)
        # print(len(we))
        # print(len(x))
        prediction = -1 if np.dot(np.transpose(we), x) <= 0 else 1
        # if np.dot((np.transpose(we)),x)*y >= 1:
        if prediction == y:
            accuracy = accuracy + 1
    return (accuracy / len(test_data)) * 100

def evaluate_eval(we, test_data):
    accuracy = 0
    predictions = []
    for r in range(len(test_data)):
        y = test_data[r,0]
        x = test_data[r,1:]
        # sample.pop(0)
        x = np.append(x,1)
        # print(len(we))
        # print(len(x))
        prediction = 0 if np.dot(np.transpose(we), x) <= 0 else 1
        # if np.dot((np.transpose(we)),x)*y >= 1:
        predictions.append(prediction)
    return predictions


def crossvalidation(f1,f2,f3,f4,f5):
    lr = [10**0,10**-1,10**-2,10**-3,10**-4]
    C = [10**3,10**2,10**1,10**0,10**-1,10**-2]
    best_lr=0
    best_c = 0
    max_acc = 0
    for c in C:
        for l in lr:
            print("running for learning rate:",l,"C:",c)
            acc = 0
            # run for f1 as test:
            # print("f1 as test")
            frames = [f2, f3, f4, f5]
            # print("len", len(f2.columns))
            # print("len", len(f3.columns))
            # print("len", len(f4.columns))
            # print("len", len(f5.columns))
            train = pd.concat(frames)
            # print("len",len(train.columns))
            # dict_acc_w= svm_cross_validation(train.to_numpy(), l, c, 20)
            dict_acc_w, d = svm(train.to_numpy(), l, c, 20)
            e,a,w = cal_max(dict_acc_w)
            acc = acc + evaluate(w, f1.to_numpy())
            # run for f2 as test
            # print("f2 as test")
            frames = [f1, f3, f4, f5]
            train = pd.concat(frames)
            # dict_acc_w= svm_cross_validation(train.to_numpy(), l, c, 20)
            dict_acc_w,d= svm(train.to_numpy(), l, c, 20)
            e,a,w = cal_max(dict_acc_w)
            acc = acc + evaluate(w ,f2.to_numpy())
            # run for f3 as test
            # print("f3 as test")
            frames = [f2, f1, f4, f5]
            train = pd.concat(frames)
            # dict_acc_w= svm_cross_validation(train.to_numpy(), l, c, 20)
            dict_acc_w,d = svm(train.to_numpy(), l, c, 20)
            e,a,w = cal_max(dict_acc_w)
            acc = acc + evaluate(w, f3.to_numpy())
            # run for f4 as test
            # print("f4 as test")
            frames = [f2, f1, f3, f5]
            train = pd.concat(frames)
            # dict_acc_w= svm_cross_validation(train.to_numpy(), l, c, 20)
            dict_acc_w, d = svm(train.to_numpy(), l, c, 20)
            e,a,w = cal_max(dict_acc_w)
            acc = acc + evaluate(w, f4.to_numpy())
            # run for f5 as test
            # print("f5 as test")
            frames = [f2, f1, f4, f3]
            train = pd.concat(frames)
            # dict_acc_w= svm_cross_validation(train.to_numpy(), l, c, 20)
            dict_acc_w, d = svm(train.to_numpy(), l, c, 20)
            e,a,w = cal_max(dict_acc_w)
            acc = acc + evaluate(w, f5.to_numpy())
            acc = acc/5
            print("Accuracy:",acc)
            if max_acc<acc:
                max_acc = acc
                best_c =c
                best_lr = l
    print("best hyper-paramter after cross validation is ","accuracy :",max_acc,"C:",best_c,"learning rate",best_lr)
            # dict_margin_lr[(m, lr)] = acc / 5
            # print(dict_margin_lr.keys())
    return max_acc,best_c,best_lr



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


def cal_svm():
    df_train = read_split_data('tfidf.train.libsvm')
    df_test = read_split_data('tfidf.test.libsvm')
    df_eval = read_split_data('tfidf.eval.anon.libsvm')
    cols_test = list(df_test.columns)
    for i in range(0, 10001):
        if i not in cols_test:
            df_test[i] = [0.0 for i in range(len(df_test))]
    cols_eval = list(df_eval.columns)
    for i in range(0, 10001):
        if i not in cols_eval:
            df_eval[i] = [0.0 for i in range(len(df_eval))]
    df_test = df_test.reindex(sorted(df_test.columns), axis=1)
    df_train = df_train.reindex(sorted(df_train.columns), axis=1)
    df_eval = df_eval.reindex(sorted(df_eval.columns), axis=1)

    # print(df_train.to_numpy())

    ### performing cross validation  un comment it if you want to run it ###
    fold1 = df_train.iloc[5000:7500,:]
    fold2 = df_train.iloc[7500:10000,:]
    fold3 = df_train.iloc[10000:12500,:]
    fold4 = df_train.iloc[12500:15000,:]
    fold5 = df_train.iloc[15000:17500,:]

    # a,c,lr = crossvalidation(fold1,fold2,fold3,fold4,fold5)
    print("running svm")
    train_numpy = df_train.to_numpy()
    # safe_max = np.abs(train_numpy).max(axis =0)
    # safe_max[safe_max==0] = 1
    # train_numpy = train_numpy / safe_max
    # dict_a_w, dict_loss = svm(train_numpy, lr, c, 100)
    dict_a_w, dict_loss = svm(train_numpy, 0.001, 1000, 50)
    e, a, w = cal_max(dict_a_w)
    test_numpy = df_test.to_numpy()
    # safe_max = np.abs(test_numpy).max(axis =0)
    # safe_max[safe_max==0] = 1
    # test_numpy = test_numpy / safe_max
    print("train Accuracy :", evaluate(w, train_numpy))

    eval_numpy = df_eval.to_numpy()
    # safe_max = np.abs(eval_numpy).max(axis =0)
    # safe_max[safe_max==0] = 1
    # eval_numpy = eval_numpy / safe_max
    print("test Accuracy :", evaluate(w, test_numpy))
    pred = evaluate_eval(w,df_eval.to_numpy())
    with open('svm.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["example_id", "label"])
        for i in range(len(pred)):
            writer.writerow([i, pred[i]])

cal_svm()
