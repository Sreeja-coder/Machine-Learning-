import csv

import pandas as pd
import importlib
import numpy as np
import math


sample = pd.read_csv('misc_train.csv')
# sample = pd.read_csv("a1a_train.csv",names = [i for i in range(0,124)])
# print(len(sample))
featureset  = [i for i in range(1,6)]

sample_nuumpy = sample.to_numpy();

# print(sample_nuumpy.shape)
# print(list(sample_nuumpy[:,0]).count(1))
#
class treenode:
    def __init__(self, featureName):
        #print("featurename",featureName)
        self.featureName = featureName
        self.children = []

def findmajoritylabel(dataset):
    n = list(dataset[:,0]).count(-1)
    p = list(dataset[:,0]).count(1)
    if n > p:
        return -1
    else:
        return 1

def check_all_label_same(dataset):
    if len(dataset) == list(dataset[:,0]).count(1):
        return 1
    if len(dataset) == list(dataset[:,0]).count(-1):
        return -1

def cal_entropy(dataset,attributeval=None, feature=None):
    nrows = len(dataset)
    if feature == None:
        p = list(dataset[:,0]).count(1)
        n = list(dataset[:,0]).count(-1)
        if p / nrows == 0 or n / nrows == 0:
            # Entropy_S = 0
            return 0
        entropy = - ((p / nrows) * math.log((p / nrows), 2)) - ((n / nrows) * math.log((n / nrows), 2))
        return  entropy
    else:
        len_dataset_feature = dataset[dataset[:,feature] == attributeval]
        if len(len_dataset_feature) == 0:
            return  0
        else:
            p = list(len_dataset_feature[:,0]).count(1)/len(len_dataset_feature)
            n = list(len_dataset_feature[:,0]).count(-1)/len(len_dataset_feature)
            if p == 0 or n == 0:
                return 0
            entropy = -  p* math.log(p,2)  - n*math.log(n,2)
            # print(feature,attributeval,(len(len_dataset_feature)/nrows)*entropy)
            return (len(len_dataset_feature)/nrows)*entropy

def cal_max_gain(dataset, feature_set):
    dataset_entropy =  cal_entropy(dataset)
    # print("entropy",dataset_entropy)
    max_gain = -1
    selected_feature = ''
    for f in feature_set:
        entropy_feature = 0
        for all_vals in set(dataset[:,f]):
            entropy_feature = entropy_feature + cal_entropy(dataset,all_vals,f)
        gain = dataset_entropy - entropy_feature
        # print(f,gain)
        if max_gain < gain:
            max_gain = gain
            selected_feature = f
    # print(selected_feature)
    return selected_feature


def ID3(dataset,featureset):
    # print("length",len(dataset))
    # if current_depth + 1 > depth:
    #     return treenode(findmajoritylabel(dataset))
    #check if all labels are the same, if yes return a node with the label
    if check_all_label_same(dataset) == -1:  #guilty
        #print("label 0" )
        return treenode(-1)
    if check_all_label_same(dataset) == 1:  #not guilty
        #print("label 1")
        return treenode(1)
    if len(featureset) == 0:
        # print("empty")
        return treenode(findmajoritylabel(dataset))
    #find feature with max info gain
    new_selected_feature = cal_max_gain(dataset, featureset)
    # print(new_selected_feature)
    node = treenode(new_selected_feature)
    children = []
    set_of_possible_values = set(dataset[:,new_selected_feature])
    for v in set_of_possible_values:
        sv =  dataset[dataset[:,new_selected_feature] == v]
        if len(sv) == 0:
            # print("entered empty",new_selected_feature,v)
            return treenode(findmajoritylabel(dataset))
        else:
            new_feature_set = featureset.copy()
            new_feature_set.remove(new_selected_feature)
            childnode = ID3(sv, new_feature_set)
            child_dict = {"value": v, "child": childnode}
            children.append(child_dict)
    node.children = children
    return node

def cal_tree_depth(node,depth):
    if len(node.children) == 0:
        return depth
    max_depth = 0
    for child_node in node.children:
        child_node_depth = cal_tree_depth(child_node['child'], depth + 1)
        if max_depth < child_node_depth:
            max_depth = child_node_depth
    return max_depth



def prediction(row, root):
    if not root.children:
        # print(root.featureName)
        prediction_rows.append(root.featureName)
        return root.featureName

    decision_to_take = row[root.featureName]
    # print(root.featureName)
    # print(decision_to_take)
    index = 0
    for i in range(len(root.children)):
        if root.children[i]['value'] == decision_to_take:
            index = i
    prediction(row, root.children[index]['child'])
    # if decision_to_take == 0:
    #     prediction(row, root.children[0]['child'])
    # if decision_to_take == 1:
    #     prediction(row, root.children[1]['child'])
    # if decision_to_take == 2:
    #     prediction(row, root.children[2]['child'])




def accuracyCal(data, root):
    list_p = []
    for i in range(0, len(data)):
        # print(i)
        v = prediction(list(data.iloc[i]), root)
        # print(v)
    correct_predictions = 0
    wrong_predictiosn = 0
    # print(len(prediction_rows))
    # print(len(data))
    for i in range(0, len(data)):
        if data.iloc[i][0] == prediction_rows[i]:
            correct_predictions += 1
    return ((correct_predictions) / len(data))
    # return prediction_rows
    # for i in range(0, len(data)):
    #     if data.iloc[i][0] == prediction_rows[i]:
    #         correct_predictions += 1
    # return (correct_predictions) / len(data)

def accuracyCal_eval(data, root):
    list_p = []
    for i in range(0, len(data)):
        # print(i)
        v = prediction(list(data.iloc[i]), root)
        # print(v)
    correct_predictions = 0
    wrong_predictiosn = 0
    # print(len(prediction_rows))
    # print(len(data))
    # for i in range(0, len(data)):
    #     if data.iloc[i][0] == prediction_rows[i]:
    #         correct_predictions += 1
    # return ((correct_predictions) / len(data))
    return prediction_rows
    # for i in range(0, len(data)):
    #     if data.iloc[i][0] == prediction_rows[i]:
    #         correct_predictions += 1
    # return (correct_predictions) / len(data)


# fold1 = sample.iloc[5000:7500,:]
root =  ID3(sample.to_numpy(),featureset)
prediction_rows = []
# print("depth",cal_tree_depth(root,0))
print("training accuracy",accuracyCal(sample,root))
prediction_rows= []
test = pd.read_csv('misc_test.csv')
t = accuracyCal(test,root)
print("testing accuracy",t)

### for eval ####
prediction_rows= []
eval = pd.read_csv('misc_eval.csv')
p = accuracyCal_eval(eval,root)
for i in range(len(p)):
    if p[i] == -1:
        p[i] = 0
with open('ID3_submission2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["example_id", "label"])
    for i in range(len(p)):
        writer.writerow([i, p[i]])

