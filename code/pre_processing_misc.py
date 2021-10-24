import pandas as pd
from word2number import w2n
from nltk.corpus import stopwords
from nltk.corpus import wordnet


# tup = [('rishav', 10), ('akash', 5), ('ram', 20), ('gaurav', 15)]
# print(sorted(tup, key = lambda x: x[1]))
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

df_glove_train = read_split_data('glove.train.libsvm')
df_glove_test = read_split_data('glove.test.libsvm')
#
file =  pd.read_csv('misc-attributes-train.csv')
# file =  pd.read_csv('misc-attributes-test.csv')
# print(file['defendant_age'].describe())
labels = list(df_glove_train.iloc[:,0])
defendants_gender = list(file['defendant_gender'])
num_victims = list(file['num_victims'])
offense_category = list(file['offence_category'])
offense_sub_category = list(file['offence_subcategory'])
# victim_gender = list(file['victim_genders'])

# vg = []
# for g in victim_gender:
#     val = g.split(";")
#     if val.contains("male") and len(val) == 1:
#         vg.append("m")
#     if val.contains("female") and len(val) == 1:
#         vg.append("f")
#     if "male" in val and "female" in val and "female" in val and len(val) == 3:
#         vg.append("mff")
#     if "male" in val and "male" in val and "male" in val and len(val) == 3:
#         vg.append("mmm")
#     if "male" in val and "male" in val and "female" in val and len(val) == 3:
#         vg.append("mmf")
#     if "male" in val and "female" in val and "female" in val and len(val) == 3:
#         vg.append("mff")
#     if "female" in val and "female" in val and "female" in val and len(val) == 3:
#         vg.append("fff")








##### binning for defendants_age ##########
defendants_age = []
for i,age in enumerate(file['defendant_age']):
    # print("age",age)
    # print(i)
    age = age.strip("(  ) -")
    if age != "not known":
        age = age.replace("years","")
        age = age.replace("about", "")
        age = age.replace("age","")
        age = age.replace("of", "" )
        age = age.replace("old", "")
        age = age.strip()
        if age.find(" ") >= 0:
            temp = age.split(" ")
            # print(temp)
            age = '-'.join(temp)
        syns = wordnet.synsets(age.strip())
        # print("A",age)
        # print("S",syns[0].lemmas()[0].name())
        age = syns[0].lemmas()[0].name()
        if age.find("-") >= 0:
            temp = age.split("-")
            # print(temp)
            age = ' '.join(temp)
        # print(age.strip())
        defendants_age.append(w2n.word_to_num(age.strip()))
    else:
        defendants_age.append(int(0))
# print("**")
# mean =  int(sum(defendants_age)/sum(1 for x in defendants_age if x > 0))
# print("mean",mean)
# print(defendants_age)
# defendants_age = [mean for age in defendants_age if age == 0]
#subsitute not known with mean value
# for i in range(len(defendants_age)):
#     if defendants_age[i] == 0:
#         defendants_age[i] = mean

# print("******")
# print(len(defendants_age))
# # print(len(labels))
# # print("********")
# age_label = list(tuple(zip(defendants_age, labels)))
# # print(age_label)
# o = (sorted(age_label))
# # print(o)
# dict_o = {}
# for key,value in o:
#     dict_o.setdefault(key, []).append(value)
#
# # print(dict_o)
# defendants_age = []
# for k,v in dict_o.items():
#     print(k)
#     print("guilty",list(v).count(-1))
#     print("not guilty", list(v).count(1))

# for age in age_label:
#     defendants_age.append(age[0])
#     # if 8 <=age[0] < 22:
#     #     defendants_age.append("A")
#     # elif 22 <= age[0] < 38:
#     #     defendants_age.append("B")
#     # elif 38 <= age[0] < 54:
#     #     defendants_age.append("C")
#     # elif 54 <= age[0] < 70:
#     #     defendants_age.append("D")
#     # elif age[0] >= 70:
# #     #     defendants_age.append("E")
# for age in age_label:
#     # defendants_age.append(age[0])
#     if age[0] < 29:
#         defendants_age.append("A")
#     if  age[0] >= 29:
#         defendants_age.append("B")

    # if 10 <=age[0] < 27:
    #     defendants_age.append("A")
    # elif 27 <= age[0] < 44:
    #     defendants_age.append("B")
    # elif 44 <= age[0] < 61:
    #     defendants_age.append("C")
    # elif 61 <= age[0] < 78:
    #     defendants_age.append("D")
    # elif age[0] >= 78:
    #     defendants_age.append("E")



###### num_victims ########
# num_vic = list(tuple(zip(num_victims, labels)))
# num_vic = (sorted(num_vic))
#
# dict_num_victims =  {}
# for key,value in num_vic:
#     dict_num_victims .setdefault(key, []).append(value)
#
# print("num victims dict")
# print(dict_num_victims)
#
# for k,v in dict_num_victims.items():
#     print(k)
#     print("guilty",list(v).count(-1))
#     print("not guilty", list(v).count(1))
# print(len(labels))
# print(len(defendants_gender))
# print(len(defendants_age))
# print(len(offense_category))
# print(len(num_victims))
# df = pd.DataFrame(list(zip(labels,defendants_age, defendants_gender,num_victims,offense_category)),  columns =["labels","defendants_age", "defendants_gender","num_victims","offense_category"])
df = pd.DataFrame(list(zip(labels, defendants_age,defendants_gender,num_victims,offense_category,offense_sub_category)),  columns =["labels","defendants_age" ,"defendants_gender","num_victims","offense_category","offense_sub_category"])

df.to_csv('misc_train.csv' ,index = False)

#
# print(df['defendants_age'].describe())




#######  for testing ############

file =  pd.read_csv('misc-attributes-test.csv')
# print(file['defendant_age'].describe())
labels = label= list(df_glove_test.iloc[:,0])
defendants_gender = list(file['defendant_gender'])
num_victims = list(file['num_victims'])
offense_category = list(file['offence_category'])
offense_sub_category = list(file['offence_subcategory'])

##### binning for defendants_age ##########
defendants_age = []
for i,age in enumerate(file['defendant_age']):
    # print("age",age)
    # print(i)
    age = age.strip("(  ) -")
    if age != "not known":
        age = age.replace("years","")
        age = age.replace("about", "")
        age = age.replace("age","")
        age = age.replace("of", "" )
        age = age.replace("old", "")
        age = age.strip()
        if age.find(" ") >= 0:
            temp = age.split(" ")
            # print(temp)
            age = '-'.join(temp)
        syns = wordnet.synsets(age.strip())
        # print("A",age)
        # print("S",syns[0].lemmas()[0].name())
        age = syns[0].lemmas()[0].name()
        if age.find("-") >= 0:
            temp = age.split("-")
            # print(temp)
            age = ' '.join(temp)
        # print(age.strip())
        defendants_age.append(w2n.word_to_num(age.strip()))
    else:
        defendants_age.append(int(0))
# print("**")
# mean =  int(sum(defendants_age)/sum(1 for x in defendants_age if x > 0))
# print("mean",mean)
# print(defendants_age)
# defendants_age = [mean for age in defendants_age if age == 0]
#subsitute not known with mean value
# for i in range(len(defendants_age)):
#     if defendants_age[i] == 0:
#         defendants_age[i] = mean


df = pd.DataFrame(list(zip(labels, defendants_age,defendants_gender,num_victims,offense_category,offense_sub_category)),  columns =["labels","defendants_age" ,"defendants_gender","num_victims","offense_category","offense_sub_category"])

df.to_csv('misc_test.csv' ,index = False)


###########  for eval ##########

file =  pd.read_csv('misc-attributes-eval.csv')
# print(file['defendant_age'].describe())
labels = [0 for i in range(len(file))]
defendants_gender = list(file['defendant_gender'])
num_victims = list(file['num_victims'])
offense_category = list(file['offence_category'])
offense_sub_category = list(file['offence_subcategory'])

defendants_age = []
for i, age in enumerate(file['defendant_age']):
    #         print("age",age)
    # print(i)
    age = age.strip("(  ) -")
    if age != "not known":
        age = age.replace("years", "")
        age = age.replace("about", "")
        age = age.replace("age", "")
        age = age.replace("of", "")
        age = age.replace("old", "")
        age = age.replace("Year", "")
        age = age.replace("his", "")
        age = age.replace("Age", "")
        age = age.strip()
        age = age.strip("d")

        if age.find(" ") >= 0:
            temp = age.split(" ")
            if "and" in temp or "or" in temp:
                age = temp[0]
            #                     temp.remove("and")
            #                 if "months" in temp or "month" in temp:
            #                     temp.remove("months")
            # #                     temp.remove("month")
            # print(temp)
            else:
                age = '-'.join(temp)
        #             print("**",age)
        syns = wordnet.synsets(age.strip())
        #             print("syns",syns)
        # print("A",age)
        # print("S",syns[0].lemmas()[0].name())
        age = syns[0].lemmas()[0].name()
        if age.find("-") >= 0:
            temp = age.split("-")
            # print(temp)
            age = ' '.join(temp)
        #             print(age)
        #             defendants_age.append(w2n.word_to_num(age.strip()))
        try:
            defendants_age.append(w2n.word_to_num(age.strip()))
        except:
            defendants_age.append(int(0))
    else:
        defendants_age.append(int(0))

df = pd.DataFrame(list(zip(labels, defendants_age,defendants_gender,num_victims,offense_category,offense_sub_category)),  columns =["labels","defendants_age" ,"defendants_gender","num_victims","offense_category","offense_sub_category"])

df.to_csv('misc_eval.csv' ,index = False)
