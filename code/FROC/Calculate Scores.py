import numpy as np
from sklearn import preprocessing, cross_validation, neighbors , svm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random

#Const
POSTIVE = 1
NEGATIVE = -1
TESTSIZE = 0.2
NUMBER_OF_RUNS = int(1/TESTSIZE)
random_state = np.random.RandomState(0)
NUMBER_OF_PATCHES_PER_IMAGE = 1078
NUMBER_OF_NORMAL_IMAGES = 1


#Read patches
df = pd.read_csv('family__bior3.9__level__1__image_type__ad.txt')
content = np.array(df.drop(['class'], 1))
classes = np.array(df['class'])

########################################
#Split patches to 5 equal lists of normal
#Split patches to 5 equal lists of noduled

content = content.tolist()
classes = classes.tolist()

for F in range(5):

    normal_patches = []
    normal_patches_class = []
    noduled_patches = []
    noduled_patches_class = []


    for i in range(len(content)):
        if(classes[i] == POSTIVE):
            noduled_patches.append(content[i])
            noduled_patches_class.append(classes[i])
        else:
            normal_patches.append(content[i])
            normal_patches_class.append(classes[i])

    random.shuffle(normal_patches)
    normal_patches = normal_patches[:NUMBER_OF_NORMAL_IMAGES * NUMBER_OF_PATCHES_PER_IMAGE]
    normal_patches_class = normal_patches_class[:NUMBER_OF_NORMAL_IMAGES * NUMBER_OF_PATCHES_PER_IMAGE]


    normal = []
    normalclass = []
    noduled = []
    noduledclass = []

    for i in range(NUMBER_OF_RUNS):
        if i == NUMBER_OF_RUNS-1:
            normal.append(normal_patches[int(i* TESTSIZE * len(normal_patches)):])
            normalclass.append(normal_patches_class[int(i* TESTSIZE * len(normal_patches_class)):])
        else:
            normal.append(normal_patches[int(i * TESTSIZE * len(normal_patches)):int((i + 1) * TESTSIZE * len(normal_patches))])
            normalclass.append(normal_patches_class[int(i * TESTSIZE * len(normal_patches_class)):int((i + 1) * TESTSIZE * len(normal_patches_class))])


    for i in range(NUMBER_OF_RUNS):
        if i == NUMBER_OF_RUNS-1:
            noduled.append(noduled_patches[int(i* TESTSIZE * len(noduled_patches)):])
            noduledclass.append(noduled_patches_class[int(i* TESTSIZE * len(noduled_patches_class)):])
        else:
            noduled.append(noduled_patches[int(i * TESTSIZE * len(noduled_patches)):int((i + 1) * TESTSIZE * len(noduled_patches))])
            noduledclass.append(noduled_patches_class[int(i * TESTSIZE * len(noduled_patches_class)):int((i + 1) * TESTSIZE * len(noduled_patches_class))])




    print("startloop")

    for run in range(NUMBER_OF_RUNS):
        X_train = []
        X_test = []
        y_train = []
        y_test = []

        for k in range(NUMBER_OF_RUNS):
            if(run == k):
                X_test.extend(normal[k])
                y_test.extend(normalclass[k])
                X_test.extend(noduled[k])
                y_test.extend(noduledclass[k])
            else:
                X_train.extend(normal[k])
                y_train.extend(normalclass[k])
                X_train.extend(noduled[k])
                y_train.extend(noduledclass[k])



        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        clf = svm.SVC(kernel='linear')#, probability=True, random_state=random_state)
        y_score = clf.fit(X_train, y_train).decision_function(X_test)
        y_score = y_score.tolist()
        print(run)
        with open(str(F) + str(run) + ".txt", 'w') as f:
            f.write('%s\n' % str(y_score)[1:-1])
