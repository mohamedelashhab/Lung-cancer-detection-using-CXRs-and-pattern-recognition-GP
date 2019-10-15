import numpy as np
from sklearn import preprocessing, cross_validation, neighbors , svm
import pandas as pd
import matplotlib.pyplot as plt
import Get_scores as s
from sklearn.metrics import roc_curve, auc

#Const
POSTIVE = 1
NEGATIVE = -1
TESTSIZE = 0.2
NUMBER_OF_RUNS = int(1/TESTSIZE)
random_state = np.random.RandomState(0)
NUMBER_OF_POSTIVE_PATCHES = 1078
NUMBER_OF_PATCHES_PER_IMAGE = 1078
NUMBER_OF_NORMAL_IMAGES = 1
NUMBER_OF_FROCs = 5
TPR = [0] * 2156
MFFPI = [0] * 2156



#Read patches
df = pd.read_csv('family__bior3.9__level__1__image_type__ad.txt')
content = np.array(df.drop(['class'], 1))
classes = np.array(df['class'])




#Split patches to 5 equal lists of normal
#Split patches to 5 equal lists of noduled

content = content.tolist()
classes = classes.tolist()

for F in range(NUMBER_OF_FROCs):
    scores = []

    for i in range(5):
        scores.append(s.get_scores(str(F) + str(i) + ".txt"))


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

    ################################################################


    #Choosing thresholds values
    thr = []
    for i in range(len(scores)):
        thr.extend(scores[i])

    thr = sorted(thr)
    #thr = set(thr)
    tpr = [0] * len(thr)
    mfppi = [0] * len(thr)
    ###########################################


    #Calculating FROC
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
                T = len(noduled[k])
            else:
                X_train.extend(normal[k])
                y_train.extend(normalclass[k])
                X_train.extend(noduled[k])
                y_train.extend(noduledclass[k])


        for j in range(len(thr)):
            Th = thr[j]
            TP , FP = 0 , 0
            for i in range(len(scores[run])):
                if (scores[run][i] > Th):
                    if (y_test[i] == POSTIVE):
                        TP = TP + 1
                    if (y_test[i] == NEGATIVE):
                        FP = FP + 1

            #print(TP , FP)
            tpr[j] += TP
            mfppi[j] += FP



    for i in range(len(thr)):
        #print(tpr[i], mfppi[i]/NUMBER_OF_NORMAL_IMAGES)
        tpr[i] /= NUMBER_OF_POSTIVE_PATCHES
        mfppi[i] /= 93
        TPR[i] += tpr[i]
        MFFPI[i] += mfppi[i]



    plt.figure()
    plt.plot(mfppi,tpr,color='darkorange',lw=2)
    plt.xlabel('Mean Number of False Positive Per Image')
    plt.ylabel('True Positive Rate')
    plt.title('Free Response Receiver Operating Characteristic')
    plt.xlim([0.0, 5])
    plt.ylim([0.0, 1.0])
    plt.legend(loc="lower right")

for i in range(len(TPR)):
    TPR[i] /= NUMBER_OF_FROCs
    MFFPI[i] /= NUMBER_OF_FROCs


plt.figure()
plt.plot(MFFPI,TPR,color='red',lw=2)
plt.xlabel('Mean Number of False Positive Per Image')
plt.ylabel('True Positive Rate')
plt.title('Free Response Receiver Operating Characteristic')
plt.xlim([0.0, 5])
plt.ylim([0.0, 1.0])
plt.legend(loc="lower right")


plt.show()