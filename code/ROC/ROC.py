import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

"""
# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
"""


random_state = np.random.RandomState(0)
df = pd.read_csv('third_experment.txt')
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])



# Binarize the output

#y = label_binarize(y, classes=[0, 1])
#n_classes = y.shape[1]
#print(n_classes)



# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)


# Learn to predict each class against the other
classifier = svm.SVC(kernel='linear', probability=True,random_state=random_state)
#OneVsRestClassifier
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()



fpr[1], tpr[1], _ = roc_curve(y_test[:], y_score[:])
roc_auc[1] = auc(fpr[1], tpr[1])
print(_)

print(roc_auc[1])


# Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
print(len(_))

plt.figure()
lw = 2
#plt.scatter(fpr[1], tpr[1])
#plt.plot(fpr[1], tpr[1], color='red',lw=lw)
plt.plot(fpr[1], tpr[1], color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])



#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('response Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()