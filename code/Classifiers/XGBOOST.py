import numpy as np
from sklearn import preprocessing, cross_validation, neighbors , svm
import pandas as pd
import os
import time
from xgboost import XGBClassifier

'''
classifier (path, file_name,run)
Parameters path : str, the path to the directory that contain the features files.
           file_name : str, the name of the txt file that contain the  feature.
           run : int, the total number of times which the classification process should be run (training and testing).    
Description:
the classifier used here is XGBoost.

'''

def classifier(path,file_name,run):
    df = pd.read_csv(path+'/'+file_name)
    X = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])
    total = 0

    runs = run

    for i in range(0,runs):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
        clf = XGBClassifier()
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        total = total + accuracy

    total = total / runs;
    print(total * 100)
    with open('system_accuracy.txt', 'a') as f:
        f.write('%s\n' % '{}                    accuracy = {}'.format(file_name[:-3],total*100))


def main():
    path = 'F://GP/Project/svm level2 system4/files'
    for f in os.listdir(path):
        try:
            classifier(path,f,20)
        except:
            with open('system_errors_classifier.txt', 'a') as fail:
                fail.write('%s\n' % str(f))
            continue

    pass

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

    

