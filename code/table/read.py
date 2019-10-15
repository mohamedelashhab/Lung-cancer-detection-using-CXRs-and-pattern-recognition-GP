import os
path = '.'
svm_details = ['system','level','family','image_type']
xgboost_details = ['system','level','family','image_type']
svm_acc = []
xgboost_acc = []
svm={}
xgboost={}
knn={}

''' system levels family image_type'''
def convertTofloat(num):
    num =  float(num)
    return float("{0:.2f}".format(num))

''' family level image_type'''
def info(sysnum):
    for fold in os.listdir(path):
        if '.' not in fold:
            if '_' in fold:
                continue
            classifier = fold
            #print(classifier)
            for txt in os.listdir('./{}'.format(fold)):
                
                system = int(txt[6])
                if system != sysnum:
                    continue
                #print(system)
                #print(txt)
                with open('./{}/{}'.format(fold,txt)) as accur:
                    allfamilies=[]
                    data = accur.readlines()[0:-1]
                    for d in data:
                        dd = d.split('\n')[0]
                        ddd = dd.split('               ')
                        #print(ddd)
                        ddd=[ddd[0].replace('.',''),ddd[1]]
                        exp = ddd[0].split('__')
                        family,level,image_type = exp[1],int(exp[3]),exp[5]
                        if family not in allfamilies:
                            #print(family)
                            allfamilies=allfamilies+[family]
                        result = ddd[1].split('=')[1].replace(' ','')
                        accuracy = convertTofloat(result)
                        #print(type(accuracy))
                        if classifier == 'svm':
                            svm[(level,family,image_type)] = accuracy
                            #svm_acc.append(accuracy)
                            #svm_details.append([system,int(level),family,image_type])
                        elif classifier == 'xgboost':
                            xgboost[(level,family,image_type)] = accuracy

                        elif classifier == 'knn':
                            knn[(level,family,image_type)] = accuracy
                            #xgboost_acc.append(accuracy)
                            #xgboost_details.append([system,int(level),family,image_type])
    #print(allfamilies)
    return (svm,xgboost,knn,allfamilies)
                        

                  
                    

                

            