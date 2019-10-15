import read as exp
sysnum = 2
__doc = 'system2.txt'
svm,xgboost,knn,families=exp.info(sysnum)
print(svm)
print(knn)
print(xgboost)
print(families)

def get(c,arg):
    #print(arg)
    try:
        val = str(c[arg])+'\%'
    except:
        val='-\\'
    return val


'''
string = '\documentclass{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage{multirow}\n\\begin{document}'
string = string+'\n\\begin{table}\n\\centering\n\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|}\n\\hline\n\\multicolumn{3}{|c|}{\multirow{2}{*}{}}\n'
string = string+'& \multicolumn{3}{c|}{KNN}\n& \multicolumn{3}{c|}{SVM}\n& \multicolumn{3}{c|}{XGBOOST} \\\\ \cline{4-12}\n\\multicolumn{3}{|c|}{}\n& \multicolumn{1}{c|}{1}\n& \multicolumn{1}{c|}{2}\n& \multicolumn{1}{c|}{3}\n& \multicolumn{1}{c|}{1}\n'
string = string+'& \multicolumn{1}{c|}{2}\n& \multicolumn{1}{c|}{3}\n& \multicolumn{1}{c|}{1}\n& \multicolumn{1}{c|}{2}\n& \multicolumn{1}{c|}{3} \\\\ \hline\n'
new = '\n\\begin{table}\n\\centering\n\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|}\n\\hline\n\\multicolumn{3}{|c|}{\multirow{2}{*}{}}\n'
new = new + '& \multicolumn{3}{c|}{KNN}\n& \multicolumn{3}{c|}{SVM}\n& \multicolumn{3}{c|}{XGBOOST} \\\\ \cline{4-12}\n\\multicolumn{3}{|c|}{}\n& \multicolumn{1}{c|}{1}\n& \multicolumn{1}{c|}{2}\n& \multicolumn{1}{c|}{3}\n& \multicolumn{1}{c|}{1}\n'
new = new + '& \multicolumn{1}{c|}{2}\n& \multicolumn{1}{c|}{3}\n& \multicolumn{1}{c|}{1}\n& \multicolumn{1}{c|}{2}\n& \multicolumn{1}{c|}{3} \\\\ \hline\n'
'''


string = '\documentclass{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage{multirow}\n\\begin{document}'
string = string+'\n\\begin{table}\n\\centering\n\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|}\n\\hline\n\\multicolumn{3}{|c|}{\multirow{2}{*}{}}\n'
string = string+'& \multicolumn{3}{c|}{KNN}\n& \multicolumn{3}{c|}{SVM}\n& \multicolumn{3}{c|}{XGBOOST} \\\\ \cline{4-12}\n\\multicolumn{3}{|c|}{}\n& \multicolumn{1}{c|}{4}\n& \multicolumn{1}{c|}{5}\n& \multicolumn{1}{c|}{6}\n& \multicolumn{1}{c|}{1}\n'
string = string+'& \multicolumn{1}{c|}{5}\n& \multicolumn{1}{c|}{6}\n& \multicolumn{1}{c|}{4}\n& \multicolumn{1}{c|}{5}\n& \multicolumn{1}{c|}{6} \\\\ \hline\n'
new = '\n\\begin{table}\n\\centering\n\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|}\n\\hline\n\\multicolumn{3}{|c|}{\multirow{2}{*}{}}\n'
new = new + '& \multicolumn{3}{c|}{KNN}\n& \multicolumn{3}{c|}{SVM}\n& \multicolumn{3}{c|}{XGBOOST} \\\\ \cline{4-12}\n\\multicolumn{3}{|c|}{}\n& \multicolumn{1}{c|}{4}\n& \multicolumn{1}{c|}{5}\n& \multicolumn{1}{c|}{6}\n& \multicolumn{1}{c|}{4}\n'
new = new + '& \multicolumn{1}{c|}{5}\n& \multicolumn{1}{c|}{6}\n& \multicolumn{1}{c|}{4}\n& \multicolumn{1}{c|}{5}\n& \multicolumn{1}{c|}{6} \\\\ \hline\n'


#print(string)
with open(__doc,'w') as doc:
    doc.write('%s\n' % str(string))

#string = '\multicolumn{2}{|c|}{\multirow{3}{*}{DB1}} &'+str(svm_detail[0][2])+''
j=0
C = 3
for family in families:
    #print(svmdt,xgboostdt)
    j = j+1
    if j==15:
        string = '\n\\end{tabular}\n\\caption{Results}\n\\end{table}'
        string = string+new
        with open(__doc,'a') as doc:
            doc.write('%s\n' % str(string))
        j=0
    for i in range(0,3):
        #print(i)
        if i==0:
            imtype='ad'
            #\multicolumn{2}{|c|}{\multirow{3}{*}{bior11}}& AD
            string = '\multicolumn{2}{|c|}{\multirow{3}{*}{'
            string = string+str(family)
            string = string+'}}}}&{}'.format(imtype)
            #print(string)
        elif i==1:
            imtype='da'
            #\multicolumn{2}{|c|}{}&AD
            string = '\multicolumn{2}{|c|}{}'+'&{}'.format(imtype)
            #print(string)
        else:
            imtype='dd'
            string = '\multicolumn{2}{|c|}{}'+'&{}'.format(imtype)
            #print(string)
        if i < 2:
            string = string + '\n& {} \n& {} \n& {} \n& {} \n& {} \n& {} \n& {} \n& {} \n& {} \ {}'.format(get(knn,(C+1,family,imtype)),get(knn,(C+2,family,imtype)),get(knn,(C+3,family,imtype)),get(svm,(C+1,family,imtype)),get(svm,(C+2,family,imtype)),get(svm,(C+3,family,imtype)),get(xgboost,(C+1,family,imtype)),get(xgboost,(C+2,family,imtype)),get(xgboost,(C+3,family,imtype)),'\\\\ \\cline{3-12}')
        else:
            string = string + '\n& {} \n& {} \n& {} \n& {} \n& {} \n& {} \n& {} \n& {} \n& {} \ {}'.format(get(knn,(C+1,family,imtype)),get(knn,(C+2,family,imtype)),'-',get(knn,(C+3,family,imtype)),get(svm,(C+2,family,imtype)),get(svm,(C+3,family,imtype)),get(xgboost,(C+1,family,imtype)),get(xgboost,(C+2,family,imtype)),get(xgboost,(C+3,family,imtype)),'\\\\ \\hline')

        #print(string)
        with open(__doc,'a') as doc:
            doc.write('%s\n' % str(string))
            string = ''
string = '\n\\end{tabular}\n\\caption{Results}\n\\end{table}\n\\end{document}'
with open(__doc,'a') as doc:
    doc.write('%s\n' % str(string))