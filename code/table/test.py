import code as exp
svm,xgboost,families=exp.info()

def get(c,arg):
    try:
        val = c[arg]
    except:
        val='-'
    return val

string = '\documentclass{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage{multirow}\n\\begin{document}\n\\begin{table}\n\\centering\n\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|}\n\\hline\n\\multicolumn{3}{|c|}{\multirow{2}{*}{}}\n'
string = string+'& \multicolumn{3}{c|}{KNN}\n& \multicolumn{3}{c|}{SVM}\n& \multicolumn{3}{c|}{XGBOOST} \\\\ \cline{4-12}\n\\multicolumn{3}{|c|}{}\n& \multicolumn{1}{c|}{1}\n& \multicolumn{1}{c|}{2}\n& \multicolumn{1}{c|}{3}\n& \multicolumn{1}{c|}{1}\n'
string = string+'& \multicolumn{1}{c|}{2}\n& \multicolumn{1}{c|}{3}\n& \multicolumn{1}{c|}{1}\n& \multicolumn{1}{c|}{2}\n& \multicolumn{1}{c|}{3} \\\\ \hline\n'



#print(string)
with open('doc.txt','w') as doc:
    doc.write('%s\n' % str(string))

#string = '\multicolumn{2}{|c|}{\multirow{3}{*}{DB1}} &'+str(svm_detail[0][2])+''
for family in families:
    #print(svmdt,xgboostdt)
    for i in range(0,3):
        if i==0:
            imtype='ad'
            #\multicolumn{2}{|c|}{\multirow{3}{*}{bior11}}& AD
            string = '\multicolumn{2}{|c|}{\multirow{3}{*}{'
            string = string+str(family)
            string = string+'}}}}&{}'.format(imtype)
            print(string)
        elif i==1:
            imtype='da'
            #\multicolumn{2}{|c|}{}&AD
            string = '\multicolumn{2}{|c|}{}'+'&{}'.format(imtype)
            print(string)
        else:
            imtype='dd'
            string = '\multicolumn{2}{|c|}{}'+'&{}'.format(imtype)
        if i < 2:
            string = string + '\n& {} \%\n& {} \%\n& {} \%\n& {} \%\n& {} \%\n& {} \%\n& {} \%\n& {} \%\n& {} \% {}'.format('-','-','-',get(svm,(1,family,imtype)),get(svm,(2,family,imtype)),get(svm,(3,family,imtype)),get(xgboost,(1,family,imtype)),get(xgboost,(2,family,imtype)),get(xgboost,(3,family,imtype)),'\\\\ \\cline{3-12}')
        else:
            string = string + '\n& {} \%\n& {} \%\n& {} \%\n& {} \%\n& {} \%\n& {} \%\n& {} \%\n& {} \%\n& {} \% {}'.format('-','-','-',get(svm,(1,family,imtype)),get(svm,(2,family,imtype)),get(svm,(3,family,imtype)),get(xgboost,(1,family,imtype)),get(xgboost,(2,family,imtype)),get(xgboost,(3,family,imtype)),'\\\\ \\hline')
            
        with open('doc.txt','a') as doc:
            doc.write('%s\n' % str(string))
            string = ''
string = '\n\\end{tabular}\n\\caption{Results}\n\\end{table}\n\\end{document}'
with open('doc.txt','a') as doc:
    doc.write('%s\n' % str(string))