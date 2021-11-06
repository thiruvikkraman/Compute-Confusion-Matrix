from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd

df = pd.read_excel("/mnt/z/Program/GitHub/compute-Confusion-Matrix/confusion matrix-- face-new.xlsx")
data = df.values.tolist()

#print(data)
y_true = data[0][0]
y_pred = data[1][0]
yt = []
yt[:0] = y_true

yp = []
yp[:0] = y_pred

print("Accuray      :", accuracy_score(yt, yp) )
#print("Precision    :", precision_score(yt, yp, average='macro'))
print("F1 score     :", f1_score(yt, yp,average='macro'))
#print("recall ", recall_score(yt, yp,average='macro') )




from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.metrics import confusion_matrix
import numpy as np


import pandas as pd

def matToVals(matrix):
    n = len(matrix)
    yActual = []
    yPred = []

    totalEntries = 0
    for i in range(n):
        for j in range(n):
            totalEntries+=matrix[i][j]
            yActual += matrix[i][j]*[i]
            yPred += matrix[i][j]*[j]
    return (yActual,yPred)

def speci(matrix):
    n = len(matrix)
    rowSum=n*[0]
    colSum=n*[0]
    diag=n*[0]
    diagSum=0
    totalSum=0

    for i in range(n):
        for j in range(n):
            rowSum[i] += matrix[i][j]
            colSum[j] += matrix[i][j]
        diag[i]=matrix[i][i]
        diagSum += diag[i]
        totalSum += rowSum[i]

    specificity = 0
    for i in range(n):
        specificity += (totalSum - rowSum[i]-colSum[i] + diag[i])/(totalSum - rowSum[i])
    specificity /= n

    return specificity

def counts_from_confusion(confusion):
    """
    Obtain TP, FN FP, and TN for each class in the confusion matrix
    """
    counts_list = []

    # Iterate through classes and store the counts
    for i in range(confusion.shape[0]):
        tp = confusion[i, i]

        fn_mask = np.zeros(confusion.shape) 
        fn_mask[i, :] = 1
        fn_mask[i, i] = 0
        fn = np.sum(np.multiply(confusion, fn_mask))

        fp_mask = np.zeros(confusion.shape)
        fp_mask[:, i] = 1
        fp_mask[i, i] = 0
        fp = np.sum(np.multiply(confusion, fp_mask))

        tn_mask = 1 - (fn_mask + fp_mask)
        tn_mask[i, i] = 0
        tn = np.sum(np.multiply(confusion, tn_mask))

        counts_list.append({'Class': i,
                            'TP': tp,
                            'FN': fn,
                            'FP': fp,
                            'TN': tn})

    return counts_list








"""

data = pd.read_excel("/mnt/z/program/files/confusion matrix-- face-yale.xlsx")
dataList = data.values.tolist()
for i in range(len(dataList)):
    dataList[i] = dataList[i][1:]

#dataList = [ [50,3,0,0],[26,8,0,1],[20,2,4,0],[12,0,0,1] ]
yTrue , yPred = matToVals(dataList)

lis = counts_from_confusion(confusion_matrix(yTrue,yPred))




print("Average Precision  : ", precision_score(yTrue,yPred,average='macro') )
print("Average Recall      : ",recall_score(yTrue,yPred,average='macro'))
print("F1 Score - macro    : ",f1_score(yTrue,yPred,average='macro'))
print("Accuracy            : ",accuracy_score(yTrue,yPred))
print("Jaccard Sore        : ",jaccard_score(yTrue,yPred,average='macro'))
print("Average Specificity : ",speci(dataList))


"""
"""
ytrue 1 2 3 2 1 2 3 2 1 2 3 2 1 2 3
ypred 1 2 1 2 3 2 1 2 3 3 2 1 2 3 1

    1   2   3
1   1   1   2   tp              tn
2   1   4   2       tn                tp
3   3   1   0           tn                  tn


1/(1+(1+2))


4+1 % 15




36 X 36



cat - cat => ture postive

not cat - dog  => ture negetive

cat - dog => false postive

dog - cat => false negetive
=================

dog - dog true p

not dog - cat => true neg


"""