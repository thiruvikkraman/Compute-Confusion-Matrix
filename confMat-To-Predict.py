from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd

import argparse

def matrixToValues(matrix):
    # Obtain yPred and yActual from confusion matrix
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

def calcSpeci(matrix):
    # Calculate Specificity given confusion matrix
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




if __name__ == "__main__":

    
    file = "confusion matrix-- face-new.xlsx"

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help = "Location of file containing the confusion matrix")

    args = parser.parse_args()

    if ( args.file ):
        file = args.file
    else:
        print("\nThis is a sample Confusion matrix. Replace \"confusion matrix-- face-new.xlsx\" ")

    data = pd.read_excel(file)
    dataList = data.values.tolist()
    for i in range(len(dataList)):
        dataList[i] = dataList[i][1:]
    yTrue , yPred = matrixToValues(dataList)


    print("Average Precision   : ", precision_score(yTrue,yPred,average='macro') )
    print("Average Recall      : ",recall_score(yTrue,yPred,average='macro'))
    print("F1 Score - macro    : ",f1_score(yTrue,yPred,average='macro'))
    print("Accuracy            : ",accuracy_score(yTrue,yPred))
    print("Jaccard Sore        : ",jaccard_score(yTrue,yPred,average='macro'))