import pandas as pd

def compute(matrix):
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
    
    avgPrecision=0
    avgRecall=0

    # Taking Average
    for i,j,k in zip(rowSum,colSum,diag):
        avgPrecision += k/j
        avgRecall += k/i
    avgPrecision /= n
    avgRecall /= n

    f1temp=n*[0]
    t1=n*[0]
    t2=n*[0]
    for i in range(n):
        t1[i] = diag[i]/rowSum[i]
        t2[i] = diag[i]/colSum[i]
    
    for i in range(n):
        f1temp[i] = 2*t1[i]*t2[i]/(t1[i]+t2[i])
    
    f1Score = 0
    for i in range(n):
        f1Score += f1temp[i]
    f1Score /= n
    
    accuracy = diagSum/totalSum

    # Jaccard score
    jaccard = 0
    for i in range(n):
        jaccard += diag[i]/(rowSum[i]+colSum[i]-diag[i])
    jaccard /= n

    # Specificity
    specificity = 0
    for i in range(n):
        specificity += (totalSum - rowSum[i]-colSum[i] + diag[i])/(totalSum - rowSum[i])
    specificity /= n

    return (avgPrecision,avgRecall,f1Score,accuracy,jaccard,specificity)

data = pd.read_excel("/mnt/z/program/files/confusion matrix-- face-yale.xlsx")
dataList = data.values.tolist()
for i in range(len(dataList)):
    dataList[i] = dataList[i][1:]

#dataList = [ [50,3,0,0],[26,8,0,1],[20,2,4,0],[12,0,0,1] ]

avgPrecision,avgRecall,f1Score,accuracy,jaccard,specificity = compute(dataList)

print("Average Precession  : ",avgPrecision)
print("Average Recall      : ",avgRecall)
print("F1 Score            : ",f1Score)
print("Accuracy            : ",accuracy)
print("Jaccard Sore        : ",jaccard)
print("Average Specificity : ",specificity)