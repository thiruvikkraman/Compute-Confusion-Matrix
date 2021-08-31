# Compute Confusion Matrix
*To find different metrics of a Confusion matrix.*


Lets consider a simple 3X3 confMatrix and calculate with respect to second label (Class B). 

 --  | a |  *B* |  c
--- | --- | --- | --- |
a  | TN | FP | TN
*B*  | FN | TP | FN
c  | TN | FP | TN

## Metric
### For Total Matrix
__Accuracy__ = (TP + TN)/(TP + TN + FP + FN)<br />

### For each class
Precision = TP/(TP+FP) <br />
Recall = TP / (TP+FN) <br /> 
F1 Score = 2 \* Precision \* Recalll / (Recall + Precesion)<br />
Jaccard Sore = TP / (FP + FN + TP)<br />
Specificity = TN / (TN + FP)

Average of these metric can be taken in two ways. Micro and Macro <br />
![Avg3](https://user-images.githubusercontent.com/46104814/131486304-589d1e16-a2cb-4ec9-b870-0637674f690a.PNG)<br />
Same follows for other  metic too

