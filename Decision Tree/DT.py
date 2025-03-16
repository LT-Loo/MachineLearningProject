# Decision Tree

import csv
import random
import math
import matplotlib.pyplot as plt
import pandas as pd

#### DECISION TREE ####
## Class structure to create decision node and leaf node
class Node():
    def __init__(self, decision = None, trueChild = None, falseChild = None, leaf = None, parentData = None):
        self.decision = decision
        self.trueChild = trueChild
        self.falseChild = falseChild
        self.leaf = leaf
        self.parentData = parentData # To store parent's node most frequent class

## Calculate entropy of decision node
def nodeEntropy(data, attributes):
    classes = data.pivot_table(index = attributes[0], aggfunc = "size") # Summarize numbers of two parties
    entropy = 0
    for n in classes:
        entropy -= (n/len(data)) * math.log2(n/len(data))

    return entropy, classes

## Calculate entropy of split
def splitEntropy(data, attribute, target):
    categorized = data.pivot_table(index = target, columns = attribute, aggfunc = "size", fill_value = 0) # Summarize votings of two parties 
    entropy = 0
    for i in categorized:
        splEnt = 0
        total = categorized[i].sum()
        for j in categorized[i]:
            if j == 0: continue
            splEnt -= (j / total) * math.log2(j / total)
        entropy += splEnt * (total / len(data))
   
    return entropy

## Find the most appropriate attribute to split
def splitAttribute(data, attributes):
    minEnt = math.inf
    best = None
    for a in attributes:
        if a == "party": continue
        ent = splitEntropy(data, a, attributes[0])
        if ent < minEnt: # If entropy smaller than current smallest entropy, replace
            minEnt = ent
            best = a

    return best, minEnt

## Generate tree using training set
def buildTree(data, attributes, decisionNode):
    nodeEnt, classes = nodeEntropy(data, attributes)
    if nodeEnt < 0.2: # If node entropy < 0.2, create leaf node
        decisionNode.leaf = classes.idxmax()
        return

    i, minEnt = splitAttribute(data, attributes) # Find best attribute to split

    decisionNode.decision = i
    if i: 
        for vote in pd.unique(data[i]):
            splitData = data[data[i] == vote] # Split dataset into two sub-dataset, voted and unvoted
            if len(splitData) == len(data) or len(splitData) == 0:
                if classes['democrat'] == classes['republican']: decisionNode.leaf = decisionNode.parentData
                else: decisionNode.leaf = classes.idxmax()
                return
            if vote == 1: # Create child node for voted dataset
                decisionNode.trueChild = Node(parentData = classes.idxmax())
                buildTree(splitData, attributes, decisionNode.trueChild) # Continue to build tree
            else: # Create child node for unvoted dataset
                decisionNode.falseChild = Node(parentData = classes.idxmax())
                buildTree(splitData, attributes, decisionNode.falseChild) # Continue to build tree

## Predict class of sample using decision tree built from training set
def testTree(node, test):
    if node.leaf: return node.leaf # If reach leaf node, return predicted class
    
    # Else, continue path 
    if test[node.decision] == 1: return testTree(node.trueChild, test)
    else: return testTree(node.falseChild, test)

## Run entire algorithm of Decision Tree
def implementDT(trainSet, testSet, attributes):
    predicted = []
    rootNode = Node() # Create root node
    buildTree(subSet, attributes, rootNode) # Generate tree from training set

    for i in range(len(testSet)): # Test every tuple in testing set
        predicted.append(testTree(rootNode, testSet.iloc[i]))

    return predicted


#### K-NEAREST NEIGHBOURS ####
## Hamming distance function
def hammingDist(x1, x2):
    dist = 0
    for i in range(1, len(x1)):
        dist += abs(x1[i] - x2[i])

    return dist

## Get k number of nearest neighbours
def kNN(trainSet, sample, k):
    copyData = list(trainSet)
    neighbours = list()
    # Calculate distance between sample and every data in trainSet
    for ind, item in enumerate(copyData): 
        neighbours.append((ind, hammingDist(item, sample)))

    neighbours.sort(key = lambda x : x[1]) # Sort neighbours in increasing distance order
    nearest = list()
    for i in range(k): # Store first k neighbours into a list
        nearest.append(copyData[neighbours[i][0]])

    return nearest

## Implement entire algorithm of kNN
def implementKNN(trainSet, testSet):
    predicted = list()
    k = int(math.sqrt(len(trainSet)))
    for sample in testSet:
        nearest = kNN(trainSet, sample, k) # Get k nearest neighbours
        testType = [n[0] for n in nearest] # Store classes of all nearest neighbours into a list
        predicted.append(max(set(testType), key = testType.count)) # Most frequent class is predicted class

    return predicted

#### PERFORMANCE EVALUATION ####
## Calculate precision, recall and f-measure of both parties
def performanceEval(testSet, predicted, attributes):
    trueR = 0; falseR = 0; trueD = 0; falseD = 0
    for i in range(len(testSet)):
        if predicted[i] == testSet.iloc[i][attributes[0]]:
            if predicted[i] == "republican": trueR += 1
            else: trueD += 1
        else:
            if predicted[i] == "democrat": falseD += 1
            else: falseR += 1

    precisionR = (trueR / (trueR + falseR)) * 100
    precisionD = (trueD / (trueD + falseD)) * 100
    recallR = (trueR / (trueR + falseD)) * 100
    recallD = (trueD / (trueD + falseR)) * 100
    fmeasureR = ((2 * precisionR * recallR) / (precisionR + recallR))
    fmeasureD = ((2 * precisionD * recallD) / (precisionD + recallD))
    
    return precisionR, precisionD, recallR, recallD, fmeasureR, fmeasureD

## Plot learning curves for both DT and kNN
def drawGraph(sizeSet, accuracyDT, accuracyKNN):
    plt.plot(sizeSet, accuracyDT, label = "Decision Tree")
    plt.plot(sizeSet, accuracyKNN, label = "K-Nearest Neighbours")
    plt.xlabel("Size of Training Set")
    plt.ylabel("Accuracy(%)")
    plt.title("The Learning Curves of Decision Tree and K-Nearest Neighbours")
    plt.legend()
    plt.show()


#### Main Program ####
data = pd.read_csv('votes.csv') # Read file
attributes = data.columns.tolist()

# Split data into 30% testing set and 70% training set
testSet = data.sample(frac = 0.3)
trainSet = data.drop(testSet.index)

testList = testSet.values.tolist() # Convert testing set into lists for kNN implementation

# Variables for learning curve purpose
subSet = pd.DataFrame()
accuracyDT = []
accuracyKNN = []
size = []

# While training set is not empty
while not trainSet.empty: 
    addSet = trainSet.sample(n = 5) # Randomly choose 5 samples from training set
    subSet = pd.concat([subSet, addSet]) # Add samples into subset
    subList = subSet.values.tolist() # Convert subset into lists for kNN implementation
    size.append(len(subSet))
    trainSet = trainSet.drop(addSet.index) # Remove samples added from training set

    predictDT = implementDT(subSet, testSet, attributes) # Run algorithm of Decision Tree
    predictKNN = implementKNN(subList, testList) # Run algorithm of kNN

    # Calculate accuracy for both classifiers
    correctDT = 0; correctKNN = 0
    for i in range(len(testSet)):
        if predictDT[i] == testSet.iloc[i][attributes[0]]: correctDT += 1
        if predictKNN[i] == testSet.iloc[i][attributes[0]]: correctKNN += 1   
    accuracyDT.append((correctDT / len(testSet)) * 100)
    accuracyKNN.append((correctKNN / len(testSet)) * 100)
    print("Size:", len(subSet), "; Accuracy of DT =", accuracyDT[-1], "%; Accuracy of KNN =", accuracyKNN[-1], "%")

    if trainSet.empty: # If all samples have been added into subset
        # Calculate precision, recall and f-measure for each party and each classifier
        dtPR, dtPD, dtRR, dtRD, dtFR, dtFD = performanceEval(testSet, predictDT, attributes)
        knnPR, knnPD, knnRR, knnRD, knnFR, knnFD = performanceEval(testSet, predictKNN, attributes)

        # Print output on terminal/screen   
        print("\n==PERFORMANCE EVALUATION==")
        print("Decision Tree:")
        print("Republican:\tPrecision =", dtPR, "%; Recall =", dtRR, "%; F-measure =", dtFR, "%")
        print("Democrat:\tPrecision =", dtPD, "%; Recall =", dtRD, "%; F-measure =", dtFD, "%")
        print("Average:\tPrecision =", (dtPR + dtPD) / 2, "%; Recall =", (dtRR + dtRD) / 2, "%, F-measure =", (dtFR + dtFD) / 2, "%")

        print("K-Nearest Neighbours:")
        print("Republican:\tPrecision =", knnPR, "%; Recall =", knnRR, "%; F-measure =", knnFR, "%")
        print("Democrat:\tPrecision =", knnPD, "%; Recall =", knnRD, "%; F-measure =", knnFD, "%")
        print("Average:\tPrecision =", (knnPR + knnPD) / 2, "%; Recall =", (knnRR + knnRD) / 2, "%, F-measure =", (knnFR + knnFD) / 2, "%")

        drawGraph(size, accuracyDT, accuracyKNN) # Plot graph
