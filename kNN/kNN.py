# k Nearest Neighbours (kNN)

import csv
import random
import math
import matplotlib.pyplot as plt

#### Defined Functions ####
## Read file and store data
def readFile(filename):
    data = list()
    with open(filename, 'r') as file:
        f = csv.reader(file)
        for line in f:
            if not line: continue
            data.append(line)
    data.pop(0) # Remove the first row (Title row)

    for row in data:
        for i in range(len(row) - 1):
            row[i] = float(row[i].strip()) # Convert strings of data into float types

    return data

## Split data into two datasets (Training set and testing set)
def splitData(data, ratio):
    k = int(len(data) * ratio)
    trainSet = list(data)
    testSet = list(random.sample(data, k))
    
    for d in testSet: trainSet.remove(d) # Remove data in testing set from training set

    return trainSet, testSet

## Euclidean distance function
def euclideanDist(x1, x2):
    dist = 0.0
    for i in range(len(x1) - 1): dist += (x1[i] - x2[i]) ** 2

    return math.sqrt(dist)

## Manhattan distance function
def manhattanDist(x1, x2):
    dist = 0.0
    for i in range(len(x1) - 1): dist += abs(x1[i] - x2[i])

    return dist

## Squared X^2 distance function
def squaredX2(x1, x2):
    dist = 0.0
    for i in range(len(x1) - 1): dist += ((x1[i] - x2[i]) ** 2) / (x1[i] + x2[i])

    return dist

## Get k number of nearest neighbours
def kNN(trainSet, sample, k, distFunction):
    copyData = list(trainSet)
    neighbours = list()

    # Calculate distance between sample and every data in trainSet
    for ind, item in enumerate(copyData): 
        neighbours.append((ind, distFunction(item, sample)))
    neighbours.sort(key = lambda x : x[1]) # Sort neighbours in increasing distance order

    nearest = list()
    for i in range(k): # Store first k neighbours into a list
        nearest.append(copyData[neighbours[i][0]])

    return nearest

## Predict the classes of samples in testSet
def classification(trainSet, testSet, k, distFunc):
    predict = list()
    for sample in testSet:
        nearest = kNN(trainSet, sample, k, distFunc) # Get k nearest neighbours
        testType = [n[-1] for n in nearest] # Store classes of all nearest neighbours into a list
        predict.append(max(set(testType), key = testType.count)) # Most frequent class is predicted class

    return predict

## Calculate accuracy
def findAccuracy(actual, predict, whichSet, k):
    correct = 0
    for i in range(len(actual)):
        if actual[i][-1] == predict[i]:
            correct += 1
            print("Sample class: ", actual[i][-1], ", Prediction Class: ", predict[i], ", Prediction correct: ", True, sep = "")
        else: print("Sample class: ", actual[i][-1], ", Prediction Class: ", predict[i], ", Prediction correct: ", False, sep = "")

    accuracy = (correct / len(actual)) * 100
    kValue = "For k = " + str(k) + ","
    print(kValue, whichSet, "accuracy:", accuracy, "%") # Print accuracy on terminal/screen

    return accuracy

## Implement entire algorithm of k-NN
def implementKNN(trainSet, testSet, whichSet, distFunc):
    kValues = list()
    accuracy = list()
    for k in range(1, len(trainSet) + 1, 2):
        if k % 3 == 0: continue
        kValues.append(k) # Store all k-values
        predict = classification(trainSet, testSet, k, distFunc) # Predict classes of samples in testSet
        accuracy.append(findAccuracy(testSet, predict, whichSet, k)) # Store accuracy respect to k
        
    return kValues, accuracy

## Plot graphs according to respective distance function
def drawGraph(trainKValues, trainAccuracy, testKValues, testAccuracy, distFunc):
    plt.plot(testKValues, testAccuracy, label = "Testing Set")
    plt.plot(trainKValues, trainAccuracy, label = "Training Set")
    plt.xlabel("k Nearest Neighbours")
    plt.ylabel("Accuracy(%)")
    graphTitle = "The Accuracy of Training Set and Testing Set in Respect to k " + distFunc
    plt.title(graphTitle)
    plt.legend()
    plt.show()


#### Main Program ####
knnData = readFile('iris.csv')
trainSet, testSet = splitData(knnData, 0.3) 

# Euclidean distance function
trainKValues, trainAccuracy = implementKNN(trainSet, trainSet, "Training set", euclideanDist)
testKValues, testAccuracy = implementKNN(trainSet, testSet, "Testing set", euclideanDist)
drawGraph(trainKValues, trainAccuracy, testKValues, testAccuracy, "(Euclidean Distance)")

# Manhattan distance function
trainKValues, trainAccuracy = implementKNN(trainSet, trainSet, "Training set", manhattanDist)
testKValues, testAccuracy = implementKNN(trainSet, testSet, "Testing set", manhattanDist)
drawGraph(trainKValues, trainAccuracy, testKValues, testAccuracy, "(Manhattan Distance)")

# Squared X^2 distance function
trainKValues, trainAccuracy = implementKNN(trainSet, trainSet, "Training set", squaredX2)
testKValues, testAccuracy = implementKNN(trainSet, testSet, "Testing set", squaredX2)
drawGraph(trainKValues, trainAccuracy, testKValues, testAccuracy, "(Squared X^2)")
