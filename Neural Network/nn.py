# Neural Network

import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time

#### Defined Functions ####
## Assign random initial weights and biases to the neurons
def initializeWeight(nInput, nHidden, nOutput):
    network = list()
    hidden = list()
    for i in range(nHidden): # For neurons in hidden layer
        neuron = {'weights':[np.random.randn() for j in range(nInput + 1)]}
        hidden.append(neuron)
    network.append(hidden)

    output = list()
    for i in range(nOutput): # For neurons in output layer
        neuron = {'weights':[np.random.randn() for j in range(nHidden + 1)]}
        output.append(neuron)
    network.append(output)

    return network

## Split training data into mini batches
def splitData(trainSet, size):
    batches = list()
    trainData = trainSet.copy()

    while trainData: # While training set is NOT empty
        batch = random.sample(trainData, size) # Randomly select data to be in a minibatch
        batches.append(batch)
        for data in batch: trainData.remove(data) # Remove selected data from training set

    return batches

## Create an empty network for each sample data to store info such as net, output, error etc.
def createSampleNet(nHidden, nOutput):
    sampleNet = list()

    hidden = [dict() for i in range(nHidden)] # For hidden layer
    sampleNet.append(hidden)

    output = [dict() for i in range(nOutput)] # For output layer
    sampleNet.append(output)

    return sampleNet

## Calculate total net input of neuron
def totalNetInput(weights, inputs):
    total = weights[-1] # Bias

    for i in range(len(inputs)): 
        total += inputs[i] * weights[i]

    return total

## Forward pass
def forwardPass(network, sampleNet, pixels):
    inputs = [p / 255.0 for p in pixels] # Normalize input values from input layer

    for netLayer, samLayer in zip(network, sampleNet): # Start from leftmost layer
        newInputs = list()
        for netNeuron, samNeuron in zip(netLayer, samLayer):
            samNeuron['net'] = totalNetInput(netNeuron['weights'], inputs) # Find total net input
            try: samNeuron['output'] = 1 / (1 + np.exp(-samNeuron['net'])) # Squash net input into the range of 0 and 1
            except OverflowError: samNeuron['output'] = 0
            newInputs.append(samNeuron['output'])
        inputs = newInputs

    return inputs # Return output values of output layer neurons

## Derivative of logistic function
def outDerivative(z): return z * (1 - z)

## Derivative of quadratic cost function
def quadratic(out, target): return out - target

## Derivative of cross entropy cost function
def crossEntropy(out, target): 
    if out == 0: return 0
    else: return ((1 - target) / (1 - out)) - (target / out)

## Backwards Pass (Backpropagation)
def backPropagate(network, sampleNet, target, sample, costFunc):
    for i in reversed(range(len(network))): # Start from the rightmost layer
        layer = sampleNet[i]

        if i == len(network) - 1: # If rightmost layer
            for j, neuron in enumerate(layer):
                if target == j: t = 1 # Convert output value into binary (0 or 1)
                else: t = 0
                neuron['dE/dO'] = costFunc(neuron['output'], t) # Find derivative of error to output
                neuron['dO/dN'] = outDerivative(neuron['output']) # Find derivative of output to net
                weightErrors(i, sample, neuron, sampleNet) # Calculate how a change in weight affect the total error

        else: # If not rightmost layer
            for j, neuron in enumerate(layer):
                neuron['dE/dO'] = 0 
                for bias, weight in zip(sampleNet[i + 1], network[i + 1]):
                    neuron['dE/dO'] += bias['weightE'][-1] * weight['weights'][j] # Calculate derivative of total error to output
                neuron['dO/dN'] = outDerivative(neuron['output']) # Find derivative of output to net
                weightErrors(i, sample, neuron, sampleNet) # Calculate how a change in weight affect the total error

## Find how much a change in each weight in a neuron affects the total error
def weightErrors(i, sample, neuron, sampleNet):
    neuron['weightE'] = list()

    if i - 1 < 0: # If the second leftmost layer
        for j in range(len(sample)):
            neuron['weightE'].append(neuron['dE/dO'] * neuron['dO/dN'] * (sample[j] / 255)) # Weights
        neuron['weightE'].append(neuron['dE/dO'] * neuron['dO/dN']) # Bias

    else: # If not the input layer nor the second leftmost layer
        for j in range(len(sampleNet[i - 1])):
            neuron['weightE'].append(neuron['dE/dO'] * neuron['dO/dN'] * sampleNet[i - 1][j]['output']) # Weights
        neuron['weightE'].append(neuron['dE/dO'] * neuron['dO/dN']) # Bias

## Update the weights and biases using stochastic gradient descent
def updateWeight(network, batchNets, rate):
    for i in range(len(network)): # Layer
        for j in range(len(network[i])): # Neuron
            for k in range(len(network[i][j]['weights'])): # Weight in neuron
                total = 0
                for sampleNet in batchNets: # Find total error in a minibatch
                    total += sampleNet[i][j]['weightE'][k]
                network[i][j]['weights'][k] -= rate * (total / len(batchNets)) # Update weights with average and learning rate

## Evaluate network on testing set          
def testNN(testSet, network, nHidden, nOutput):
    correct = 0

    for sample in testSet: # For each testing sample
        sampleNet = createSampleNet(nHidden, nOutput) # Create an empty network for this sample
        results = forwardPass(network, sampleNet, sample[1:]) # Forward pass to get outputs
        predict = results.index(max(results)) # Take neuron with the highest activation as prediction
        if predict == sample[0]: correct += 1

    return (correct / len(testSet)) * 100 # Return accuracy percentage

## Plot graph of test accuracy vs epoch
def drawGraph(nEpoch, accuracy):
    plt.plot(nEpoch, accuracy)
    plt.xlabel("Number of epochs")
    plt.ylabel("Test Accuracy(%)")
    plt.title("Graph of Test Accuracy and Number of Epochs")
    plt.show()


#### Main Program ####
# Read input data from command line
nInput = int(sys.argv[1])
nHidden = int(sys.argv[2])
nOutput = int(sys.argv[3])

trainSet = np.loadtxt(sys.argv[4], dtype = 'int', delimiter = ',', skiprows = 1)
trainList = trainSet.tolist() # Convert numpy array into traning list
trainList = random.sample(trainList, 5000)

testSet = np.loadtxt(sys.argv[5], dtype = 'int', delimiter = ',', skiprows = 1)
testList = testSet.tolist() # Convert numpy array into testing list
testList = random.sample(testList, 1000)

# Default value for epoch, minibatch size and learning rate
EPOCH = 30
BATCH_SIZE = 20
LEARN_RATE = 3

# Build neural network
network = initializeWeight(nInput, nHidden, nOutput) # Initialize weights and biases to all neurons
minibatches = splitData(trainList, BATCH_SIZE) # Split training set into minibatches

accuracy = []
nEpoch = []
count = 0

# Train and test network
for e in range(EPOCH): 
    count += 1

    for batch in minibatches: # For each minibatch
        batchNets = list() # Empty list to store sample networks in the minibatch

        for sample in batch: # For every sample in minibatch
            sampleNet = createSampleNet(nHidden, nOutput) # Create empty network for sample data
            forwardPass(network, sampleNet, sample[1:]) # Forward pass
            backPropagate(network, sampleNet, sample[0], sample[1:], crossEntropy) # Backpropagation
            batchNets.append(sampleNet) # Insert sample net into list

        updateWeight(network, batchNets, LEARN_RATE) #Update weights and biases

    nEpoch.append(count)
    accuracy.append(testNN(testList, network, nHidden, nOutput)) # Evaluate network and calculate accuracy
    print("Epoch:", count, "; Accuracy:", accuracy[-1], "%") # Print results on terminal / screen

drawGraph(nEpoch, accuracy)
