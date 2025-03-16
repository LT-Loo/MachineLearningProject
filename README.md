## Introduction
This is a machine learning project explores the characteristics of various supervised learning algorithms. It analyses and compares their learning curves and prediction accuracy to identify patterns and optimise model selection for improved performance.

## Learning Algorithms
### Decision Tree
A non-parametric supervised learning algorithm, which is utilised for both classification and regression tasks. It is a flowchart-like model that makes predictions by splitting data into branches based on feature conditions. It recursively divides data into subsets until in reaches a decision. Decision trees are easy to interpret but can overfit if not pruned properly.

### k-Nearest Neighbours (k-NN)
A simple, instance-based learning algorithm that classifies data points based on the majority class of their k nearest neightbours in the feature space. Distance functions are incorporated to measure the similarity between data points to determine the nearest neighbours.
<br><br>
Distance functions implemented include:
- Euclidean distance
- Manhattan distance
- Squared-x distance

### Neural Network
A neural network is a model inspired by the human brain, consisting of layers of interconnected neurons (nodes). It processes inputs through weighted connections and activation functions to learn complex patterns. Neural network excels in tasks like image and speech recognition.
<br><br>
Activation cost function used include:
- Quadratic cost function
- Cross entropy cost function

## Technology Used
- Language: Python
- Library: Matplot, Pandas

## Usage Guide
**Decision Tree & k-NN**<br>
Run the program with `python [filename].py`.

**Neural Network**
1. Download the training and test datasets from [here](https://drive.google.com/drive/folders/12h5rDSY49SYkegXTkv6mMQOYDvwsaTeu?usp=drive_link).
2. Run the program with
```
python [filename].py [NInput] [NHidden] [NOutput] fashion-mnist_train.csv.gz fashion-mnist_test.csv.gz
```
*where<br>
NInput = Number of neurons in input layer<br>
NHidden = Number of neurons in hidden layer<br>
NOutput = Number of neurons in output layer<br>
fashion-mnist_train.csv.gz = The training set<br>
fashion-mnist_test.csv.gz = The test set*

## Developer
Loo<br>
loo.workspace@gmail.com
