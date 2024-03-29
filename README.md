# Small Molecule Drug Development Driver

## Overview
The objective of the support vector machine is to identify which compounds have the ability to inhibit the BRAF V600E mutation and to programmatically determine which attributes affect the identification. Using chemical compounds retrieved from PubChem, the model is able to identify the compounds with high performance.

## Prerequisites
- Save 'braf_classifier.py', 'chemical_compounds.csv', and 'requirements.txt' in the same directory.
- Python 3.9 or higher

To install requirements, navigate to the appropriate directory in the CLI and enter the following command:

**C:\Model>py -m pip -v install -r requirements.txt**

## Usage
**C:\Model>py braf_classifier.py [-h] [-i ITERATIONS] [-f FEATURES] [-t THRESHOLD] [-k KERNEL]**

Optional Arguments:

-h Help message.

-i Number of iterations to run. Default 50

-f Number of features to use. Default 50

-t Variance threshold to use. Default 0.001

-k Kernel type to use. Default linear


## Output
The files below are saved to the directory 'Performance_Results' created in current working directory after execution:
1) Log book containing the performance of each iteration and a summary of all iterations
2) ROC Curve for each iteration (see Iteration 1 ROC Curve.png in repository).

## Basic Execution with Default Hyperparameters

**C:\Model>py braf_classifier.py**

## Tuned Execution with Desired Hyperparameters
Example hyperparameters:
- 1 iteration
- 50 features used
- 0.005 variance threshold
- RBF kernel

**C:\Model>py braf_classifier.py -i 1 -f 50 -t 0.005 -k rbf**


## Access Log Book After Execution
**C:\Model>cd Performance_Results**

**C:\Model\Performance_Results>cat log_book.json**
