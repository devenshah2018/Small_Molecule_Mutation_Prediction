# Small Molecule Drug Development Driver

## Overview


## Prerequisites
- Save driver.py and 'chemical_compounds.csv' in the same directory.
- Python 3.7 or higher

## Usage
C:\Model>driver.py [-h] [-i ITERATIONS] [-f FEATURES] [-t THRESHOLD] [-k KERNEL]

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

C:\Model>py driver.py

--------------------------------------------------------------------

## Tuned Execution with Desired Hyperparameters

C:\Model>py driver.py -i 1 -f 50 -t 0.005 -k rbf

--------------------------------------------------------------------

## Access log book
C:\Model>cd Performance_Results
C:\Model>cat log_book.json
