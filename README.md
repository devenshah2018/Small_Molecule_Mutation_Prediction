# Small Molecule Drug Development Driver

## Overview


## Prerequisites
- Save driver.py and 'chemical_compounds.csv' in the same directory.
- Python7 or higher

## Usage
C:\Model>driver.py [-h] [-i ITERATIONS] [-f FEATURES] [-t THRESHOLD] [-k KERNEL]

Optional Arguments:

-h Help message.

-i Number of iterations to run. Default 5

-f Number of features to use. Default 75

-t Variance threshold to use. Default 0.001

-k Kernel type to use. Default 'linear'


## Output
The files below are save to the current working directory after execution:
1) Log book containing the performance of each iteration and a summary of all iterations (see sample_log_book.json in repository).
2) ROC Curve for each iteration (see Iteration 1 ROC Curve.png in repository).

## Basic Execution

C:\Model>py driver.py

--------------------------------------------------------------------

HYPERPARAMETERS
Number of iterations: 5
Number of features to use: 75
Variance threshold to use: 0.001
Kernel: linear

===========Iteration 1===========

5 instances had null attributes. Instances removed.
The following column(s) were encoded: ['PUBCHEM_COORDINATE_TYPE']
53 column(s) had a variance of 0.001 or below. Columns were removed.


===========Iteration 2===========

5 instances had null attributes. Instances removed.
The following column(s) were encoded: ['PUBCHEM_COORDINATE_TYPE']
53 column(s) had a variance of 0.001 or below. Columns were removed.


===========Iteration 3===========

5 instances had null attributes. Instances removed.
The following column(s) were encoded: ['PUBCHEM_COORDINATE_TYPE']
53 column(s) had a variance of 0.001 or below. Columns were removed.


===========Iteration 4===========

5 instances had null attributes. Instances removed.
The following column(s) were encoded: ['PUBCHEM_COORDINATE_TYPE']
53 column(s) had a variance of 0.001 or below. Columns were removed.


===========Iteration 5===========

5 instances had null attributes. Instances removed.
The following column(s) were encoded: ['PUBCHEM_COORDINATE_TYPE']
53 column(s) had a variance of 0.001 or below. Columns were removed.

--------------------------------------------------------------------

## Tuned Execution

C:\Model>py driver.py -i 1 -f 50 -t 0.005 -k rbf

--------------------------------------------------------------------

HYPERPARAMETERS
Number of iterations: 1
Number of features to use: 50
Variance threshold to use: 0.005
Kernel: rbf

===========Iteration 1===========

5 instances had null attributes. Instances removed.
The following column(s) were encoded: ['PUBCHEM_COORDINATE_TYPE']
81 column(s) had a variance of 0.005 or below. Columns were removed.

--------------------------------------------------------------------
