'''
TODO
- Encode PUBCHEM_COORDINATE_TYPE better
- Tune hyperparameters

Basic Execution: py driver.py -d chemical_compounds.csv
Detailed Execution: py driver.py -d chemical_compounds.csv -p

'''

# Imports
from sklearn import svm
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
import argparse
import sys


def main():

    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Add CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, help='File name of target dataset.')
    parser.add_argument('-p', '--procedural', action='store_true', help='View detailed messages.')
    opts = parser.parse_args()

    # Get dataset file name
    if opts.dataset:
        dataset_file_name = str(opts.dataset)
    elif opts.dataset == None:
        print("Please specify target dataset. Exiting...")
        sys.exit()

    # Build dataframe and remove unusable data
    if opts.procedural:
        print('\nExtracting data...')
    df = pd.read_csv(dataset_file_name)
    if opts.procedural:
        print('Removing unusable data...')
    df.drop(labels='PUBCHEM_COORDINATE_TYPE', axis=1, inplace=True)
    df.dropna(inplace=True)
    if opts.procedural:
        print('Data extracted.')

    # Train test split
    if opts.procedural:
        print('\nSplitting data into 80/20 train-test split...')
    x_data = df.loc[:, df.columns != 'Class']
    y_data = df.loc[:, df.columns == 'Class']
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=True)
    if opts.procedural:
        print('Length of training sample: ' + str(len(y_train)))
        print('Length of testing sample: ' + str(len(y_test)))

    # Build SVM model
    if opts.procedural:
        print('\nBuilding and fitting SVM model...')
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    if opts.procedural:
        print('Created and fitted SVM model.')

    # Predict
    if opts.procedural:
        print('\nPredicting...')
    y_preds = clf.predict(X_test)
    prediction_df = pd.DataFrame()
    prediction_df['Prediction'] = y_preds
    prediction_df['Validation'] = y_test['Class'].to_numpy()
    if opts.procedural:
        print('Predictions: ' + str(y_preds))

    # Precision, Recall, Accuracy
    if opts.procedural:
        print('\nAnalyzing predictions...')
    precision = precision_score(y_test['Class'].to_numpy(), y_preds)
    recall = recall_score(y_test['Class'].to_numpy(), y_preds)
    accuracy = accuracy_score(y_test['Class'].to_numpy(), y_preds)

    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("Accuracy: " + str(accuracy))
    return prediction_df, precision, recall, accuracy


if __name__ == '__main__':
    main()