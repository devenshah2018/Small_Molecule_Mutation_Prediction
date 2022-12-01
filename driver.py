# Imports
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import warnings
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse

# Argument input for hyperparameters. Default to optimal model if no arguments provided.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--iterations', required=False, help='Number of iterations to run.')
parser.add_argument('-f', '--features', required=False, help='Number of features to use.')
parser.add_argument('-t', '--threshold', required=False, help='Variance threshold to use.')
parser.add_argument('-k', '--kernel', required=False, help="SVM kernel to use. 'rbf', 'linear', 'poly', 'sigmoid', 'precomputed'.")
opts = parser.parse_args()
if opts.iterations:
    NUMBER_OF_ITERATIONS = int(opts.iterations)
else:
    NUMBER_OF_ITERATIONS = 5
if opts.features:
    NUMBER_OF_FEATURES = int(opts.features)
else:
    NUMBER_OF_FEATURES = 75
if opts.threshold:
    VARIANCE_THRESHOLD = float(opts.threshold)
else:
    VARIANCE_THRESHOLD = 0.001
if opts.kernel:
    SVM_KERNEL = str(opts.kernel)
else:
    SVM_KERNEL = 'linear'
print('\nHYPERPARAMETERS:\n')
print('Number of iterations: ' + str(NUMBER_OF_ITERATIONS))
print('Number of features to use: ' + str(NUMBER_OF_FEATURES))
print('Variance threshold to use: ' + str(VARIANCE_THRESHOLD))
print('Kernel: ' + str(SVM_KERNEL))

def main(num_features, var_thresh, svm_kernel=None):

    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Build dataframe and remove instances with null attributes
    df = pd.read_csv('chemical_compounds.csv')
    original_total_tuples = df.shape[0]
    df.drop(labels=['CID'], axis=1, inplace=True)
    df.dropna(inplace=True)
    updated_total_tuples = df.shape[0]
    print(str(original_total_tuples - updated_total_tuples) + ' instances had null attributes. Instances removed.')

    # Encode non-numerical features
    non_num_column_index = []
    index = 0
    for data_type in df.dtypes:
        if data_type == 'object':
            non_num_column_index.append(index)
        index = index + 1
    non_num_columns = []
    for index in non_num_column_index:
        non_num_columns.append(df.columns[index])
    for column in non_num_columns:
        df[column] = df[column].astype('category')
        df[column] = df[column].cat.codes
    print('The following column(s) were encoded: ' + str(non_num_columns))
    del non_num_column_index
    del index
    del non_num_columns
    
    # Separate features and target
    x_data = df.loc[:, df.columns != 'Class']
    y_data = df.loc[:, df.columns == 'Class']

    # Remove constant and quasi-constant features 
    variance_threshold = VarianceThreshold(threshold=var_thresh)
    variance_threshold.fit(x_data)
    constant_columns = [column for column in x_data.columns if column not in x_data.columns[variance_threshold.get_support()]]
    x_data.drop(constant_columns, axis=1, inplace=True)
    print(str(len(constant_columns)) + ' column(s) had a variance of ' + str(VARIANCE_THRESHOLD) + ' or below. Columns were removed.\n')
    del variance_threshold
    del constant_columns

    # Use random forest to choose top 100 most important features
    all_features = x_data.columns
    random_forest = RandomForestClassifier()
    random_forest.fit(x_data, y_data)
    importances = random_forest.feature_importances_
    sorted_importances_index = np.argsort(importances)
    most_important_features = all_features[sorted_importances_index]
    most_important_features = most_important_features[-1 * num_features:]
    x_data = x_data[most_important_features]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=True)
    del x_data
    del y_data

    # Build SVM model
    clf = svm.SVC(kernel=svm_kernel)
    clf.fit(X_train, y_train)

    # Predict
    predictions = clf.predict(X_test)
    prediction_df = pd.DataFrame()
    prediction_df['Prediction'] = predictions
    prediction_df['Validation'] = y_test['Class'].to_numpy()

    # Precision, Recall, F1-Score, Accuracy
    precision = precision_score(y_test['Class'].to_numpy(), predictions)
    recall = recall_score(y_test['Class'].to_numpy(), predictions)
    f1 = f1_score(y_test['Class'].to_numpy(), predictions)
    accuracy = accuracy_score(y_test['Class'].to_numpy(), predictions)

    # ROC and AUC-ROC
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test['Class'].to_numpy(), predictions)
    plt.clf()
    plt.plot(false_positive_rate, true_positive_rate)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-Curve")
    area_under_roc = roc_auc_score(y_test['Class'].to_numpy(), predictions)

    return f1, accuracy, most_important_features, plt, area_under_roc


if __name__ == '__main__':
    iterations = NUMBER_OF_ITERATIONS

    # Initialize log book
    log_book = {}
    '''
    { 
        Iteration: {
            f1_score: float,
            accuracy: float,
            area_under_roc: float
        }
        Summary: {
            num_features: int,
            variance_threshold: float,
            svm_kernel: 'rbf', 'linear', 'poly', 'sigmoid', 'precomputed',
            average_f1_score: float,
            average_accuracy: float,
            features_present_in_all: list
        }
    '''
    all_features_used = []

    # Run model
    for i in range(iterations):
        i = i + 1
        print('\n===========Iteration ' + str(i) + '===========\n')
        if SVM_KERNEL == None:
            f1score, accuracy, features_used, roc, area_under_roc = main(num_features=NUMBER_OF_FEATURES, var_thresh=VARIANCE_THRESHOLD)
            features_used = features_used.tolist()
        else:
            f1score, accuracy, features_used, roc, area_under_roc = main(num_features=NUMBER_OF_FEATURES, var_thresh=VARIANCE_THRESHOLD, svm_kernel=SVM_KERNEL)
            features_used = features_used.tolist()
        
        # Add to log book
        log_book["Iteration " + str(i)] = {
            'f1_score': f1score,
            'accuracy': accuracy,
            'area_under_roc': area_under_roc
        }    
        all_features_used.append(features_used)

        # Save ROC curve as PNG
        roc.savefig('Iteration ' + str(i) + ' ROC Curve')
    
    # Calculate average f1 score, accuracy, and features present in all iterations
    average_f1_score = 0
    average_accuracy = 0
    average_auc = 0
    for iteration in log_book:
        average_f1_score = average_f1_score + log_book[iteration]['f1_score']
        average_accuracy = average_accuracy + log_book[iteration]['accuracy']
        average_auc = average_auc + log_book[iteration]['area_under_roc']
    average_accuracy = average_accuracy / iterations
    average_f1_score = average_f1_score / iterations
    average_auc = average_auc / iterations
    feature_intersection = set.intersection(*map(set,all_features_used))
    feature_intersection = sorted(list(feature_intersection))
    log_book["Summary"] = {
        'num_features': NUMBER_OF_FEATURES,
        'variance_threshold': VARIANCE_THRESHOLD,
        'kernel': SVM_KERNEL,
        'average_f1_score': average_f1_score,
        'average_accuracy': average_accuracy,
        'average_auc': average_auc,
        'features_present_in_all': str(feature_intersection)
    }

    # Save log book as JSON
    with open("log_book.json", "w") as write_file:
        json.dump(log_book, write_file, indent=4)

    # User messages
    print("ROC Curves are saved to working directory.")
    print("Log book is saved to working directory.")
    print("Enter 'cat log_book.json' to view performance of each iteration.")
    
    
    
