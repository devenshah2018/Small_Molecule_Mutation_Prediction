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
import os
import shutil

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
    NUMBER_OF_ITERATIONS = 50
if opts.features:
    NUMBER_OF_FEATURES = int(opts.features)
else:
    NUMBER_OF_FEATURES = 50
if opts.threshold:
    VARIANCE_THRESHOLD = float(opts.threshold)
else:
    VARIANCE_THRESHOLD = 0.001
if opts.kernel:
    SVM_KERNEL = str(opts.kernel)
else:
    SVM_KERNEL = 'linear'
print('\n=========HYPERPARAMETERS=========\n')
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
    del variance_threshold
    del constant_columns

    # Use random forest to choose top N most important features
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
    accuracy = accuracy_score(y_test['Class'].to_numpy(), predictions)

    # ROC and AUC-ROC
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test['Class'].to_numpy(), predictions)
    plt.clf()
    plt.plot(false_positive_rate, true_positive_rate)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-Curve")
    area_under_roc = roc_auc_score(y_test['Class'].to_numpy(), predictions)
    print("Successfully executed.")

    return accuracy, most_important_features, plt, area_under_roc, precision, recall


if __name__ == '__main__':
    iterations = NUMBER_OF_ITERATIONS

    # Create directory in current working directory to store performance results
    shutil.rmtree('Performance_Results')
    try:
        os.makedirs('Performance_Results')
    except FileExistsError:
        pass
    result_destination = 'Performance_Results/'

    # Initialize log book
    log_book = {}
    all_features_used = []

    # Run model
    for i in range(iterations):
        i = i + 1
        print('\nIteration ' + str(i) + '===========\n')
        if SVM_KERNEL == None:
            accuracy, features_used, roc, area_under_roc, precision, recall = main(num_features=NUMBER_OF_FEATURES, var_thresh=VARIANCE_THRESHOLD)
            features_used = features_used.tolist()
        else:
            accuracy, features_used, roc, area_under_roc, precision, recall = main(num_features=NUMBER_OF_FEATURES, var_thresh=VARIANCE_THRESHOLD, svm_kernel=SVM_KERNEL)
            features_used = features_used.tolist()
        
        # Add to log book
        log_book["Iteration " + str(i)] = {
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'area_under_roc': area_under_roc
        }    
        all_features_used.append(features_used)

        # Save ROC curve as PNG
        roc.savefig(result_destination + 'Iteration_' + str(i) + '_ROC_Curve.png')
    
    # Calculate average f1 score, accuracy, and features present in all iterations
    average_accuracy = 0
    average_auc = 0
    average_precision = 0
    average_recall = 0
    for iteration in log_book:
        average_accuracy = average_accuracy + log_book[iteration]['accuracy']
        average_auc = average_auc + log_book[iteration]['area_under_roc']
        average_precision = average_precision + log_book[iteration]['precision']
        average_recall = average_recall + log_book[iteration]['recall']
    average_accuracy = average_accuracy / iterations
    average_precision = average_precision / iterations
    average_recall = average_recall / iterations
    average_auc = average_auc / iterations
    feature_intersection = set.intersection(*map(set,all_features_used))
    feature_intersection = sorted(list(feature_intersection))
    log_book['Summary'] = {
        'num_features': NUMBER_OF_FEATURES,
        'variance_threshold': VARIANCE_THRESHOLD,
        'kernel': SVM_KERNEL,
        'average_precision': average_precision,
        'average_recall': average_recall,
        'average_accuracy': average_accuracy,
        'average_auc': average_auc,
        'features_present_in_all': str(feature_intersection)
    }

    # Save log book as JSON
    with open(result_destination + "log_book.json", "w") as write_file:
        json.dump(log_book, write_file, indent=4)
    
    # Print detailed summary
    all_precision = []
    all_recall = []
    all_accuracy = []
    all_auc = []
    for iter in log_book:
        if iter != 'Summary':
            all_precision.append(log_book[iter]['precision'])
            all_recall.append(log_book[iter]['recall'])
            all_accuracy.append(log_book[iter]['accuracy'])
            all_auc.append(log_book[iter]['area_under_roc'])
    print('\n=============SUMMARY=============\n')
    print("Average precision: " + str(log_book["Summary"]['average_precision']))
    print("Average recall: " + str(log_book["Summary"]['average_recall']))
    print("Average accuracy: " + str(log_book["Summary"]['average_accuracy']))
    print("Average AUC-ROC: " + str(log_book["Summary"]['average_auc']))
    perfect_performing_iterations = []
    for i in range(1,NUMBER_OF_ITERATIONS):
        if all_precision[i] == all_recall[i] == all_accuracy[i] == all_auc[i] == 1.0:
            perfect_performing_iterations.append(i + 1)
    perfect_performing_iterations = str(perfect_performing_iterations)
    perfect_performing_iterations = perfect_performing_iterations[1:-1]
    if len(perfect_performing_iterations) == 0:
        perfect_performing_iterations = 'None'
    print("Iterations with perfect performance: " + perfect_performing_iterations)
    min = all_precision[0]
    index = 0
    for i in range(1,len(all_precision)):
        if all_precision[i] < min:
            min = all_precision[i]
            index = i
    print("Iteration " + str(index + 1) + " had the lowest precision: " + str(all_precision[index]))
    min = all_recall[0]
    index = 0
    for i in range(1,len(all_recall)):
        if all_recall[i] < min:
            min = all_recall[i]
            index = i
    print("Iteration " + str(index + 1) + " had the lowest recall: " + str(all_recall[index]))
    min = all_accuracy[0]
    index = 0
    for i in range(1,len(all_accuracy)):
        if all_accuracy[i] < min:
            min = all_accuracy[i]
            index = i
    print("Iteration " + str(index + 1) + " had the lowest accuracy: " + str(all_accuracy[index]))
    min = all_auc[0]
    index = 0
    for i in range(1,len(all_auc)):
        if all_auc[i] < min:
            min = all_auc[i]
            index = i
    print("Iteration " + str(index + 1) + " had the lowest accuracy: " + str(all_auc[index]))
 
    # User messages for where files are saved and accessing log book
    print('\n=========ACCESSING FILES=========\n')  
    print("ROC Curves are saved to: /Performance_Results.")
    print("Log book is saved to: /Performance_Results.")
    print("Enter 'cd Performance_Results', then 'cat log_book.json' to view performance of each iteration.")
    
