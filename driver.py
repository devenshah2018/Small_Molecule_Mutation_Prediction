'''
TODO
- Feature selection
- Tune hyperparams
'''

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

NUMBER_OF_ITERATIONS = 1 # Number of iterations to run
NUMBER_OF_FEATURES = 75 # Number of features to use
VARIANCE_THRESHOLD = 0.001 # Variance threshold, between 0.0 and 1.0
SVM_KERNEL = 'linear' # SVM kernel to use. 'rbf', 'linear', 'poly', 'sigmoid', 'precomputed'.


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

    # Performance
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1-Score: " + str(f1))
    print("Accuracy: " + str(accuracy))
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test['Class'].to_numpy(), predictions)
    plt.plot(false_positive_rate, true_positive_rate)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-Curve")
    area_under_roc = roc_auc_score(y_test['Class'].to_numpy(), predictions)
    return num_features, var_thresh, svm_kernel, f1, accuracy, most_important_features, plt, area_under_roc


if __name__ == '__main__':
    iterations = NUMBER_OF_ITERATIONS
    log_book = {}
    '''
    { 
        iteration: [
            {
                iteration: int
                num_features: int,
                variance_threshold: float,
                svm_kernel = 'rbf', 'linear', 'poly', 'sigmoid', 'precomputed',
                precision: float,
                recall: float,
                f1_score: float,
                accuracy: float,
                features_used: list
            }
        ]
    }
    '''
    for i in range(iterations):
        i = i + 1
        print('\n===========Iteration ' + str(i) + '===========\n')
        if SVM_KERNEL == None:
            num_features, var_threshold, kernel, f1score, accuracy, features_used, roc, area_under_roc = main(num_features=NUMBER_OF_FEATURES, var_thresh=VARIANCE_THRESHOLD)
        else:
            num_features, var_threshold, kernel, f1score, accuracy, features_used, roc, area_under_roc = main(num_features=NUMBER_OF_FEATURES, var_thresh=VARIANCE_THRESHOLD, svm_kernel=SVM_KERNEL)
        log_book["Iteration " + str(i)] = {
            'num_features': num_features,
            'variance_threshold': var_threshold,
            'kernel': kernel,
            'f1_score': f1score,
            'accuracy': accuracy,
            'area_under_roc': area_under_roc,
            'features_used': str(features_used)
        }
        roc.savefig('Iteration ' + str(i) + ' ROC Curve')
    average_f1_score = 0
    average_accuracy = 0
    for iteration in log_book:
        average_f1_score = average_f1_score + log_book[iteration]['f1_score']
        average_accuracy = average_accuracy + log_book[iteration]['accuracy']
    average_accuracy = average_accuracy / NUMBER_OF_ITERATIONS
    average_f1_score = average_f1_score / NUMBER_OF_ITERATIONS
    log_book["Summary"] = {
        'average_f1_score': average_f1_score,
        'average_accuracy': average_accuracy
    }
    with open("log_book.json", "w") as write_file:
        json.dump(log_book, write_file, indent=4)
    
    
