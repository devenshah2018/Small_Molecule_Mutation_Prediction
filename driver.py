'''
TODO
- Feature selection
- Tune hyperparams
'''

# Imports
from sklearn import svm
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
import warnings
from sklearn.feature_selection import VarianceThreshold


def main():

    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Build dataframe and remove instances with null attributes
    df = pd.read_csv('chemical_compounds.csv')
    df.dropna(inplace=True)

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
    variance_threshold = VarianceThreshold(threshold=0.005)
    variance_threshold.fit(x_data)
    constant_columns = [column for column in x_data.columns if column not in x_data.columns[variance_threshold.get_support()]]
    x_data.drop(constant_columns, axis=1, inplace=True)
    del variance_threshold
    del constant_columns

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=True)

    # Build SVM model
    clf = svm.SVC()
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
    print("Accuracy: " + str(accuracy))
    print("F1-Score: " + str(f1))


if __name__ == '__main__':
    main()
