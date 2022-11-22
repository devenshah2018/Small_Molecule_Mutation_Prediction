'''
TODO
- Encode PUBCHEM_COORDINATE_TYPE better
- Tune hyperparameters
'''

# Imports
from sklearn import svm
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
import warnings


def main():

    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Build dataframe and remove unusable data
    df = pd.read_csv('chemical_compounds.csv')
    df.drop(labels='PUBCHEM_COORDINATE_TYPE', axis=1, inplace=True)
    df.dropna(inplace=True)

    # Train test split
    x_data = df.loc[:, df.columns != 'Class']
    y_data = df.loc[:, df.columns == 'Class']
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=True)

    # Build SVM model
    clf = svm.SVC()
    clf.fit(X_train, y_train)

    # Predict
    y_preds = clf.predict(X_test)
    prediction_df = pd.DataFrame()
    prediction_df['Prediction'] = y_preds
    prediction_df['Validation'] = y_test['Class'].to_numpy()

    # Precision, Recall, F1-Score, Accuracy
    precision = precision_score(y_test['Class'].to_numpy(), y_preds)
    recall = recall_score(y_test['Class'].to_numpy(), y_preds)
    f1 = f1_score(y_test['Class'].to_numpy(), y_preds)
    accuracy = accuracy_score(y_test['Class'].to_numpy(), y_preds)
    
    # Visualization
    ## Display prediction dataframe
    # display(prediction_df)

    ## Display precision, recall, f1, and/or accuracy
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("Accuracy: " + str(accuracy))
    print("F1-Score: " + str(f1))


if __name__ == '__main__':
    for i in range(5):
        main()
