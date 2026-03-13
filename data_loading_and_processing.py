from sklearn.datasets import load_digits
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import pandas as pd


def loadCategoricalData():
    #Loading categorical data
    return fetch_ucirepo(id=73)

def loadNumericalData():
    #Loading numerical data
    return load_digits()

def extractNumericalFeatures(pure_numerical_data):
    #Extracting the features 
    X = pure_numerical_data.data
    Y = pure_numerical_data.target

    X_train_num, X_test_num, Y_train_num, Y_test_num = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train_num, X_test_num, Y_train_num, Y_test_num

def extractCategoricalFeatures(pure_categorical_data):
    #Extracting the features 
    X = pure_categorical_data.data.features
    Y = pure_categorical_data.data.targets
    df1 = pd.DataFrame(X)
    df1 ['target'] = Y
    return df1

def processCategoricalData(df):
    #Processing the categorical data
    df_processed = df.apply(lambda x: x.fillna(x.mode()[0]))    

    # train-test split
    X = df_processed.drop('target', axis=1).values
    Y = df_processed['target'].values
    X_train_cat, X_test_cat, Y_train_cat, Y_test_cat = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train_cat, X_test_cat, Y_train_cat, Y_test_cat

def dataLoadingAndExtractingNumericalFeatures():
    pure_numerical_data = loadNumericalData()
    return extractNumericalFeatures(pure_numerical_data)

def dataLoadingAndExtractingCategoricalFeatures():
    pure_categorical_data = loadCategoricalData()
    df = extractCategoricalFeatures(pure_categorical_data)
    return processCategoricalData(df)