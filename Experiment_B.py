# Experiment B
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report
from naive_bayes_classifier_model import NumericalNaiveBayes
from PCA import PCA_implementation
from sklearn.preprocessing import LabelEncoder
import numpy as np

#Passing the PC's to the Naive bayes Classifier
#Evaluation

# Data will be loaded in the main block or passed directly

def get_baseline (X_train, X_test, Y_train, Y_test) :

    nb1 =  NumericalNaiveBayes()
    accs1 , cm1 , cr1 = nb1.evaluate (X_train , Y_train ,X_test , Y_test)
    return float(accs1)             


def NB_PCA (k,baseline_accuracy, X_train, X_test, Y_train, Y_test) :
    
    Final_result_XTrain , Final_result_XTest = PCA_implementation(X_train, X_test, k)
    nb2 = NumericalNaiveBayes()
    accs2 , cm2 , cr2 = nb2.evaluate (Final_result_XTrain , Y_train ,
                              Final_result_XTest, Y_test)
    print ('Results after applying PCA')
    print (f"PCA accuracy : k= {k} {accs2:.2%}")
    return float(accs2)

def runExperimentB(X_train, X_test, Y_train, Y_test):
    baseline_accuracy = get_baseline(X_train, X_test, Y_train, Y_test)
    k_values = [10,20,30,40,50]
    best_acc = 0
    for k in k_values :
       acc = NB_PCA(k, baseline_accuracy, X_train, X_test, Y_train, Y_test)
       if acc > best_acc: best_acc = acc
    return best_acc

def Encoding_categorical_data (X) :
    encoded_words = np.zeros_like(X, dtype=float)
    for i in range (X.shape[1]): 
       le = LabelEncoder()
       encoded_words [: , i] = le.fit_transform (X [ : , i ])
    return encoded_words


def get_baseline_categorical (X_train_cat, X_test_cat, Y_train_cat, Y_test_cat) :
    nb3=  NumericalNaiveBayes()
    X_train_encoded = Encoding_categorical_data (X_train_cat)
    X_test_encoded = Encoding_categorical_data (X_test_cat)
    accs3 , cm3 , cr3 = nb3.evaluate (X_train_encoded, Y_train_cat,
                                X_test_encoded,  Y_test_cat)

def NBA_PCA_categorical (k,baseline_accuracy,X_train_cat, X_test_cat,
                       Y_train_cat, Y_test_cat) :
    X_train_encoded = Encoding_categorical_data (X_train_cat)
    X_test_encoded = Encoding_categorical_data (X_test_cat)
    Final_result_XTrain , Final_result_XTest = PCA_implementation(X_train_encoded, X_test_encoded, k)
    nb4 = NumericalNaiveBayes()
    accs4 , cm4 , cr4 = nb4.evaluate (Final_result_XTrain , Y_train_cat ,
                              Final_result_XTest, Y_test_cat)
    print ('Results after applying PCA')
    print (f"PCA accuracy : k= {k} {accs4:.2%}")


def runExperimentB_categorical(X_train_cat, X_test_cat, Y_train_cat, Y_test_cat):
    baseline_accuracy = get_baseline_categorical(X_train_cat, X_test_cat, Y_train_cat, Y_test_cat)
    k_values_cat = [5,10,15,20]
    for k in k_values_cat :
        NBA_PCA_categorical(k, baseline_accuracy, X_train_cat, X_test_cat, Y_train_cat, Y_test_cat)



def runExperimentB (X_train, X_test, Y_train, Y_test):
#for numerical data
    baseline_accuracy = get_baseline(X_train, X_test, Y_train, Y_test)
    k_values = [10,20,30,40,50]
    for k in k_values :
       NB_PCA (k,baseline_accuracy)
#for categorical data
    X_train_cat, X_test_cat, Y_train_cat, Y_test_cat = dataLoadingAndExtractingCategoricalFeatures()
    baseline_accuracy = get_baseline_categorical(X_train_cat, X_test_cat, Y_train_cat, Y_test_cat)
    k_values_cat = [5,10,15,20]
    for k in k_values_cat :
       NBA_PCA_categorical (k,baseline_accuracy,X_train_cat, X_test_cat, Y_train_cat, Y_test_cat)