# Experiment B
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report
from naive_bayes_classifier_model import NumericalNaiveBayes
from PCA import PCA_implementation
from data_loading import dataLoadingAndExtractingNumericalFeatures
from data_loading import dataLoadingAndExtractingCategoricalFeatures
from sklearn.preprocessing import LabelEncoder
import numpy as np

#Passing the PC's to the Naiive bayes Classifier
#Evaluation

X_train, X_test, Y_train, Y_test = dataLoadingAndExtractingNumericalFeatures()
X_train_cat, X_test_cat, Y_train_cat, Y_test_cat = dataLoadingAndExtractingCategoricalFeatures()

def get_baseline (X_train, X_test, Y_train, Y_test) :

    nb1 =  NumericalNaiveBayes()
    accs1 , cm1 , cr1 = nb1.evaluate (X_train , Y_train ,X_test , Y_test)
    print ('Results before applying PCA')
    print (accs1)
    print (cm1)
    print (cr1)
    return float(accs1)             


def NB_PCA (k,baseline_accuracy) :
    
    Final_result_XTrain , Final_result_XTest = PCA_implementation(X_train, X_test, k)
    nb2 = NumericalNaiveBayes()
    accs2 , cm2 , cr2 = nb2.evaluate (Final_result_XTrain , Y_train ,
                              Final_result_XTest, Y_test)
    print ('Results after applying PCA')
    print (accs2)
    print (cm2)
    print (cr2)

#Comparison with baseline 
    print (f"baseline accuracy : {baseline_accuracy:.2%}")
    print (f"PCA accuracy : k= {k} {accs2:.2%}")
    if baseline_accuracy > accs2  :
       print ('baseline is more accurate than PCA ')
    else :
      print ('PCA is more accurate than baseline')
    


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
    print ('Results before applying PCA')
    print (accs3)
    print (cm3)
    print (cr3)
    return float(accs3) 

def NBA_PCA_categorical (k,baseline_accuracy,X_train_cat, X_test_cat,
                       Y_train_cat, Y_test_cat) :
    X_train_encoded = Encoding_categorical_data (X_train_cat)
    X_test_encoded = Encoding_categorical_data (X_test_cat)
    Final_result_XTrain , Final_result_XTest = PCA_implementation(X_train_encoded, X_test_encoded, k)
    nb4 = NumericalNaiveBayes()
    accs4 , cm4 , cr4 = nb4.evaluate (Final_result_XTrain , Y_train_cat ,
                              Final_result_XTest, Y_test_cat)
    print ('Results after applying PCA')
    print (accs4)
    print (cm4)
    print (cr4)

#Comparison with baseline 
    print (f"baseline accuracy : {baseline_accuracy:.2%}")
    print (f"PCA accuracy : k= {k} {accs4:.2%}")
    if baseline_accuracy > float(accs4)  :
       print ('baseline is more accurate than PCA ')
    else :
      print ('PCA is more accurate than baseline')



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



#k Experiments 
if __name__ == "__main__":
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
