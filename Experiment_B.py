# Experiment B
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report
from naive_bayes_classifier_model import NumericalNaiveBayes
from PCA import PCA_implementation
from data_loading import dataLoadingAndExtractingNumericalFeatures

def get_baseline (X_train, X_test, Y_train, Y_test) :

    nb1 =  NumericalNaiveBayes()
    accs1 , cm1 , cr1 = nb1.evaluate (X_train , Y_train ,X_test , Y_test)
    print ('Results before applying PCA')
    print (accs1)
    print (cm1)
    print (cr1)
    return float(accs1)             


def NB_PCA (k,baseline_accuracy,X_train, X_test, Y_train, Y_test) :
    
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
    
def runExperimentB(X_train_num, X_test_num, Y_train_num, Y_test_num):
    baseline_accuracy = get_baseline(X_train_num, X_test_num, Y_train_num, Y_test_num)
    k_values = [10,20,30,40,50]
    for k in k_values :
        NB_PCA (k,baseline_accuracy,X_train_num, X_test_num, Y_train_num, Y_test_num)
