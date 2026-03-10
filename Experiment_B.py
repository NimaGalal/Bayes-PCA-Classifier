# Experiment B

from sklearn.metrics import confusion_matrix , accuracy_score , classification_report
from naive_bayes_classsifer_model import NumericalNaiveBayes
from pca import PCA_implementation
from dataloading import dataLoadingAndExtractingNumericalFeatures

#Passing the PC's to the Naiive bayes Classifier
#Evaluation

X_train, X_test, Y_train, Y_test = dataLoadingAndExtractingNumericalFeatures()

def get_baseline (X_train, X_test, Y_train, Y_test) :

    nb1 =  NumericalNaiveBayes()
    cm1 , accs1 , cr1 = nb1.evaluate (X_train , Y_train ,X_test , Y_test)
    print ('Results before applying PCA')
    print (cm1)
    print (accs1)
    print (cr1)
    return accs1             


def NB_PCA (k,baseline_accuracy) :
    
    Final_result_XTrain , Final_result_XTest = PCA_implementation(X_train, X_test, k)
    nb2 = NumericalNaiveBayes()
    cm2 , accs2 , cr2 = nb2.evaluate (Final_result_XTrain , Y_train ,
                              Final_result_XTest, Y_test)
    print ('Results after applying PCA')
    print (cm2)
    print (accs2)
    print (cr2)

#Comparison with baseline 
    print (f"baseline accuracy : {baseline_accuracy:.2%}")
    print (f"PCA accuracy : k= {k} {accs2:.2%}")
    if baseline_accuracy > accs2  :
       print ('baseline is more accurate than PCA ')
    else :
      print ('PCA is more accurate than baseline')
    

#k Experiments 
baseline_accuracy = get_baseline(X_train, X_test, Y_train, Y_test)
k_values = [10,20,30,40,50]
for k in k_values :
    NB_PCA (k,baseline_accuracy)
