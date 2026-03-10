# Experiment B
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report
from naivebayesclassifier import NumericalNaiveBayes
from pca import PCA_implementation
from dataloading import dataLoadingAndExtractingNumericalFeatures


#Passing the PC's to the Naiive bayes Classifier
#Evaluation

X_train, X_test, Y_train, Y_test = dataLoadingAndExtractingNumericalFeatures()
def NB_PCA (k) :
    Final_result_XTrain , Final_result_XTest = PCA_implementation(X_train, X_test, k)
    nb = NumericalNaiveBayes()
    cm , accs , cr = nb.evaluate (Final_result_XTrain , Y_train ,
                              Final_result_XTest, Y_test)
    print (cm)
    print (accs)
    print (cr)

#Comparison with baseline 



#k Experiments 
k_values = [10,20,30,40,50]
for k in k_values :
    NB_PCA (k)
