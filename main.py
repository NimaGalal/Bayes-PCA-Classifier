import data_loading
from naive_bayes_classifier_model import NumericalNaiveBayes, CategoricalNaiveBayes
import plotting
import Experiment_A
import Experiment_B

def experimentOnNumericalFeatures():
    X_train_num, X_test_num, Y_train_num, Y_test_num = data_loading.dataLoadingAndExtractingNumericalFeatures()
    nb_numerical = NumericalNaiveBayes()
    accuracy_num = nb_numerical.numericalDataEvaluation(X_train_num, X_test_num, Y_train_num, Y_test_num)
    print (f"Accuracy on numerical features: {accuracy_num:.2%}")
    accuracy_exp_a = Experiment_A.runExperimentA(X_train_num, X_test_num, Y_train_num, Y_test_num)
    print (f"Accuracy on numerical features after experiment A: {accuracy_exp_a:.2%}")
    accuracy_exp_b = Experiment_B.runExperimentB(X_train_num, X_test_num, Y_train_num, Y_test_num)

def experimentOnCategoricalFeatures():
    X_train_cat, X_test_cat, Y_train_cat, Y_test_cat = data_loading.dataLoadingAndExtractingCategoricalFeatures()
    nb_categorical = CategoricalNaiveBayes()
    accuracy_cat = nb_categorical.categoricalDataEvaluation(X_train_cat, X_test_cat, Y_train_cat, Y_test_cat)
    print (f"Accuracy on categorical features: {accuracy_cat:.2%}")
def main():
    experimentOnNumericalFeatures()
    # experimentOnCategoricalFeatures()
if __name__ == "__main__":
    main()