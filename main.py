import data_loading
from naive_bayes_classifier_model import NumericalNaiveBayes, CategoricalNaiveBayes
from experiment_b import NB_PCA, get_baseline
from dataloading import dataLoadingAndExtractingNumericalFeatures

def numericalDataEvaluation():
    X_train_num, X_test_num, Y_train_num, Y_test_num = data_loading.dataLoadingAndExtractingNumericalFeatures()
    nb_numerical = NumericalNaiveBayes()
    accuracy_num, cm_num, cr_num = nb_numerical.evaluate(X_train_num, Y_train_num, X_test_num, Y_test_num)
    print("Numerical Data Evaluation:")
    print(f"Accuracy: {accuracy_num}")
    print("Confusion Matrix:")
    print(cm_num)
    print("Classification Report:")
    print(cr_num)

def categoricalDataEvaluation():
    X_train_cat, X_test_cat, Y_train_cat, Y_test_cat = data_loading.dataLoadingAndExtractingCategoricalFeatures()
    nb_categorical = CategoricalNaiveBayes()
    accuracy_cat, cm_cat, cr_cat = nb_categorical.evaluate(X_train_cat, Y_train_cat, X_test_cat, Y_test_cat)
    print("Categorical Data Evaluation:")
    print(f"Accuracy: {accuracy_cat}")
    print("Confusion Matrix:")
    print(cm_cat)
    print("Classification Report:")
    print(cr_cat)

def experimentBEvaluation():
    X_train, X_test, Y_train, Y_test = dataLoadingAndExtractingNumericalFeatures()
    baseline_accuracy = get_baseline(X_train, X_test, Y_train, Y_test)
    k_values = [10, 20, 30, 40, 50]
    for k in k_values:
        NB_PCA(k,baseline_accuracy)

def main():
    numericalDataEvaluation()
    print("\n" + "="*50 + "\n")
    categoricalDataEvaluation()
    print("\n" + "="*50 + "\n")
    experimentBEvaluation()

if __name__ == "__main__":
    main()
