import data_loading
from naive_bayes_classifier_model.py import NumericalNaiveBayes, CategoricalNaiveBayes

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

def main():
    numericalDataEvaluation()
    print("\n" + "="*50 + "\n")
    categoricalDataEvaluation()

if __name__ == "__main__":
    main()
