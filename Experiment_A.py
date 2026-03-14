from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
from naive_bayes_classifier_model import NumericalNaiveBayes
import plotting

def ANOVASelection(X_train_num, X_test_num, Y_train_num, Y_test_num, k=2):
    """
    Applies ANOVA F-value feature selection to pick the top k features.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train_num, Y_train_num)
    X_test_selected = selector.transform(X_test_num)
    selected_indices = selector.get_support(indices=True)
    return X_train_selected, X_test_selected, selected_indices

def runExperimentA(X_train_num, X_test_num, Y_train_num, Y_test_num, k=50):
    X_train_sel, X_test_sel, selected_indices = ANOVASelection(
        X_train_num, X_test_num, Y_train_num, Y_test_num, k=k
    )
    nb_numerical = NumericalNaiveBayes()
    accuracy, cm, cr = nb_numerical.evaluate(X_train_sel, Y_train_num, X_test_sel, Y_test_num)
    plotting.plotConfusionMatrix(cm, Y_test_num, title='Confusion Matrix for feature selection')
    return accuracy
