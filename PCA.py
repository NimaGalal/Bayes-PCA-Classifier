import numpy as np

def Standarize_Values (X) :
    mean = np.mean (X, axis=0)
    standard_deviation = np.std (X, axis=0)
    standard_deviation[standard_deviation == 0] = 1e-9 
    return  (X-mean)/standard_deviation , mean , standard_deviation


def Covariance_Matrix (standarized_matrix) :
    n = standarized_matrix.shape[0]
    return np.dot (standarized_matrix.T , standarized_matrix)/n


def Compute_Eigns (Covariance_matrix) :
     Eignvalues , Eignvectors = np.linalg.eigh (Covariance_matrix)
     return Eignvalues , Eignvectors
    

def Sort_Select (Eignvalues , Eignvectors,k) :
    sorted_indicies = np.argsort (Eignvalues )[:: -1]
    Eignvalues = Eignvalues [sorted_indicies]
    Eignvectors = Eignvectors[:,sorted_indicies]
    Top_EignVector = Eignvectors [:,: k]
    return Top_EignVector



def Project (X_std,Top_EignVector) :
    return np.dot ( X_std , Top_EignVector)


def PCA_implementation (X_train, X_test, k ) :
    X_train_std , mean , standard_deviation  = Standarize_Values (X_train)
    X_test_std = (X_test- mean)/standard_deviation 
    X_train_cov = Covariance_Matrix(X_train_std)
    Eignvalues_std , Eignvectors_std = Compute_Eigns (X_train_cov)
    Top_EignVector_std = Sort_Select (Eignvalues_std , Eignvectors_std, k )
    Final_result_XTrain = Project (X_train_std ,Top_EignVector_std)
    Final_result_XTest = Project (X_test_std ,Top_EignVector_std)
    return Final_result_XTrain , Final_result_XTest

def PCA_implementation_with_one_input (X, k ) :
    X_std , mean , standard_deviation  = Standarize_Values (X)
    X_cov = Covariance_Matrix(X_std)
    Eignvalues_std , Eignvectors_std = Compute_Eigns (X_cov)
    Top_EignVector_std = Sort_Select (Eignvalues_std , Eignvectors_std, k )
    Final_result_X = Project (X_std ,Top_EignVector_std)
    return Final_result_X
