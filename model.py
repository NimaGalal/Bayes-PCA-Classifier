import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class NumericalNaiveBayes:
    def computePrior(self, Y_train):
        m = len(Y_train)
        classes = np.unique(Y_train)
        priors = {}
        for c in classes:
            priors[c] = np.sum(Y_train == c) / m
        return priors, classes

    def computeMean(self, X, Y, label):
        data = X[Y == label]
        mean = np.mean(data, axis=0)
        return mean

    def computeCovariance(self, X):
        data = X
        covariance = np.cov(data,rowvar=False,bias=True)
        epsilon = 1e-9
        covariance += np.eye(covariance.shape[0]) * epsilon
        return covariance

    def GaussianPDF(self, x, mean, covariance):
        diff = x - mean
        inv_cov = np.linalg.inv(covariance)
        spreadProb = -0.5 * np.sum((diff @ inv_cov) * diff, axis=1)
        determinant = np.linalg.det(covariance)
        return spreadProb - 0.5 * np.log(determinant)

    def computeProbabilities(self, X, Y, X_test):
        priors, classes = self.computePrior(Y)
        sigma = self.computeCovariance(X)
        
        log_posteriors = {}
        for c in classes:
            mean_c = self.computeMean(X, Y, c)
            log_prob_x_given_c = self.GaussianPDF(X_test, mean_c, sigma)
            log_posteriors[c] = log_prob_x_given_c + np.log(priors[c])

        # Returning dictionary of unnormalized log-posteriors
        return log_posteriors

    def predict(self, X_train, Y_train, X_test):
        X_test = np.atleast_2d(X_test)
        log_posteriors = self.computeProbabilities(X_train, Y_train, X_test)
        predictions = np.array([max(log_posteriors, key=lambda c: log_posteriors[c][i]) for i in range(X_test.shape[0])])
        return predictions

    def evaluate(self, X_train, Y_train, X_test, Y_test):
        predictions = self.predict(X_train, Y_train, X_test)
        accuracy = accuracy_score(Y_test, predictions)
        cm = confusion_matrix(Y_test, predictions)
        cr = classification_report(Y_test, predictions)
        return accuracy, cm, cr

class CategoricalNaiveBayes:
    def computePrior(self, Y_train):
        m = len(Y_train)
        classes = np.unique(Y_train)
        priors = {}
        for c in classes:
            priors[c] = np.sum(Y_train == c) / m
        return priors, classes

    def computeConditionalProbabilities(self, X, Y):
        priors, classes = self.computePrior(Y)
        probabilities = {}
        for c in classes:
            probabilities[c] = {}
            for i in range(X.shape[1]):
                probabilities[c][i] = {}
                for j in np.unique(X[:,i]):
                    probabilities[c][i][j] = np.sum((X[Y == c][:, i] == j)) / np.sum(Y == c)
        return probabilities

    def computeProbabilities(self, X, Y, X_test):
        priors, classes = self.computePrior(Y)
        probabilities = self.computeConditionalProbabilities(X, Y)
        log_posteriors = {}
        epsilon = 1e-6 # Smoothing for unseen categories
        for c in classes:
            # Initialize an array of log probabilities for each sample
            log_prob_c = np.full(X_test.shape[0], np.log(priors[c] + 1e-9))
            for i in range(X_test.shape[1]):
                col_vals = X_test[:, i]
                col_probs = np.array([probabilities[c][i].get(val, epsilon) for val in col_vals])
                log_prob_c += np.log(col_probs + 1e-9)
            log_posteriors[c] = log_prob_c

        return log_posteriors

    def predict(self, X_train, Y_train, X_test):
        X_test = np.atleast_2d(X_test)
        log_posteriors = self.computeProbabilities(X_train, Y_train, X_test)
        predictions = np.array([max(log_posteriors, key=lambda c: log_posteriors[c][i]) for i in range(X_test.shape[0])])
        return predictions

    def evaluate(self, X_train, Y_train, X_test, Y_test):
        predictions = self.predict(X_train, Y_train, X_test)
        accuracy = accuracy_score(Y_test, predictions)
        cm = confusion_matrix(Y_test, predictions)
        cr = classification_report(Y_test, predictions)
        return accuracy, cm, cr