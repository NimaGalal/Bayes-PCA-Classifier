import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PCA import PCA_implementation_with_one_input

def plotConfusionMatrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    classes = np.unique(classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plotDataWithPCAAndGaussianPDF(X, Y, title='Data with PCA and Gaussian PDF'):
    X_pca = PCA_implementation_with_one_input(X, 2)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=Y, palette='viridis')
    
    #Ai Implementing the Gaussian PDF plot
    # Calculate grid for contour
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    pos = np.c_[xx.ravel(), yy.ravel()]
    
    # Plot contour for each class
    classes = np.unique(Y)
    for c in classes:
        X_c = X_pca[Y == c]
        if len(X_c) < 2:
            continue
        mean = np.mean(X_c, axis=0)
        cov = np.cov(X_c, rowvar=False)
        
        # Add epsilon to diagonal for numerical stability
        cov += np.eye(2) * 1e-9
        
        diff = pos - mean
        inv_cov = np.linalg.inv(cov)
        exponent = -0.5 * np.sum((diff @ inv_cov) * diff, axis=1)
        det = np.linalg.det(cov)
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** 2 * det)
        Z = np.reshape(norm_const * np.exp(exponent), xx.shape)
        
        plt.contour(xx, yy, Z, levels=5, alpha=0.3)

    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()