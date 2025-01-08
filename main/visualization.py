import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.tree import plot_tree

# Plot regression results
def plot_regression_results(x_train, y_train, x_range, nn_predictions, tree_predictions):
    plt.figure(figsize=(8, 6))
    plt.scatter(x_train, y_train, label="True Data", alpha=0.6, color="green")
    plt.plot(x_range, nn_predictions, label="Neural Network Predictions", color="blue", linewidth=2)
    plt.plot(x_range, tree_predictions, label="Decision Tree Predictions", color="red", linewidth=2)
    plt.legend()
    plt.title("Regression: Neural Network vs. Decision Tree")
    plt.xlabel("Input Feature")
    plt.ylabel("Target Value")
    plt.show()

# Plot classification decision boundary
def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    if len(Z.shape) > 1:  # For classification with probabilities
        Z = Z.argmax(axis=1)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Plot tree structure
def plot_tree_structure(tree, feature_names, task="classification"):
    plt.figure(figsize=(12, 8))
    plot_tree(tree, 
              feature_names=feature_names, 
              class_names=["Class 0", "Class 1"] if task == "classification" else None, 
              filled=True)
    plt.title(f"{task.capitalize()} Tree Structure")
    plt.show()

# Compare metrics for regression
def plot_regression_metrics(y_test, nn_predictions, tree_predictions):
    mse_nn = mean_squared_error(y_test, nn_predictions)
    mse_tree = mean_squared_error(y_test, tree_predictions)
    plt.bar(["Neural Network", "Decision Tree"], [mse_nn, mse_tree], color=["blue", "red"])
    plt.title("Regression MSE Comparison")
    plt.ylabel("Mean Squared Error")
    plt.show()
   

# Compare metrics for classification
def plot_classification_metrics(y_test, nn_predictions, tree_predictions):
    acc_nn = accuracy_score(y_test, nn_predictions.argmax(axis=1))
    acc_tree = accuracy_score(y_test, tree_predictions)
    plt.bar(["Neural Network", "Decision Tree"], [acc_nn, acc_tree], color=["blue", "red"])
    plt.title("Classification Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.show()

def plot_residuals(y_true, y_pred, title="Residual Plot", save_path=None):
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, color="purple")
    plt.axhline(0, color='red', linestyle='--')
    plt.title(title)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    if save_path:
        plt.savefig(save_path, format="png", dpi=300)
    plt.show()
