from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from neural_network import build_nn, train_nn, evaluate_nn
from tree_generator import extract_classification_tree, extract_regression_tree, evaluate_tree

task = "classification"  # or "regression"

# Generate valid dataset
if task == "classification":
    x, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_informative=3, n_clusters_per_class=1, random_state=42)
else:
    from sklearn.datasets import make_regression
    x, y = make_regression(n_samples=1000, n_features=20, random_state=42)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 1. Train Neural Network
input_dim = x_train.shape[1]
output_dim = len(np.unique(y_train)) if task == "classification" else 1
nn_model = build_nn(input_dim, output_dim, task=task)
train_nn(nn_model, x_train, y_train)

# 2. Extract Decision Tree
if task == "classification":
    tree_model = extract_classification_tree(nn_model, x_train, y_train)
else:
    tree_model = extract_regression_tree(nn_model, x_train, y_train)

# 3. Evaluate and Compare
print("\nEvaluating Neural Network...")
nn_performance = evaluate_nn(nn_model, x_test, y_test, task=task)

print("\nEvaluating Decision Tree...")
tree_performance = evaluate_tree(tree_model, x_test, y_test, task=task)

# 4. Print Comparison Results
if task == "classification":
    print(f"\nNeural Network Accuracy: {nn_performance}")
    print(f"Decision Tree Accuracy: {tree_performance}")
else:
    print(f"\nNeural Network MSE: {nn_performance}")
    print(f"Decision Tree MSE: {tree_performance}")
