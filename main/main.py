from sklearn.datasets import make_regression, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from neural_network import build_nn, train_nn
from tree_generator import extract_regression_tree, extract_classification_tree
from sklearn.metrics import mean_squared_error, accuracy_score
from model_tuning import tune_decision_tree_regressor
from models import build_random_forest_regressor, build_random_forest_classifier
import numpy as np

from visualization import (
    plot_regression_results, 
    plot_decision_boundary, 
    plot_tree_structure, 
    plot_regression_metrics, 
    plot_classification_metrics,
    plot_residuals
)

#Generate regression dataset
def generate_regression_data():
    x, y = make_regression(n_samples=1000, n_features=1, noise=0.1)
    y = y**2 #Non-linear transformation
    return x, y

#Generate classification dataset
def generate_classification_data():
    x, y =  make_moons(n_samples=1000, noise=0.1, random_state=42)
    return x, y

#Preprocess data: normalizing and splitting
def preprocessing(x, y):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

#Example
if __name__ == "__main__":
    # Regression task
    x_reg_train, x_reg_test, y_reg_train, y_reg_test = preprocessing(*generate_regression_data())
    model_reg = build_nn(input_dim=x_reg_train.shape[1], output_dim=1, task="regression")
    train_nn(model_reg, x_reg_train, y_reg_train, epochs=50)

    # Classification task
    x_cls_train, x_cls_test, y_cls_train, y_cls_test = preprocessing(*generate_classification_data())
    model_cls = build_nn(input_dim=x_cls_train.shape[1], output_dim=2, task="classification")
    train_nn(model_cls, x_cls_train, y_cls_train, epochs=50)

    print("Training completed for both tasks!")

if __name__ == "__main__":
    # --- Regression Tree ---
    # Extract decision tree for regression
    regression_tree = extract_regression_tree(model_reg, x_reg_train, y_reg_train)
    
    # Evaluate the tree on test data
    reg_tree_predictions = regression_tree.predict(x_reg_test)
    print("Regression Tree MSE:", mean_squared_error(y_reg_test, reg_tree_predictions))

    # --- Classification Tree ---
    # Extract decision tree for classification
    classification_tree = extract_classification_tree(model_cls, x_cls_train, y_cls_train)
    
    # Evaluate the tree on test data
    cls_tree_predictions = classification_tree.predict(x_cls_test)
    print("Classification Tree Accuracy:", accuracy_score(y_cls_test, cls_tree_predictions))

#---Visualization---

# Regression predictions
x_range = np.linspace(x_reg_train.min(), x_reg_train.max(), 500).reshape(-1, 1)
nn_predictions = model_reg.predict(x_range).flatten()
tree_predictions = regression_tree.predict(x_range)

# Plot regression results
plot_regression_results(x_reg_train, y_reg_train, x_range, nn_predictions, tree_predictions)

# Compare metrics
nn_test_predictions = model_reg.predict(x_reg_test).flatten()
tree_test_predictions = regression_tree.predict(x_reg_test)
plot_regression_metrics(y_reg_test, nn_test_predictions, tree_test_predictions)

# Visualize regression tree
plot_tree_structure(regression_tree, feature_names=["Feature"], task="regression")

# Plot decision boundaries
plot_decision_boundary(model_cls, x_cls_train, y_cls_train, title="Neural Network Decision Boundary")
plot_decision_boundary(classification_tree, x_cls_train, y_cls_train, title="Decision Tree Decision Boundary")

# Compare metrics
nn_test_predictions_cls = model_cls.predict(x_cls_test)
tree_test_predictions_cls = classification_tree.predict(x_cls_test)
# Visualize classification tree
plot_tree_structure(classification_tree, feature_names=["Feature 1", "Feature 2"], task="classification")


# Tune the regression tree
best_reg_tree, best_params = tune_decision_tree_regressor(x_reg_train, y_reg_train)
print(f"Best Regression Tree Parameters: {best_params}")


# Train random forest for regression
rf_regressor = build_random_forest_regressor(x_reg_train, y_reg_train)
rf_predictions = rf_regressor.predict(x_reg_test)

# Train random forest for classification
rf_classifier = build_random_forest_classifier(x_cls_train, y_cls_train)
rf_cls_predictions = rf_classifier.predict(x_cls_test)


# Plot residuals for regression
plot_residuals(y_reg_test, nn_test_predictions, title="NN Residuals")
plot_residuals(y_reg_test, tree_test_predictions, title="Tree Residuals")
