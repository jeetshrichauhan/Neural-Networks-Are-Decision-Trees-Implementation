from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

# Extract Decision Tree for Regression
def extract_regression_tree(nn_model, x_train, y_train, max_depth=4):
    nn_predictions = nn_model.predict(x_train).flatten()
    tree = DecisionTreeRegressor(max_depth=max_depth)
    tree.fit(x_train, nn_predictions)
    return tree

# Extract Decision Tree for Classification
def extract_classification_tree(nn_model, x_train, y_train, max_depth=4):
    nn_predictions = nn_model.predict(x_train)
    predicted_classes = nn_predictions.argmax(axis=1)
    tree = DecisionTreeClassifier(max_depth=max_depth)
    tree.fit(x_train, predicted_classes)
    return tree

# Evaluate Tree Model
def evaluate_tree(tree_model, x_test, y_test, task="regression"):
    predictions = tree_model.predict(x_test)
    if task == "classification":
        accuracy = accuracy_score(y_test, predictions)
        print("Decision Tree Classification Report:")
        print(classification_report(y_test, predictions))
        return accuracy
    else:
        mse = mean_squared_error(y_test, predictions)
        print(f"Decision Tree MSE: {mse}")
        return mse
