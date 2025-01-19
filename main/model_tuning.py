from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

def tune_decision_tree_regressor(x_train, y_train):
    param_grid = {"max_depth": [3, 5, 10, None],
                  "min_samples_split": [2, 5, 10],
                  "min_samples_leaf": [1, 2, 4]}
    grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_
