from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# Build and train Random Forest models
def build_random_forest_regressor(x_train, y_train):
    rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_regressor.fit(x_train, y_train)
    return rf_regressor

def build_random_forest_classifier(x_train, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_classifier.fit(x_train, y_train)
    return rf_classifier
