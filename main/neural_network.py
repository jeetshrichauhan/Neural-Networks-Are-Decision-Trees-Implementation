import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

def build_nn(input_dim, output_dim, task="regression"):
    model = Sequential([
        Dense(16, activation="relu", input_shape=(input_dim,)),
        Dense(16, activation="relu"),
        Dense(output_dim, activation="softmax" if task == "classification" else "linear")
    ])

    loss = "mse" if task == "regression" else "sparse_categorical_crossentropy"
    model.compile(optimizer="adam", loss=loss, metrics=["mae"] if task == "regression" else ["accuracy"])
    return model

def train_nn(model, x_train, y_train, epochs=50, batch_size=32):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def evaluate_nn(model, x_test, y_test, task="regression"):
    predictions = model.predict(x_test)
    if task == "classification":
        predictions = predictions.argmax(axis=1)  # Get class labels
        accuracy = accuracy_score(y_test, predictions)
        print("Neural Network Classification Report:")
        print(classification_report(y_test, predictions))
        return accuracy
    else:
        mse = mean_squared_error(y_test, predictions)
        print(f"Neural Network MSE: {mse}")
        return mse
