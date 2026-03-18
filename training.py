# training.py (Modified for CICIDS2017)
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model import create_lstm_model
from preprocess_custom import load_dataset, clean_data, normalize_data, encode_labels
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_local_model(model, X_local, y_local, epochs=20, batch_size=32):
    """Train a local model on the given data and return history + model."""
    X_local = np.array(X_local, dtype=np.float32)
    y_local = np.array(y_local)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
    checkpoint = ModelCheckpoint(
        "centralized_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )

    history = model.fit(
        X_local, y_local,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.1,
        callbacks=[early_stop, checkpoint]
    )
    return history, model

def prepare_data(file_path, target_column="Label"):
    """Load and preprocess CICIDS2017 dataset, return train/test splits."""
    df = load_dataset(file_path)
    df = clean_data(df)
    df, target_column = encode_labels(df)  # 'label_binary'

    # Drop remaining non-numeric columns
    text_cols = [c for c in df.columns if df[c].dtype == 'object']
    df.drop(columns=text_cols, inplace=True, errors='ignore')
    print(f"✅ Dropped non-numeric columns: {text_cols}")

    # Normalize all numeric columns
    df = normalize_data(df)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode target labels
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    num_classes = len(le.classes_)
    y = to_categorical(y_int, num_classes)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Reshape for LSTM: (samples, timesteps, features)
    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train, X_test, y_train, y_test, num_classes

if __name__ == "__main__":
    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)

    file_path = r"C:\Users\Ishita\OneDrive\Desktop\MAJOR-1\CIC-IDS2017\data\CICIDS2017_merged.csv"
    X_train, X_test, y_train, y_test, num_classes = prepare_data(file_path)

    # Save preprocessed X/Y
    np.save(os.path.join(output_dir, "X_train_central.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train_central.npy"), y_train)
    np.save(os.path.join(output_dir, "X_test_central.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test_central.npy"), y_test)
    print("✅ Preprocessed train & test datasets saved as .npy files")

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape, num_classes)
    model.summary()

    history, model = train_local_model(model, X_train, y_train, epochs=20, batch_size=64)

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")

    # Save history
    with open(os.path.join(output_dir, "centralized_history.pkl"), "wb") as f:
        pickle.dump(history.history, f)

    # Save predictions
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    np.save(os.path.join(output_dir, "y_test_pred_central.npy"), y_pred)
    np.save(os.path.join(output_dir, "y_test_true_central.npy"), y_true)

    # Visualization
    plt.figure(figsize=(8,5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Centralized Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "centralized_accuracy.png"))
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Centralized Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "centralized_loss.png"))
    plt.show()

    print("✅ Centralized training complete. Model, history, and predictions saved.")
