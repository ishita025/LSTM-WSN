# federated_learning.py
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import clone_model
from tensorflow.keras.optimizers import Adam

from preprocess_custom import load_dataset, clean_data, normalize_data, split_data, balance_data, encode_labels
from model import create_lstm_model

# -------------------------
# Weighted Federated Averaging
# -------------------------
def federated_aggregation_weighted(local_weights, client_sizes):
    aggregated_weights = []
    total_samples = float(np.sum(client_sizes))
    for weights_tuple in zip(*local_weights):
        weighted_sum = np.zeros_like(np.array(weights_tuple[0]), dtype=np.float32)
        for w, size in zip(weights_tuple, client_sizes):
            weighted_sum += np.array(w, dtype=np.float32) * (size / total_samples)
        aggregated_weights.append(weighted_sum)
    return aggregated_weights

# -------------------------
# Model evaluation helper
# -------------------------
def evaluate_model_metrics(model, X, y_onehot):
    y_pred_proba = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_onehot, axis=1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "y_pred": y_pred,
        "y_true": y_true
    }

# -------------------------
# Main federated learning
# -------------------------
if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    # -------------------------
    # Output folder
    # -------------------------
    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Load & preprocess dataset
    # -------------------------
    file_path = r"C:\Users\Ishita\OneDrive\Desktop\MAJOR-1\CIC-IDS2017\data\CICIDS2017_merged.csv"
    df = load_dataset(file_path)
    df = clean_data(df)
    df, target_column = encode_labels(df)
    df = normalize_data(df)

    X_train_df, X_val_df, y_train_ser, y_val_ser = split_data(df, target_column)
    X_train_df, y_train_ser = balance_data(X_train_df, y_train_ser)

    # Encode for LSTM
    le = LabelEncoder()
    y_train_int = le.fit_transform(y_train_ser)
    y_val_int = le.transform(y_val_ser)
    num_classes = len(le.classes_)
    y_train = to_categorical(y_train_int, num_classes)
    y_val = to_categorical(y_val_int, num_classes)

    # Reshape for LSTM
    X_train = X_train_df.values.astype(np.float32).reshape((X_train_df.shape[0], 1, X_train_df.shape[1]))
    X_val = X_val_df.values.astype(np.float32).reshape((X_val_df.shape[0], 1, X_val_df.shape[1]))

    # -------------------------
    # Federated parameters
    # -------------------------
    num_clients = 5
    num_rounds = 20
    local_epochs = 5
    local_batch_size = 64
    learning_rate = 1e-3

    # -------------------------
    # Stratified client split
    # -------------------------
    labels_for_split = np.argmax(y_train, axis=1)
    class_indices = {cls: np.where(labels_for_split == cls)[0] for cls in np.unique(labels_for_split)}
    client_indices = [[] for _ in range(num_clients)]
    for cls, inds in class_indices.items():
        np.random.shuffle(inds)
        splits = np.array_split(inds, num_clients)
        for i, s in enumerate(splits):
            client_indices[i].extend(s.tolist())

    client_datasets, client_sizes = [], []
    for i in range(num_clients):
        inds = np.array(client_indices[i], dtype=int)
        X_c, y_c = X_train[inds], y_train[inds]
        client_datasets.append((X_c, y_c))
        client_sizes.append(len(inds))
        unique, counts = np.unique(np.argmax(y_c, axis=1), return_counts=True)
        print(f"Client {i+1}: {len(inds)} samples, class distribution: {dict(zip(unique, counts))}")

    # -------------------------
    # Initialize global model
    # -------------------------
    input_shape = (X_train.shape[1], X_train.shape[2])
    global_model = create_lstm_model(input_shape, num_classes)
    global_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

    federated_history = []

    # -------------------------
    # Federated training loop
    # -------------------------
    for rnd in range(1, num_rounds + 1):
        print(f"\n--- Federated Round {rnd}/{num_rounds} ---")
        local_weights, local_sizes = [], []

        for client_id in range(num_clients):
            X_c, y_c = client_datasets[client_id]
            local_model = clone_model(global_model)
            local_model.set_weights(global_model.get_weights())
            local_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
            local_model.fit(X_c, y_c, epochs=local_epochs, batch_size=local_batch_size, verbose=1)
            local_weights.append(local_model.get_weights())
            local_sizes.append(len(X_c))

        # Aggregate
        aggregated = federated_aggregation_weighted(local_weights, local_sizes)
        global_model.set_weights(aggregated)

        # Evaluate global model
        metrics = evaluate_model_metrics(global_model, X_val, y_val)
        print(f"Global eval — Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        federated_history.append(metrics)

    # -------------------------
    # Server fine-tuning
    # -------------------------
    global_model.fit(X_val, y_val, epochs=2, batch_size=64, verbose=1)
    final_metrics = evaluate_model_metrics(global_model, X_val, y_val)

    # -------------------------
    # Save results
    # -------------------------
    with open(os.path.join(output_dir, "federated_history.pkl"), "wb") as f:
        pickle.dump(federated_history, f)
    np.save(os.path.join(output_dir, "y_val_pred_final.npy"), final_metrics["y_pred"])
    np.save(os.path.join(output_dir, "y_val_true_final.npy"), final_metrics["y_true"])
    global_model.save(os.path.join(output_dir, "global_federated_model.h5"))

    print(f"✅ Federated global model and history saved in '{output_dir}' folder")
