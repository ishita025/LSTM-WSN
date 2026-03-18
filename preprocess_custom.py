# preprocess_custom.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# -------------------------
# Load dataset (with chunks if large)
# -------------------------
def load_dataset(file_path, chunksize=500000):
    """Load large CSV file in chunks and concatenate into a single DataFrame."""
    print("Loading CSV in chunks...")
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"✅ Loaded {len(df)} rows")

    # Clean column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    print("✅ Cleaned column names")
    return df

# -------------------------
# Clean data
# -------------------------
def clean_data(df):
    """Basic cleaning: drop duplicates, replace inf/-inf, fill NaNs."""
    df = df.drop_duplicates()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0)
    return df

# -------------------------
# Normalize numeric columns
# -------------------------
def normalize_data(df, numeric_cols=None):
    """Normalize numeric columns to [0,1] using MinMaxScaler."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Clip extremely large values to avoid overflow
    df[numeric_cols] = df[numeric_cols].clip(lower=-1e10, upper=1e10)
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# -------------------------
# Encode labels for binary classification
# -------------------------
def encode_labels(df, target_column=None):
    """
    Detect target column (attack labels) if not specified.
    Converts to binary labels: 0 = benign, 1 = attack.
    """
    possible_cols = ['Label', 'Class', 'Attack', 'attack', 'category']
    if target_column is None:
        found = [c for c in possible_cols if c in df.columns]
        if not found:
            raise ValueError(f"No target column found. Checked: {possible_cols}")
        target_column = found[0]
        print(f"✅ Detected target column: '{target_column}'")
    else:
        if target_column not in df.columns:
            raise ValueError(f"Specified target column '{target_column}' not found in dataset. Available columns: {df.columns.tolist()}")
        print(f"✅ Using specified target column: '{target_column}'")

    # Convert to binary: 0 = benign, 1 = attack
    df[target_column] = df[target_column].apply(lambda x: 0 if str(x).lower() == 'benign' else 1)
    return df, target_column

# -------------------------
# Split dataset
# -------------------------
def split_data(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_val, y_train, y_val

# -------------------------
# Balance dataset (oversampling minority class)
# -------------------------
def balance_data(X, y):
    """
    Simple oversampling of minority class for binary classification.
    """
    class_counts = y.value_counts()
    if len(class_counts) != 2:
        print("⚠️ Warning: dataset not binary, skipping balancing")
        return X, y

    max_count = class_counts.max()
    dfs = []
    for cls in class_counts.index:
        X_cls = X[y == cls]
        y_cls = y[y == cls]
        repeats = max_count // len(X_cls)
        remainder = max_count % len(X_cls)

        X_rep = pd.concat([X_cls]*repeats + [X_cls.sample(n=remainder, replace=True)], ignore_index=True)
        y_rep = pd.concat([y_cls]*repeats + [y_cls.sample(n=remainder, replace=True)], ignore_index=True)
        dfs.append((X_rep, y_rep))

    X_bal = pd.concat([d[0] for d in dfs], ignore_index=True)
    y_bal = pd.concat([d[1] for d in dfs], ignore_index=True)
    print(f"✅ Balanced dataset to {len(X_bal)} samples")
    return X_bal, y_bal

# -------------------------
# Main (for testing)
# -------------------------
if __name__ == "__main__":
    # -------------------------
    # Specify the dataset path
    # -------------------------
    file_path = r"C:\Users\Ishita\OneDrive\Desktop\MAJOR-1\CIC-IDS2017\data\CICIDS2017_merged.csv"
    
    # -------------------------
    # Specify target column manually to avoid detection errors
    # -------------------------
    TARGET_COLUMN = "Label"

    # -------------------------
    # Load and preprocess dataset
    # -------------------------
    df = load_dataset(file_path)
    df = clean_data(df)
    df, target_column = encode_labels(df, target_column=TARGET_COLUMN)
    df = normalize_data(df)

    X_train, X_val, y_train, y_val = split_data(df, target_column)
    X_train, y_train = balance_data(X_train, y_train)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
