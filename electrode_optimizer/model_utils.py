import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from .constants import DEFAULT_MODEL_PATH, DEFAULT_SCALER_PATH

def build_small_ann(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.08),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
    return model

def load_or_train_model(model_path=DEFAULT_MODEL_PATH, scaler_path=DEFAULT_SCALER_PATH,
                        csv_path=None, epochs=80, verbose=1, signals=None):
    """Load a saved Keras model and scaler, or train from CSV if provided."""
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        if signals:
            signals.log.emit(f"Loading model from {model_path} and scaler from {scaler_path} ...")
        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        return model, scaler

    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError("No pretrained model found and CSV to train from not provided.")

    if signals:
        signals.log.emit(f"Training model from CSV: {csv_path}  (this may take a while)...")
    df = pd.read_csv(csv_path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.shape[0] < 50:
        raise ValueError("CSV seems too small to train a model reliably.")

    X = df.iloc[:, :-1].values.astype(np.float32)
    Y = df.iloc[:, -1].values.astype(np.float32)
    Y = np.nan_to_num(Y, nan=0.0, posinf=np.nanmax(Y[np.isfinite(Y)]), neginf=np.nanmin(Y[np.isfinite(Y)]))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

    model = build_small_ann(X_train.shape[1])
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                        epochs=epochs, batch_size=32, verbose=verbose)

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    if signals:
        signals.log.emit(f"Model trained and saved to {model_path}; scaler saved to {scaler_path}")
    return model, scaler
