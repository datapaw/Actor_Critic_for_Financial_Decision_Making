import os
from typing import List, Tuple

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


class AgentEnsemble:
    """Ensemble of 5 different networks that vote on best return timestamp.

    Usage:
      agent = AgentEnsemble(csv_path, window=30, horizon=5)
      agent.prepare_data()
      agent.build_models()
      agent.train_models(epochs=5)
      ts, votes = agent.predict_best_for_latest()
    """

    def __init__(self, csv_path: str, window: int = 30, horizon: int = 5):
        self.csv_path = csv_path
        self.window = window
        self.horizon = horizon
        self.df = None
        self.features = None
        self.timestamps = None

        self.X = None
        self.y = None
        self.models: List[tf.keras.Model] = []

    def load_csv(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(self.csv_path)
        self.df = pd.read_csv(self.csv_path)
        # try to keep timestamp column if present
        if 'timestamp' in self.df.columns:
            self.timestamps = pd.to_datetime(self.df['timestamp']).values

    def _select_price_column(self) -> str:
        for name in ('price', 'close', 'adj_close', 'close_price', 'last'):
            if name in self.df.columns:
                return name
        # fallback: pick first numeric column
        numeric = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric) == 0:
            raise ValueError('No numeric columns found in CSV')
        return numeric[0]

    def prepare_data(self, test_split: float = 0.1):
        if self.df is None:
            self.load_csv()

        df = self.df.copy()
        price_col = self._select_price_column()
        values = df.select_dtypes(include=[np.number]).fillna(0).values

        seqs = []
        targets = []
        timestamps = []

        n = len(values)
        for i in range(0, n - self.window - self.horizon + 1):
            x = values[i:i + self.window]
            future_base = values[i + self.window - 1]
            # compute returns for each offset in horizon using price_col index
            price_idx = list(df.select_dtypes(include=[np.number]).columns).index(price_col)
            base_price = float(values[i + self.window - 1, price_idx])
            # If base is zero, avoid divide by zero
            if base_price == 0:
                ret = [0.0 for _ in range(self.horizon)]
            else:
                ret = []
                for h in range(1, self.horizon + 1):
                    future_price = float(values[i + self.window - 1 + h, price_idx])
                    ret.append((future_price - base_price) / base_price)

            seqs.append(x)
            targets.append(ret)
            if self.timestamps is not None:
                timestamps.append(self.timestamps[i + self.window - 1 + self.horizon])

        self.X = np.array(seqs, dtype=np.float32)
        self.y = np.array(targets, dtype=np.float32)
        if len(self.X) == 0:
            raise ValueError('Not enough rows for given window and horizon')

        # simple train/test split
        split = int(len(self.X) * (1 - test_split))
        self.train_data = (self.X[:split], self.y[:split])
        self.val_data = (self.X[split:], self.y[split:])
        self.all_timestamps = np.array(timestamps) if timestamps else None

    def build_models(self):
        input_shape = self.X.shape[1:]
        horizon = self.horizon

        # 1) LSTM
        inp = layers.Input(shape=input_shape)
        x = layers.LSTM(64)(inp)
        out = layers.Dense(horizon, activation='linear')(x)
        m1 = models.Model(inp, out, name='lstm')

        # 2) GRU
        inp = layers.Input(shape=input_shape)
        x = layers.GRU(64)(inp)
        out = layers.Dense(horizon, activation='linear')(x)
        m2 = models.Model(inp, out, name='gru')

        # 3) Conv1D + LSTM
        inp = layers.Input(shape=input_shape)
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(inp)
        x = layers.MaxPooling1D(2)(x)
        x = layers.LSTM(32)(x)
        out = layers.Dense(horizon, activation='linear')(x)
        m3 = models.Model(inp, out, name='conv_lstm')

        # 4) MLP (flat)
        inp = layers.Input(shape=input_shape)
        x = layers.Flatten()(inp)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        out = layers.Dense(horizon, activation='linear')(x)
        m4 = models.Model(inp, out, name='mlp')

        # 5) Conv1D + GlobalPool
        inp = layers.Input(shape=input_shape)
        x = layers.Conv1D(64, 5, activation='relu', padding='same')(inp)
        x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling1D()(x)
        out = layers.Dense(horizon, activation='linear')(x)
        m5 = models.Model(inp, out, name='conv_pool')

        self.models = [m1, m2, m3, m4, m5]

        for m in self.models:
            m.compile(optimizer='adam', loss='mse')

    def train_models(self, epochs: int = 5, batch_size: int = 32):
        X_train, y_train = self.train_data
        X_val, y_val = self.val_data

        es = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        for i, m in enumerate(self.models):
            print(f'Training model {i+1}/{len(self.models)}: {m.name}')
            m.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[es],
                verbose=1,
            )

    def predict_for_sequence(self, seq: np.ndarray) -> List[np.ndarray]:
        # seq shape: (window, features) -> model expects (1, window, features)
        inp = np.expand_dims(seq, axis=0)
        preds = [m.predict(inp)[0] for m in self.models]
        return preds

    def predict_best_for_latest(self) -> Tuple[pd.Timestamp, dict]:
        # take last available sequence from prepared data
        if self.X is None:
            raise ValueError('Call prepare_data() first')

        latest_seq = self.X[-1]
        preds = self.predict_for_sequence(latest_seq)

        # each pred is a vector of horizon returns -> pick argmax offset
        chosen_offsets = [int(np.argmax(p)) for p in preds]

        # voting
        counts = {}
        for off in chosen_offsets:
            counts[off] = counts.get(off, 0) + 1

        # best offset is majority vote; if tie, pick highest average return across models
        max_votes = max(counts.values())
        candidates = [k for k, v in counts.items() if v == max_votes]

        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            # tie-breaker: choose candidate with highest mean predicted return
            mean_returns = {c: np.mean([p[c] for p in preds]) for c in candidates}
            chosen = max(mean_returns.items(), key=lambda x: x[1])[0]

        # map chosen offset to timestamp (if available)
        if self.all_timestamps is None:
            chosen_ts = None
        else:
            chosen_ts = self.all_timestamps[-1]

        return chosen, {'votes': counts, 'preds': preds}


if __name__ == '__main__':
    # Quick demo using the dataset in repository
    path = os.path.join('data', 'dataset', 'news_prices_combined.csv')
    agent = AgentEnsemble(path, window=30, horizon=5)
    try:
        agent.prepare_data()
        agent.build_models()
        # small training for demo; increase epochs for real runs
        agent.train_models(epochs=3, batch_size=64)
        chosen_offset, info = agent.predict_best_for_latest()
        print('Chosen offset (0-based):', chosen_offset)
        print('Vote counts:', info['votes'])
    except Exception as e:
        print('Demo failed:', e)
