from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, SimpleRNN

def build_lstm_model(input_shape):
    """构建LSTM模型"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_rnn_model(input_shape):
    """构建RNN模型（用于对比实验）"""
    model = Sequential([
        SimpleRNN(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        SimpleRNN(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model