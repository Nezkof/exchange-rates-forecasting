from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LibLSTMTrainer:
   def __init__(self,hidden_size, output_size, X_train):
      self.lstm = Sequential([
         LSTM(hidden_size, activation='tanh', input_shape=(X_train.shape[1], 1)),
         Dense(output_size)
         ])
      self.lstm.compile(optimizer='adam', loss='mse')
      self.lstm.summary()

   def fit(self, X_train, Y_train, epochs, batch_size, X_control, Y_control):
      self.lstm.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_control, Y_control))

   def compute(self, X_control):
      return self.lstm.predict(X_control)
