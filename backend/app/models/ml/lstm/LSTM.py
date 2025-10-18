import os
import numpy as np

from app.models.ml.lstm import LSTMGate
from app.models.ml.lstm.optimizers import OptimizerFactory

class LSTM: 
   def __init__(self, optimizer_type, hidden_size, features_number, output_size, learning_rate, lr_decrease_speed):      
      self.lstm_parameters = OptimizerFactory.create_optimizer(optimizer_type, hidden_size, features_number, output_size, learning_rate)
      self.hidden_size = hidden_size
      self.features_number = features_number
      self.learning_rate = learning_rate
      self.lr_decrease_speed = lr_decrease_speed
      self.nodes = []
      self.loss = np.inf

      self.c_prev = np.zeros(self.hidden_size)
      self.h_prev = np.zeros(self.hidden_size)

   def __forward(self, train_sequences):
      lstm_out = []

      for i in range(len(train_sequences)):
         self.c_prev, self.h_prev, y_out = self.nodes[i].forward(train_sequences[i], self.c_prev, self.h_prev)
         lstm_out.append(y_out)

      return lstm_out
   
   def __backward(self, y_train):
      delta_h_next = np.zeros(self.hidden_size)
      delta_c_next = np.zeros(self.hidden_size)
      avg_loss = 0

      for i in reversed(range(len(y_train))):
         delta_h_next, delta_c_next = self.nodes[i].backward(
               y_train[i], delta_h_next, delta_c_next
         )
         avg_loss += self.nodes[i].loss
      
      avg_loss = avg_loss / len(self.nodes)

      return avg_loss
   
   # def fit(self, x_train, y_train, epochs, precision):
   #    self.nodes = [LSTMGate(self.lstm_parameters, self.hidden_size, self.features_number) for _ in range(len(x_train))]

   #    epoch = 0
   #    eta0 = self.learning_rate

   #    while epoch < epochs and self.loss > precision:
   #       self.c_prev = np.zeros(self.hidden_size)
   #       self.h_prev = np.zeros(self.hidden_size)
   #       lstm_out = self.__forward(x_train)
   #       self.loss = self.__backward(y_train)
   #       self.lstm_parameters.update_parameters()

   #       ## LOGGER FUNCTIONALITY
   #       clear = lambda: os.system('cls')
   #       clear()
   #       print(f"Epoch: {epoch} ")
   #       print("loss:", "%.3e" % self.loss)
   #       print("lr:", "%.3e" % self.learning_rate)
   #       ## 

   #       self.learning_rate = eta0 / (1 + self.lr_decrease_speed * epoch)
   #       epoch += 1

   def fit(self, x_train, y_train, epochs, precision, batch_size=64):
      n_samples = len(x_train)
      eta0 = self.learning_rate

      epoch = 0
      while epoch < epochs and self.loss > precision:
         indices = np.arange(n_samples)
         np.random.shuffle(indices)
         x_train, y_train = x_train[indices], y_train[indices]

         self.loss = 0
         self.c_prev = np.zeros(self.hidden_size)
         self.h_prev = np.zeros(self.hidden_size)

         for start in range(0, n_samples, batch_size):
               end = min(start + batch_size, n_samples)
               batch_x = x_train[start:end]
               batch_y = y_train[start:end]

               self.nodes = [LSTMGate(self.lstm_parameters, self.hidden_size, self.features_number) for _ in range(len(batch_x))]

               lstm_out = self.__forward(batch_x)
               batch_loss = self.__backward(batch_y)
               self.lstm_parameters.average_parameters(len(batch_x))
               self.lstm_parameters.update_parameters()

               self.loss += batch_loss * len(batch_x)  

         self.loss /= n_samples

         ## LOGGER
         clear = lambda: os.system('cls')
         clear()
         print(f"Epoch: {epoch}")
         print("loss:", "%.3e" % self.loss)
         print("lr:", "%.3e" % self.learning_rate)
         ##

         self.learning_rate = eta0 / (1 + self.lr_decrease_speed * epoch)
         epoch += 1


   def compute(self, sequence, reset_params = False):
      if (reset_params == True):
         self.c_prev = np.zeros(self.hidden_size)
         self.h_prev = np.zeros(self.hidden_size)
      self.nodes = [LSTMGate(self.lstm_parameters, self.hidden_size, self.features_number) for _ in range(len(sequence))]
      lstm_out = self.__forward(sequence)
      return lstm_out
   
   def get_parameters(self):
      return self.lstm_parameters
   
   def get_loss(self):
      return self.loss