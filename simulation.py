import numpy as np
import pydot
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.models import Model
from keras.utils.vis_utils import plot_model
import pytorch_tabnet
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

def identify_representations(num_samples=1000, num_features=6, p=.9,
                             layer_widths=[2], batch_size=10, epochs_per_run=10):
  start = time()

  # Set random seeds for replicability
  np.random.seed(1)
  tf.random.set_seed(1)

  # Create random features
  X = np.random.randn(num_samples, num_features)
  q = 1 - p

  # Create a variable A that is well-predicted by two of the random variables
  A = np.array([])
  for x in X:
    xor = -1 if (np.sign(x[0]) == np.sign(x[1])) else 1
    A = np.append(A, np.random.choice([xor*abs(np.random.randn()), -xor*abs(np.random.randn())], p=[p, q]))

  # Create a variable that is well-predicted by two other random variables
  B = np.array([])
  for x in X:
    xor = -1 if (np.sign(x[2]) == np.sign(x[3])) else 1
    B = np.append(B, np.random.choice([xor*abs(np.random.randn()), -xor*abs(np.random.randn())], p=[p, q]))

  # Create a variable Y that is well-predicted by A
  Y = np.array([])
  for a in A:
    Y = np.append(Y, np.random.choice([a, -a], p=[p, q]))
  # Alternatively, well-predicted by A and B
  # for i in range(len(A)):
  #   Y = np.append(Y, np.random.choice([A[i] + B[i], -(A[i] + B[i])], p=[p, q]))

  # Prepare training test split
  X_train, X_test, Y_train, Y_test, A_train, A_test, B_train, B_test = train_test_split(
      np.array(X), np.array(Y), np.array(A), np.array(B), test_size=0.2, random_state=1)

  # Fit linear models for comparison
  lm1 = sm.OLS(Y_train, X_train).fit()
  lm2 = sm.OLS(A_train, X_train).fit()
  lm3 = sm.OLS(B_train, X_train).fit()

  # Make predictions for linear models
  predictions1r = lm1.predict(X_test)
  i = 0
  for idx, j in enumerate(predictions1r):
      if (np.sign(j) == np.sign(Y_test[idx])):
          i += 1
  print(f'LM accuracy on Y: {i / len(X_test)}')

  predictions2r = lm2.predict(X_test)
  i = 0
  for idx, j in enumerate(predictions2r):
      if (np.sign(j) == np.sign(A_test[idx])):
          i += 1
  print(f'LM accuracy on A: {i / len(X_test)}')

  predictions3r = lm3.predict(X_test)
  i = 0
  for idx, j in enumerate(predictions3r):
      if (np.sign(j) == np.sign(B_test[idx])):
          i += 1
  print(f'LM accuracy for B: {i / len(X_test)}')


  # Define model layers, varying extent of representations
  # To-do: Allow for > 2 layers
  ip_layer = Input(shape=(X.shape[1]))
  if len(layer_widths) >= 2:
      if len(layer_widths) > 2:
        print('Too many layers. Using first 2.')
      dl1 = Dense(layer_widths[0], activation='relu')(ip_layer)
      dl2 = Dense(layer_widths[1], activation='relu')(ip_layer)
      output = Dense(1)(dl2)
  else:
      dl1 = Dense(layer_widths[0], activation='relu')(ip_layer)
      output = Dense(1)(dl1)

  # Define model and compile
  global model
  model = Model(inputs=ip_layer, outputs=output)
  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer="adam")

  # Train model 1
  history1 = model.fit(X_train, Y_train,
                      validation_data=(X_test,Y_test),
                      batch_size=batch_size, epochs=epochs_per_run)

  # Plot performance 1
  fig1 = plt.figure()
  plt.plot(history1.history['loss'])
  plt.plot(history1.history['val_loss'])
  plt.title('Y: loss/epoch')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.ylim([0.8, 1.1])
  plt.legend(['train_loss','val_loss'], loc='upper left')

  # Make predictions 1
  predictions1n = model.predict(X_test).T[0]
  i = 0
  for idx, j in enumerate(predictions1n):
      if (np.sign(j) == np.sign(Y_test[idx])):
          i += 1
  print(f'NN accuracy on Y: {i / len(X_test)}')

  # Freeze all except final layer for transfer learning
  for i in range(len(model.layers)-1):
    model.layers[i].trainable = False

  # Train model 2
  history2 = model.fit(X_train, A_train,
                      validation_data=(X_test,A_test),
                      batch_size=batch_size, epochs=epochs_per_run)

  # Plot performance 2
  fig2 = plt.figure()
  plt.plot(history2.history['loss'])
  plt.plot(history2.history['val_loss'])
  plt.title('A: loss/epoch')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.ylim([0.8, 1.1])
  plt.legend(['train_loss','val_loss'], loc='upper left')

  # Make predictions 2
  predictions2n = model.predict(X_test).T[0]
  i = 0
  for idx, j in enumerate(predictions2n):
      if (np.sign(j) == np.sign(A_test[idx])):
          i += 1
  print(f'NN accuracy on A: {i / len(X_test)}')

  # Train model 3
  history3 = model.fit(X_train, B_train,
                       validation_data=(X_test,B_test),
                       batch_size=batch_size, epochs=epochs_per_run)

  # Plot performance 3
  fig3 = plt.figure()
  plt.plot(history3.history['loss'])
  plt.plot(history3.history['val_loss'])
  plt.title('B: loss/epoch')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.ylim([0.8, 1.1])
  plt.legend(['train_loss','val_loss'], loc='upper left')

  # Make predictions 3
  predictions3n = model.predict(X_test).T[0]
  i = 0
  for idx, j in enumerate(predictions3n):
      if (np.sign(j) == np.sign(B_test[idx])):
          i += 1
  print(f'NN accuracy on B: {i / len(X_test)}')

  print(f'Time elapsed: {(time() - start)/60} minutes')

# Run the simulation
identify_representations(num_samples=10000, num_features=10, p=.9,
                             layer_widths=[2], batch_size=10, epochs_per_run=50)
