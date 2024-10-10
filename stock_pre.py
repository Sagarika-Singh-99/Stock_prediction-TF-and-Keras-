!pip install tensorflow

#Import relevant libs
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split

#Read data
import pandas as pd
stock_ds = pd.read_csv("/content/uber_stock.csv")
print(stock_ds.head())
print(stock_ds.tail())

#Data details:
print(stock_ds.info())
print(stock_ds.describe())
print(stock_ds.isnull().sum())

#Feature graph:

from matplotlib import pyplot as plt

plt.figure()

plt.plot(stock_ds["Open"])
plt.plot(stock_ds["High"])
plt.plot(stock_ds["Low"])
plt.plot(stock_ds["Close"])

plt.title('Uber stock price history')

plt.ylabel('Price (USD)')
plt.xlabel('Days')

plt.legend(['Open','High','Low','Close'], loc='upper left')
plt.show()

#Adj close graph:

from matplotlib import pyplot as plt

plt.figure()

plt.plot(stock_ds["Adj Close"])

plt.title('Uber stock price history - Adj Close Value')

plt.ylabel('Price (USD)')
plt.xlabel('Days')

plt.legend(['Adj Close'], loc='upper left')
plt.show()

# Features and target feature we will use:

features = stock_ds[['Open', 'High', 'Low']]
target = stock_ds['Adj Close']


print(features)
print(target)

#Scaling of features:

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

scaler = MinMaxScaler()

sf = scaler.fit_transform(features)
sf = pd.DataFrame(sf, columns=features.columns, index=stock_ds.index)
print(sf.head())

#Spliting of data into training data and testing data:

from sklearn.model_selection import train_test_split

features_train, features_test = train_test_split(sf, train_size=0.8, test_size=0.2, shuffle=True)
target_train, target_test = train_test_split(target, train_size=0.8, test_size=0.2, shuffle=True)

print("Features - Train and Test size:", len(features_train), len(features_test))
print("Target - Train and Test size:", len(target_train), len(target_test))

#Conversion of data:

import numpy as np

features_train_3d = np.array(features_train).reshape(features_train.shape[0], 1, features_train.shape[1])
features_test_3d = np.array(features_test).reshape(features_test.shape[0], 1, features_test.shape[1])

target_train = np.array(target_train)
target_test = np.array(target_test)

#Model:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Define the LSTM model
def l_model(input_shape):
    model = Sequential([
        LSTM(50, input_shape=input_shape, return_sequences=True, activation='tanh'),
        Dropout(0.2),
        LSTM(50, return_sequences=False, activation='tanh'),
        Dropout(0.2),
        Dense(20, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Create the model with the shape of the input data
model = l_model((1, features_train_3d.shape[2]))

# Setup early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

model.save('model.pb')

from google.colab import drive
import shutil

# Mount Google Drive
drive.mount('/content/drive')

# Path to your .pb file in Colab environment
pb_file_path = '/saved_model.pb'

# Path to where you want to copy the file in Google Drive
destination_path = 'x'

# Copy the file to Google Drive
shutil.copy(pb_file_path, destination_path)

#Model arch:

from tensorflow.keras.utils import plot_model
from IPython.display import SVG

plot_model(model, show_shapes = True)

model.summary()

#Model training:

history = model.fit(features_train_3d, target_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

# Plot training & validation loss values:

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

#Metrics:

score = model.evaluate(features_train_3d, target_train, verbose=1)
print('Test score:', score)
print('Test accuracy:', score)

#Metrics:

import numpy as np


mse_1 = np.mean((y_pred - target_test[:, np.newaxis])**2, axis=1)
rmse_1 = np.sqrt(mse_1)

# Calculate overall MSE and RMSE
mse_f = np.mean(mse_1)
rmse_f = np.mean(rmse_1)

print("Overall MSE:", mse_f)
print("Overall RMSE:", rmse_f)

#Predictins vs Actual Values:

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Plot true values vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(target_test, label='True Values')
plt.plot(y_pred, label='Predicted Values')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('True vs Predicted Stock Prices')
plt.legend()
plt.show()

#Predicting for 30 days and 1 year:

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def l_model(input_shape):
    model = Sequential([
        LSTM(50, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(20, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

model = l_model((10, 3))



#Function to predict:

def predict_future_stock(model, initial_data, n_days):
    seq = initial_data[-10:].reshape(1, 10, 3)
    predictions = []

    for _ in range(n_days):
        predicted_value = model.predict(seq)[0][0]
        predictions.append(predicted_value)

        seq = np.roll(seq, -1, axis=1)
        seq[0, -1, :] = predicted_value

    return predictions


recent_data = features_test_3d
future_30_days = predict_future_stock(model, recent_data, 30)
future_365_days = predict_future_stock(model, recent_data, 365)

print("30-day Future Predictions:", future_30_days)
print("365-day Future Predictions:", future_365_days)

import matplotlib.pyplot as plt

# Plotting predictions for 30 days
plt.figure(figsize=(10, 5))
plt.plot(future_30_days, label='Next 30 Days Prediction')
plt.title('Stock Price Prediction for the Next 30 Days')
plt.xlabel('Days')
plt.ylabel('Predicted Stock Price')
plt.legend()
plt.show()

# Plotting predictions for a year
plt.figure(figsize=(10, 5))
plt.plot(future_365_days, label='Next Year Prediction')
plt.title('Stock Price Prediction for the Next Year')
plt.xlabel('Days')
plt.ylabel('Predicted Stock Price')
plt.legend()
plt.show()





