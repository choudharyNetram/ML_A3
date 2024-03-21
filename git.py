import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the data
data = pd.read_csv("D:\Downloads\csvfile.txt", sep=r'\s+')
data['Date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
data.set_index('Date', inplace=True)
data.index.freq = 'MS'

# Preprocess data
data_diff = data['average'].diff().dropna()

def sort_and_split_data(data, K, T):
    # Sort data based on the month column
    sorted_data = data.sort_values(by=['year', 'month'])

    # Set the index to a DateTimeIndex
    sorted_data['Date'] = pd.to_datetime(sorted_data[['year', 'month']].assign(day=1))
    sorted_data.set_index('Date', inplace=True)

    # Create X and y datasets
    X_train = sorted_data[sorted_data.index.month <= K][['year', 'month']]
    X_test = sorted_data[(sorted_data.index.month > K) & (sorted_data.index.month <= K + T)][['year', 'month']]

    # Adjust indices to align with shifted y values
    y_train = sorted_data['average'].shift(-T).dropna().iloc[:len(X_train)]
    y_test = sorted_data['average'].shift(-T).dropna().iloc[len(X_train):len(X_train) + len(X_test)]

    return X_train, X_test, y_train.values, y_test.values

# Specify parameters
K = 3  # Number of previous readings
T = 1  # Number of readings to predict

# Split data into training and testing sets
X_train, X_test, y_train, y_test = sort_and_split_data(data, K, T)

# Check if there is enough data for testing
if len(X_test) == 0:
    print("Not enough data for testing. Adjust parameters K and T.")
else:
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # MLP model
    model = Sequential([
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1,activation='relu')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)

    # Make predictions
    y_pred_mlp = model.predict(X_test_scaled).flatten()

    # Align y_test with y_pred_mlp
    y_test_aligned = y_test[:len(y_pred_mlp)]

    # MA model
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    y_pred_ma = moving_average(data['average'], K)[-len(y_test):]

    # ARIMA model
    model_arma = ARIMA(data_diff, order=(K, 0, 0))
    model_arma_fit = model_arma.fit()  # Example values, adjust as needed
    y_pred_arma = model_arma_fit.forecast(steps=len(y_test))

    # Evaluate models
    mse_mlp = mean_squared_error(y_test_aligned, y_pred_mlp)
    mse_ma = mean_squared_error(y_test_aligned, y_pred_ma)
    mse_arma = mean_squared_error(y_test_aligned, y_pred_arma)

    print("MLP MSE:", mse_mlp)
    print("Moving Average MSE:", mse_ma)
    print("ARMA MSE:", mse_arma)
    #print(y_train)
    #print(y_test)
    #print(y_pred_arma)
    #print(y_pred_ma)
    #print(y_pred_mlp)