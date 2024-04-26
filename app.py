import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from flask import Flask, send_file
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)

@app.route('/plot')
def plot_chart():

    df=pd.read_csv(r'D:\Project\HistoricalData_DAX.csv')
    df.head()

    new_df=df.reset_index()['Close/Last']

    scaler=MinMaxScaler()
    scaled_data=scaler.fit_transform(np.array(new_df).reshape(-1,1))

    train_size=int(len(scaled_data)*0.8)
    train_data,test_data=scaled_data[:train_size],scaled_data[train_size:]

    n_past=120

    X_train,y_train=[],[]
    for i in range(n_past,len(train_data)):
        X_train.append(train_data[i-n_past:i,0])
        y_train.append(train_data[i,0])

    X_train,y_train=np.array(X_train),np.array(y_train)

    X_test,y_test=[],[]
    for i in range(n_past,len(test_data)):
        X_test.append(test_data[i-n_past:i,0])
        y_test.append(train_data[i,0])

    X_test,y_test=np.array(X_test),np.array(y_test)

    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)

    plt.figure(figsize = (18,9))
    plt.plot(range(df.shape[0]),(df['Close/Last']))
    plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price',fontsize=18)

    close_prices = df['Close/Last'].values

    train_data = close_prices[:11000]
    test_data = close_prices[11000:]

    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1,1)
    test_data = test_data.reshape(-1,1)

    smoothing_window_size = 2500
    for di in range(0, 10000, smoothing_window_size):
        if di + smoothing_window_size <= len(train_data):
            scaler.fit(train_data[di:di+smoothing_window_size,:])
            train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

    # Normalize the last bit of remaining data if any
    if di + smoothing_window_size < len(train_data):
        scaler.fit(train_data[di+smoothing_window_size:,:])
        train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

    EMA = 0.0
    gamma = 0.1
    for ti in range(len(train_data)):
        EMA = gamma * train_data[ti] + (1 - gamma) * EMA
        train_data[ti] = EMA

    # Concatenate train_data and test_data for visualization and testing
    all_close_data = np.concatenate([train_data, test_data], axis=0)

    window_size = 100
    N = train_data.size
    std_avg_predictions = []
    std_avg_x = []
    mse_errors = []

    for pred_idx in range(window_size,N):

        if pred_idx >= N:
        # Use the date from the previous prediction or any other appropriate method to generate the next date
            previous_date = df.loc[pred_idx - 1, 'Date']
            date = dt.datetime.strptime(previous_date, '%Y-%m-%d').date() + dt.timedelta(days=1)
        else:
            date = df.loc[pred_idx, 'Date']

        std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
        mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
        std_avg_x.append(date)

    window_size = 100
    N = train_data.size

    run_avg_predictions = []
    mse_errors = []

    running_mean = 0.0
    run_avg_predictions.append(running_mean)

    decay = 0.5

    for pred_idx in range(1, N):
        running_mean = running_mean * decay + (1.0 - decay) * train_data[pred_idx - 1]
        prediction = running_mean.item()  # Convert running_mean to a scalar value
        run_avg_predictions.append(prediction)
        mse_errors.append((prediction - train_data[pred_idx]) ** 2)

    # Convert run_avg_predictions to a NumPy array
    np_run_avg_predictions = np.array(run_avg_predictions)

    # Flatten the arrays
    true_close_prices = all_close_data.flatten()
    predicted_close_prices = np_run_avg_predictions[:len(all_close_data)].flatten()

    # Create a DataFrame with true and predicted mid prices
    data = {
        'Date': df['Date'],
        'True Close Price': true_close_prices,
        'Predicted Close Price': predicted_close_prices
    }

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')

    checkpoints=ModelCheckpoint(filepath='my_weights.h5',save_best_only=True)
    early_stopping=EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True)

    model.fit(X_train,y_train,
            validation_data=(X_test,y_test),
            epochs=100,
            batch_size=32,
            verbose=1,
            callbacks=[checkpoints,early_stopping])

    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)

    # Get the original shape of the input data
    n_samples, n_timesteps, n_features = X_train.shape

    # Reshape train_predict and test_predict to match the original shape of the input data
    train_predict_reshaped = train_predict.reshape(n_samples, -1, 1)
    test_predict_reshaped = test_predict.reshape(X_test.shape[0], -1, 1)

    # Reshape train_predict and test_predict to match the expected shape for inverse transformation
    train_predict_reshaped = train_predict_reshaped.reshape(-1, 1)
    test_predict_reshaped = test_predict_reshaped.reshape(-1, 1)

    scaler.fit(train_predict_reshaped)

    look_back=120

    trainPredictPlot=np.empty_like(new_df)
    trainPredictPlot[:]=np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back]=train_predict.flatten()

    testPredictPlot=np.empty_like(new_df)
    testPredictPlot[:]=np.nan
    test_start=len(new_df)-len(test_predict)
    testPredictPlot[test_start:]=test_predict.flatten()

    original_scaled_data=scaler.inverse_transform(scaled_data)

    last_sequence=X_test[-1]

    last_sequence=last_sequence.reshape(1,n_past,1)

    future_predictions=[]
    for _ in range(365):
        next_day_prediction=model.predict(last_sequence)
        future_predictions.append(next_day_prediction[0,0])
        last_sequence=np.roll(last_sequence,-1,axis=1)
        last_sequence[0,-1,0]=next_day_prediction

    future_predictions=scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))

    future_predictions_percent = future_predictions.flatten() * 100
    
    # Plot the data
    plt.figure(figsize=(15, 6))
    plt.plot(future_predictions, marker='*', color='green', label='Predicted Price')
    plt.title('Predicted Stock Price for Next 365 Days')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Close the plot to free up resources
    plt.close()

    # Return the plot as the response
    return send_file(buffer, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)