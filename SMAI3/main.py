from fastapi import FastAPI
from typing import List, Dict
from datetime import datetime
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from pandas_datareader import data as pdr
from datetime import datetime
import yfinance as yf
from fastapi import FastAPI,Form
import uvicorn
import pickle

yf.pdr_override()

app = FastAPI()

class Item(BaseModel):
    value_variable: str
    days: int

class ItemStorage:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0,1))
    @app.post("/train_model/")
    async def train_model(value_variable: str = Form(), days: int = Form()):
        from pandas_datareader import data as pdr
        from datetime import datetime
        yf.pdr_override()
        tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
        tech_list.extend(value_variable)
        end = datetime.now()
        start = datetime(end.year - 1, end.month, end.day)
        for stock in tech_list:
            globals()[stock] = yf.download(stock, start, end)        
        df = pdr.get_data_yahoo(value_variable, start='2012-01-01', end=datetime.now())
        data = df.filter(['Close'])
        dataset = data.values
        training_data_len = int(np.ceil( len(dataset) * .95 ))

        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)
        train_data = scaled_data[0:int(training_data_len), :]
        train_data_df = pd.DataFrame(train_data) 

        x_train = []
        y_train = []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        from keras.models import Sequential
        from keras.layers import Dense, LSTM
        
        model = pickle.load(open('C:/Users/Kisliy/Desktop/SMAI3/models/model.pkl','rb'))

        # self.model = Sequential()
        # self.model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        # self.model.add(LSTM(units=50, return_sequences=False))
        # self.model.add(Dense(units=25))
        # self.model.add(Dense(units=1))
        # self.model.compile(optimizer='adam', loss='mean_squared_error')
        # self.model.fit(x_train, y_train, batch_size=1, epochs=2)

        test_data = scaled_data[training_data_len - 60: , :]
        future_days = 60
        x_future = []
        x_test = []
        y_test = dataset[training_data_len:, :]

        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
            x_future.append(test_data[i-60:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
        predictions = self.model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions, squared=False)

        train = data[:training_data_len].copy()
        valid = data[training_data_len:].copy()
        valid.loc[:, 'Прогнозы'] = predictions
        x_future = np.array(x_future)
        x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))

        future_predictions = self.model.predict(x_future)
        future_predictions = scaler.inverse_transform(future_predictions)

        future_dates = pd.date_range(df.index[-1], periods=future_days+1, freq='B')[1:]
        future_df = pd.DataFrame(index=future_dates, columns=['Future Close'])
        future_df.index.name = 'Date'                
        future_df['Future Close'] = future_predictions[:60].flatten()
        selected_days = days
        future_df = pd.DataFrame(index=future_dates[:selected_days], columns=['Future Close'])
        future_df.index.name = 'Date'
        future_df['Future Close'] = future_predictions[:selected_days].flatten()
        final_df = pd.concat([df['Close'], future_df['Future Close']])
        df_json = df.to_json(orient="index", date_format='iso')
        future_df_json = future_df.to_json(orient="index", date_format='iso')
        final_df_json = final_df.to_json(orient="index", date_format='iso')

        return {"df": df_json, "final_df": final_df_json, "future_df": future_df_json}


storage = ItemStorage()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
