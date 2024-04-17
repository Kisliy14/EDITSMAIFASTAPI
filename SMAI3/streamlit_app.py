import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import tensorflow as tf
import requests
import json
import time
import requests
from fastapi import FastAPI,Form

st.title("Stock Price Visualization")

days = st.sidebar.slider("Select number of days", min_value=1, max_value=60, value=60)

st.sidebar.subheader("Select Stock(s)")
selected_stocks = st.sidebar.text_input("Enter stock symbols separated by commas (e.g., AAPL,MSFT)", value="AAPL")
stocks = [s.strip().upper() for s in selected_stocks.split(',')]
value_variable = selected_stocks

def plot_stock_price(ticker_symbol, days):
    data = yf.download(ticker_symbol, period=f"{days}d")
    plt.figure(figsize=(10, 6))
    plt.plot(data['Adj Close'])
    plt.title(f"{ticker_symbol} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    st.pyplot(plt)

#url = f"http://127.0.0.1:8000/train_model/?value_variable={value_variable}&days={days}"
url = 'http://127.0.0.1:8000/train_model/'
headers = {'Content-Type': 'application/json'}
data = {'value_variable': value_variable, 'days': days}
response = requests.post(url, headers=headers, data=data)
response_json = response.json()
st.write(response_json)
future_df = response.json()['future_df']
final_df = response.json()['final_df']

# Вывод структуры данных
st.write("Структура данных future_df:")
st.write(future_df)
st.write("Структура данных final_df:")
st.write(final_df)

# Вывод содержимого данных
st.write("Содержимое future_df:")
st.write(future_df['future_df'])
st.write("Содержимое final_df:")
st.write(final_df['final_df'])
# while True:  # Бесконечный цикл
#     response = requests.post(url, headers=headers, data=data)
#     response_json = response.json()
    
#     if 'final_df' in response_json and 'future_df' in response_json:
#         # Если 'final_df' и 'future_df' присутствуют в ответе, выходим из цикла
#         final_df = response.json()['final_df']
#         future_df = response.json()['future_df']

#         break
#     else:
#         print("Ожидаемые данные еще не получены, повторяем запрос...")
#         time.sleep(1)  # Пауза в 1 секунду между запросами

# final_df = response.json()['final_df']
# future_df = response.json()['future_df']

st.title("Stock Price Visualization")

plt.figure(figsize=(16, 6))
plt.title('Future Close Price Prediction')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price in USD ($)', fontsize=18)
plt.plot(final_df, label='Actual Data')
plt.axvline(x=df.index[-1], color='r', linestyle='--', label='End of Actual Data')
plt.plot(future_df['Future Close'], label='Future Predictions', linestyle='-')
plt.legend()
st.pyplot()

