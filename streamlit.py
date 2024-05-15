import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.express as px
import plotly.graph_objs as go

# Load the data
# Replace this line with loading your actual data
data = pd.read_csv('/path/to/your/data.csv')

# Apply a matte background
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("Future is Yours")

# Asset selection
assets = ['Bitcoin', 'Ethereum', 'Litecoin', 'Ripple']
selected_assets = st.multiselect('Select assets', assets, default=assets)

# Display asset charts
st.header("Asset Charts")
fig, ax = plt.subplots()
for asset in selected_assets:
    ax.plot(data['Date'], data[asset], label=asset)
ax.legend()
st.pyplot(fig)

# Option to display all assets on the same chart with Bitcoin rescale
if st.checkbox('Display all assets on the same chart with Bitcoin rescale'):
    fig, ax = plt.subplots()
    for asset in selected_assets:
        if asset == 'Bitcoin':
            ax.plot(data['Date'], data[asset], label=asset)
        else:
            ax.plot(data['Date'], data[asset] * data['Bitcoin'].iloc[-1] / data[asset].iloc[-1], label=asset)
    ax.legend()
    st.pyplot(fig)

# Correlation matrix
st.header("Correlation Matrix")
corr_matrix = data[selected_assets].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
st.pyplot()

# Returns for each asset
st.header("Asset Returns")
returns = data[selected_assets].pct_change()
for asset in selected_assets:
    fig, ax = plt.subplots()
    ax.plot(data['Date'], returns[asset])
    ax.set_title(f'{asset} Returns')
    st.pyplot(fig)

# Correlation of returns
st.header("Correlation of Asset Returns")
corr_returns = returns.corr()
sns.heatmap(corr_returns, annot=True, cmap='coolwarm')
st.pyplot()

# Bollinger Bands for each asset
st.header("Bollinger Bands")
for asset in selected_assets:
    window = 20
    data[f'{asset}_SMA'] = data[asset].rolling(window).mean()
    data[f'{asset}_STD'] = data[asset].rolling(window).std()
    data[f'{asset}_Upper'] = data[f'{asset}_SMA'] + (data[f'{asset}_STD'] * 2)
    data[f'{asset}_Lower'] = data[f'{asset}_SMA'] - (data[f'{asset}_STD'] * 2)
    
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data[asset], label=asset)
    ax.plot(data['Date'], data[f'{asset}_SMA'], label='SMA')
    ax.plot(data['Date'], data[f'{asset}_Upper'], label='Upper Band')
    ax.plot(data['Date'], data[f'{asset}_Lower'], label='Lower Band')
    ax.legend()
    st.pyplot(fig)

# Prediction model selection
st.header("Prediction")
model_choice = st.selectbox('Choose a prediction model', ['Linear Regression', 'ARIMA', 'LSTM price', 'LSTM price with correlation', 'LSTM returns'])

# User input for investment amount and period
investment_amount = st.number_input('Investment amount', min_value=0)
investment_end_date = st.date_input('Investment end date', max_value=datetime.date(2025, 7, 1))

# Display prediction graphs and returns
st.header("Prediction Graphs and Returns")

def predict_with_linear_regression(data, asset, days):
    model = LinearRegression()
    X = np.arange(len(data)).reshape(-1, 1)
    y = data[asset].values
    model.fit(X, y)
    future_X = np.arange(len(data) + days).reshape(-1, 1)
    future_y = model.predict(future_X)
    return future_y

def predict_with_arima(data, asset, days):
    model = ARIMA(data[asset], order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    forecast = model_fit.forecast(steps=days)[0]
    return np.concatenate((data[asset].values, forecast))

# Example prediction for Bitcoin using Linear Regression
if model_choice == 'Linear Regression':
    days = (investment_end_date - data['Date'].max()).days
    future_prices = predict_with_linear_regression(data, 'Bitcoin', days)
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(data)), data['Bitcoin'], label='Historical')
    ax.plot(np.arange(len(future_prices)), future_prices, label='Prediction')
    ax.legend()
    st.pyplot(fig)

# Display the projected return (simplified example)
if model_choice == 'Linear Regression':
    future_prices = predict_with_linear_regression(data, 'Bitcoin', days)
    future_price_at_end_date = future_prices[-1]
    initial_price = data['Bitcoin'].iloc[-1]
    projected_return = (future_price_at_end_date - initial_price) / initial_price * 100
    st.write(f"The projected return is {projected_return:.2f}%.")

# Add the other prediction models as needed

