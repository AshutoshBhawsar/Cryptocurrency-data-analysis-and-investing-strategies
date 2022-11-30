import streamlit as st
from datetime import datetime

from prophet.serialize import model_from_json
from prophet.plot import plot_plotly
from plotly import graph_objs as go

import pandas as pd

# import matplotlib.pyplot as plt

btc_main = pd.read_csv('../data/bitcoin_daily_data_cleaned.csv')#, index_col='Timestamp')

st.set_page_config(page_title='Cryptocurrency Price Prediction App',
    layout='wide')

st.title("Cryptocurrency Price Prediction")
st.subheader("Dataset")
st.write(btc_main.tail())

st.subheader("1. FB Prophet")

with open('../models/prophet_model.json', 'r') as fin:
    prophet_model = model_from_json(fin.read())

forecasting = pd.read_csv('../data/forecasting.csv')
st.write(forecasting.tail())

st.markdown("**Please select date for BTC prediction:**")
# day = st.slider('Select day', 1, 30, 15, 1)
# month = st.slider('Select month', 1, 12, 6, 1)
# year = st.slider('Select year', 2012, 2021, 2020, 1)
input_date = st.date_input('Input Date', value = datetime(2021,3,31).date())


if input_date is None:
    st.info('Please select date to predict')
else:
    if input_date > datetime(2021,3,31).date():
        st.info("Date exceeds prediction limit")
    else:
        forecasting.ds = pd.to_datetime(forecasting.ds).dt.date
        # st.text(type(input_date))
        # st.text(type(forecasting.ds.iloc[1]))
        predicted_vals = forecasting[forecasting.ds == input_date]
        # st.text(type(predicted_vals))
        # st.text(pd.to_datetime(btc_main.Timestamp.iloc[1]).date())


        # output
        st.text("Predicted high value:")
        st.info(predicted_vals.yhat_upper.values[0])
        st.text("Predicted low value:")
        st.info(predicted_vals.yhat_lower.values[0])
        st.text("Predicted value:")
        st.info(predicted_vals.yhat.values[0])



fig1 = plot_plotly(prophet_model, forecasting)
st.plotly_chart(fig1)

# plt.title('Price Forecasting for Bitcoin')
# plt.ylabel('Price (USD)')
# plt.xlabel('Date')