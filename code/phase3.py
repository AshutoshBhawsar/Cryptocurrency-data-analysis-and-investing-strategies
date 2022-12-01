import streamlit as st
from datetime import datetime

from prophet.serialize import model_from_json
from prophet.plot import plot_plotly
from plotly import graph_objs as go

import numpy as np

import joblib

# from keras.models import load_model

import pandas as pd

# import matplotlib.pyplot as plt

btc_main = pd.read_csv('../data/bitcoin_daily_data_cleaned.csv')#, index_col='Timestamp')

st.set_page_config(page_title='Cryptocurrency Price Prediction App',
    layout='wide')

st.title("Cryptocurrency Price Prediction")
st.subheader("Dataset sample:")
st.write(btc_main.tail())



# st.subheader("1. LSTM")

st.subheader("1. Random Forest Classifier")
# random_forest_classifier = joblib.load("../models/random_forest_classifier.joblib")
random_forest_predictions = pd.read_csv('../data/random_forest_predictions.csv')

st.markdown("**Please select date for getting trade call (Buy / Sell):**")
input_date_rf = st.date_input('Input Date', value = datetime(2021,3,31).date())


if input_date_rf is None:
    st.info('Please select date to predict')
else:
    if input_date_rf < datetime(2019,11,14).date() or input_date_rf > datetime(2021,3,31).date():
        st.info("Date exceeds prediction limit")
    else:
        random_forest_predictions.Timestamp = pd.to_datetime(random_forest_predictions.Timestamp).dt.date
        predicted_calls = random_forest_predictions[random_forest_predictions.Timestamp == input_date_rf]


        # output
        string_date = input_date_rf.strftime("%m/%d/%Y")
        st.text(f"Trade signal for {string_date} is:")
        trade_call = predicted_calls.Pred.values[0]
        if trade_call == 1:
            st.info("BUY")
        else:
            st.info("SELL")
        
        # st.markdown(f"*Your portfolio of 1 BTC can have maximum equity of \${predicted_vals.yhat_upper.values[0].round(2)} and minimum equity of \${predicted_vals.yhat_lower.values[0].round(2)} on {string_date}*")


st.line_chart(random_forest_predictions, x="Timestamp", y="Pred")





st.subheader("2. FB Prophet")

with open('../models/prophet_model.json', 'r') as fin:
    prophet_model = model_from_json(fin.read())

forecasting = pd.read_csv('../data/forecasting.csv')
st.write(forecasting.tail())

st.markdown("**Please select date for BTC prediction:**")
# day = st.slider('Select day', 1, 30, 15, 1)
# month = st.slider('Select month', 1, 12, 6, 1)
# year = st.slider('Select year', 2012, 2021, 2020, 1)
input_date = st.date_input('Date Input', value = datetime(2021,3,31).date())


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
        string_date = input_date.strftime("%m/%d/%Y")
        st.text("Predicted weighted price:")
        st.info(predicted_vals.yhat.values[0])
        st.text("Predicted high value:")
        st.info(predicted_vals.yhat_upper.values[0])
        st.text("Predicted low value:")
        st.info(predicted_vals.yhat_lower.values[0])
        st.markdown(f"*Your portfolio of 1 BTC can have maximum equity of \${predicted_vals.yhat_upper.values[0].round(2)} and minimum equity of \${predicted_vals.yhat_lower.values[0].round(2)} on {string_date}*")



fig2 = plot_plotly(prophet_model, forecasting)
st.plotly_chart(fig2)

# plt.title('Price Forecasting for Bitcoin')
# plt.ylabel('Price (USD)')
# plt.xlabel('Date')








