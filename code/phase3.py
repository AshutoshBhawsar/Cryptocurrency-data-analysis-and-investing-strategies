
# Importing required libraries
import streamlit as st
from datetime import datetime

from prophet.serialize import model_from_json
from prophet.plot import plot_plotly
from plotly import graph_objs as go

import numpy as np

import joblib

import pandas as pd


# Setting Page title for web app
st.set_page_config(page_title='Cryptocurrency Price Prediction App', layout='wide')

# Setting app heading as project name
st.title("Cryptocurrency Data Analysis and Trading Strategy")

# Reading cleaned data and printing a sample
btc_main = pd.read_csv('../data/bitcoin_daily_data_cleaned.csv')#, index_col='Timestamp')
btc_main.Timestamp = pd.to_datetime(btc_main.Timestamp).dt.date
st.subheader("Dataset sample:")
st.write(btc_main.tail())


# Section 1 - Random Forest Classification for giving out trading calls
st.subheader("1. Random Forest Classifier")
# random_forest_classifier = joblib.load("../models/random_forest_classifier.joblib")
random_forest_predictions = pd.read_csv('../data/random_forest_predictions.csv')

# Getting input date for trade call
st.markdown("**Please select date for getting trade call (Buy / Sell):**")
input_date_rf = st.date_input('Input Date', value = datetime(2021,3,31).date())

# Some error handling
if input_date_rf is None:
    st.info('Please select date to predict')
else:
    if input_date_rf < datetime(2019,11,14).date() or input_date_rf > datetime(2021,3,31).date():
        st.info("Date exceeds prediction limit")
    else:

        # Getting predictions from model output
        random_forest_predictions.Timestamp = pd.to_datetime(random_forest_predictions.Timestamp).dt.date
        predicted_calls = random_forest_predictions[random_forest_predictions.Timestamp == input_date_rf]


        # output printing
        string_date = input_date_rf.strftime("%Y/%m/%d")
        st.text(f"Trade signal for {string_date} is:")
        trade_call = predicted_calls.Pred.values[0]
        if trade_call == 1:
            st.info("BUY")
        else:
            st.info("SELL")

# Printing out interactive graph of RF classifier output        
st.line_chart(random_forest_predictions, x="Timestamp", y="Pred")



##################################  SECTION BREAK  ##################################################



# Section 2 - FaceBook Prophet model
st.subheader("2. FB Prophet")

# Retrieving model
with open('../models/prophet_model.json', 'r') as fin:
    prophet_model = model_from_json(fin.read())

# Pre loading the model output
forecasting = pd.read_csv('../data/forecasting.csv')
# st.write(forecasting)

# Getting 3 inputs from user: investment date, liquidation date, and size of investment
st.markdown("**Please select date for BTC prediction:**")
start_date = st.date_input('Investment date', value = datetime(2021,3,31).date())
end_date = st.date_input('Liquidation date', value = datetime(2021,3,31).date())
number_of_coins = st.number_input('Insert number of bitcoins to invest', min_value = 0.00, value = 0.00)

# More Error Handling
if number_of_coins == 0.00:
    st.info("Please input all fields to proceed")
else:
    if start_date > datetime(2021,3,31).date() or end_date > datetime(2021,3,31).date():
        st.info("Date exceeds prediction limit")
    elif start_date > end_date:
        st.info("Portfolio liquidation date cannot be before investment date, please try again")
    else:
        # Retrieving portfolio value for investment start date
        usd_price_start = btc_main[btc_main.Timestamp == start_date].Close.values[0]
        # usd_price_end = btc_main[btc_main.Timestamp == end_date].Close.values[0]
        portfolio_value = usd_price_start * number_of_coins
        
        # Printing out initial portfolio value
        st.markdown(f"**Your initial portfolio value on {start_date} is \${(portfolio_value).round(2)}**")

        # Getting model output for the given liquidation date
        forecasting.ds = pd.to_datetime(forecasting.ds).dt.date
        predicted_vals = forecasting[forecasting.ds == end_date]


        # Calculation and output for high, low, and predicted end portfolio value on liquidation
        string_date = end_date.strftime("%Y/%m/%d")
        st.text("Predicted weighted price:")
        st.info(predicted_vals.yhat.values[0] * number_of_coins)
        st.text("Predicted high value:")
        st.info(predicted_vals.yhat_upper.values[0] * number_of_coins)
        st.text("Predicted low value:")
        st.info(predicted_vals.yhat_lower.values[0] * number_of_coins)

        # Output message from model to determine trading strategy
        st.markdown(f"**Your portfolio of {number_of_coins} BTC can have maximum equity of \${(predicted_vals.yhat_upper.values[0] * number_of_coins).round(2)} and minimum equity of \${(predicted_vals.yhat_lower.values[0] * number_of_coins).round(2)} on {string_date}**")

        # Printing interactive output graph of prophet model for better insights on time series
        fig2 = plot_plotly(prophet_model, forecasting)
        st.plotly_chart(fig2)

#######################################  END  #######################################################
