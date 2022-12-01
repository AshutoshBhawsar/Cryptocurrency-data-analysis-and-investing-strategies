# Cryptocurrency-data-analysis-and-investing-strategies
DIC Project Fall 2022, State University of New York at Buffalo

Data Source : https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data

# Phase 1 Steps:
1. Extract the Bitcoin dataset from above given link.
2. Setup python in your machine using following steps- https://realpython.com/installing-python/
3. Open the code in any Python IDE. 
4. Under the code folder in the code file for phase 1- phase1.ipynb, first change the path for bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv and give the correcsponding path of where the file is located in your machine.
5. Now run the file phase1.ipynb 
6. Each section in the code contains steps for cleaning the data, for which the output is displayed. 
7. The next part of the phase contains steps for preprocessing the data, which are needed for buidind and training our model in future.
8. The clean and pre processed data is finally generated and output is stored in the file bitcoin_daily_data_cleaned.csv under the data foler.



# Phase 2 Steps-
1. Under the code foler, open the file - phase2.ipynb, first change the path for bitcoin_daily_data_cleaned.csv and give the correcsponding path of where the file is located in your machine.
2. Now run the file phase2.ipynb.
3. Each section of code contains models which are used to train the data set for the purpose of either classification or regression. Then the test data is run on the trained model and prediction is done.
4. In the output various visualisation graphs for the models can be viewed.


# Phase 3 steps-
1. For the phase3 install and setup streamlit - https://docs.streamlit.io/library/get-started/installation
2. Under the code folder run the file phase3.py
3. In the webpage that is displayed, for Random Forest model the user can select any date. The model will run and output the user a decision of whether the user can buy or sell the stock on the selected date. User can also view the graph for that date.
4. The next part is the Prophet model for the which user can select and date and in the output he can view the predicted stock price, Upper most bound of the predicted stock price and lower most bound. 
