# Description: This program attempts to predict the future price of a stock
# Import libraries
import math

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import streamlit as st

st.title('Stock Price Prediction')

#  Collect and clean the data
df = pd.read_csv('Apple_stockprices.csv')
df = df.dropna()  # The dropna() method removes the rows that contains NULL values.

# Describing the data
st.subheader('Date from 1980 to 2022')
st.write(df.describe())

st.divider()

# Visualization
st.subheader('Closing Price vs.Time Chart')
# Show the data visually
fig = plt.figure(figsize=(20, 10))
plt.plot(df.Close)
st.pyplot(fig)
st.divider()

st.subheader('Closing Price vs.Time Chart with 100 Days MA')
# 100 days minimum average
ma100 = df.Close.rolling(100).mean()
fig1 = plt.figure(figsize=(20, 10))
plt.plot(df.Close)
plt.plot(ma100, 'r')
st.pyplot(fig1)

st.divider()

st.subheader('Closing Price vs.Time Chart with 100 Days and 200 Days MA')
# 200 days minimum average
ma200 = df.Close.rolling(200).mean()
fig2 = plt.figure(figsize=(20, 10))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
st.pyplot(fig2)

st.divider()


# Prediction of new values
st.subheader('Price Prediction(in $)')

X = df[['Open', 'High', 'Low', 'Volume']]
Y = df['Close']

# split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create the model
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, Y_train)

user_input1 = st.number_input('Enter the Open value (in $) of the stock: ')
user_input2 = st.number_input('Enter the High value (in $) of the stock: ')
user_input3 = st.number_input('Enter the Low value (in $) of the stock: ')
user_input4 = st.number_input('Enter the Volume value (in $) of the stock: ')

# Take the inputs from user
value = [float(user_input1), float(user_input2),
         float(user_input3), float(user_input1)]
value = pd.DataFrame([value], columns=X.columns)

# Make the predictions
prediction = model.predict(value)[0]

st.write('The Predicted Closing value (in $) of stack is: ', prediction)

st.divider()

# Calculating min errors
# split the data into training and testing sets
st.subheader('Errors')

Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
st.write("Mean Squared Error: ", mse)

rsme = math.sqrt(mse)
st.write("Root Mean Squared Error: ", rsme)

mae = mean_absolute_error(Y_test, Y_pred)
st.write("Mean Absolute Error: ", mae)

st.divider()

st.subheader('Accuracy:')
# Calculate accuracy score

r2 = r2_score(Y_test, Y_pred)
st.write("Accuracy: ", (r2 * 100))

st.divider()
