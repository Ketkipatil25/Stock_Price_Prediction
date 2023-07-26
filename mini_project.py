# -*- coding: utf-8 -*-
"""Mini Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VhW9eHc-BXGiy3Een_FeyLBgNVeiQ6qq
"""

# Description: This program attempts to predict the future price of a stock
# Import libraries
import pandas as pd
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#  Collect and clean the data
df = pd.read_csv('Apple_stockprices.csv')
df = df.dropna()                                   # The dropna() method removes the rows that contains NULL values.

# Look at the data
df

# Show the data visually
ax = df.plot(x='Date', y='Close', figsize=(20, 10))

# add labels and title
ax.set_xlabel("Timeline(in year)")
ax.set_ylabel("Closing Price(in $)")
ax.set_title("Stock prices")

# show the plot
plt.show()

# Moving average of 100 days
# A moving average is simply an arithmetic mean of a certain number of data points.
ma100 = df.Close.rolling(100).mean()
ma100

plt.figure(figsize= (20,10))
plt.plot(df.Close)
plt.plot(ma100 , 'r')

ma200 = df.Close.rolling(200).mean()
ma200

plt.figure(figsize= (20,10))
plt.plot(df.Close)
plt.plot(ma100 , 'r')
plt.plot(ma200 , 'g')

p=1;

while p==1 :
  print("\n******Menu******")
  print("1.Predict Value of Stock \n2.Graph of Stock Prices\n")
  ch1 = int(input("Enter your option:\n"))

  if ch1 == 1:
    # select features and target variable
    X = df[['Open', 'High', 'Low', 'Volume']]
    Y = df['Close']

    # split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Create the model
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, Y_train)

    # Take the inputs from user
    value = []
    value.append(float(input('\nEnter the open value(in $) : ')))
    value.append(float(input('Enter the high value(in $) : ')))
    value.append(float(input('Enter the low value(in $) : ')))
    value.append(float(input('Enter the volume value(in $) : ')))
    value = pd.DataFrame([value], columns=X.columns)

    # Make the predictions
    prediction = model.predict(value)[0]

    print(f"The predicted stock value is: {prediction}")

  elif ch1==2:
    n=1

    while n==1 :
      print("\n******Menu******")
      print("1.1980-1990 \n2.1990-2000 \n3.2000-2010 \n4.2010-2020 \n5.2020-till date\n")
      ch = int(input("Enter the year option to view respective stock prices graph:\n"))

      if ch == 1:
        g1 = df.plot(x='Date', y='Close', figsize=(10, 6))

        # add labels and title
        g1.set_xlabel("Date")
        g1.set_ylabel("Close")
        g1.set_title("Stock prices")

        plt.xlim([0,2289])
        # show the plot
        plt.show()

      elif ch==2:
        g2 = df.plot(x='Date', y='Close', figsize=(10, 6))

        # add labels and title
        g2.set_xlabel("Date")
        g2.set_ylabel("Close")
        g2.set_title("Stock prices")

        plt.xlim([2290,4817])
        # show the plot
        plt.show()

      elif ch==3:
        g3 = df.plot(x='Date', y='Close', figsize=(10, 6))

        # add labels and title
        g3.set_xlabel("Date")
        g3.set_ylabel("Close")
        g3.set_title("Stock prices")

        plt.xlim([4818,7332])
        # show the plot
        plt.show()

      elif ch==4:
        g4 = df.plot(x='Date', y='Close', figsize=(10, 6))

        # add labels and title
        g4.set_xlabel("Date")
        g4.set_ylabel("Close")
        g4.set_title("Stock prices")

        plt.xlim([7333,9848])
        # show the plot
        plt.show()

      elif ch==5:
        g5 = df.plot(x='Date', y='Close', figsize=(10, 6))

        # add labels and title
        g5.set_xlabel("Date")
        g5.set_ylabel("Close")
        g5.set_title("Stock prices")

        plt.xlim([9849,10469])
        # show the plot
        plt.show()

      else :
        print("Please enter valid year option.")

      n = int(input("Do you want to continue to view graph?\n1.Yes\n2.No\n"))

    if n==2:
      print("Thank you!\n")


  else :
    print("Please enter valid year option.")

  p = int(input("\nDo you want to continue?\n1.Yes\n2.No\n"))

if p==2:
  print("\nThank you!")

# Calculating min errors
# split the data into training and testing sets

Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error: ",mse)

rsme = math.sqrt(mse)
print("Root Mean Squared Error: ", rsme)

mae = mean_absolute_error(Y_test, Y_pred)
print("Mean Absolute Error: ", mae)

# Calculate accuracy score
from sklearn.metrics import r2_score
r2 = r2_score(Y_test,Y_pred)
print("Accuracy: ")
print(r2 *100)
print()