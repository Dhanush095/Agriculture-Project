import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('your_dataset.csv')  # Replace 'your_dataset.csv' with your actual dataset filename
X = dataset.iloc[:, :-1].values  # Use all columns except the last one as features
y = dataset.iloc[:, -1].values  # Use the last column as the target variable

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fitting Linear Regression to the Training set
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# Taking user input for moisture, humidity, and temperature
moisture_input = float(input("Enter the moisture level: "))
humidity_input = float(input("Enter the humidity level: "))
temperature_input = float(input("Enter the temperature: "))

# Predicting the time for the user input parameters
input_parameters = np.array([[moisture_input, humidity_input, temperature_input]])
predicted_time = linear_regressor.predict(input_parameters)
print(f"For input parameters: Moisture={moisture_input}, Humidity={humidity_input}, Temperature={temperature_input}, the predicted time is: {predicted_time[0]} seconds")

# R-squared score for Linear Regression on the test set
linear_r2 = r2_score(y_test, linear_regressor.predict(X_test))
print(f"R-squared score for Linear Regression on the test set: {linear_r2:.2f}")

# Visualising the Training set results
plt.scatter(X_train[:, 0], y_train, color='red', label='Actual')
plt.scatter(X_train[:, 0], linear_regressor.predict(X_train), color='blue', label='Predicted')  # Plotting the predicted values
plt.title('Time vs Moisture (Training set)')
plt.xlabel('Moisture')
plt.ylabel('Time')
plt.legend()
plt.show()