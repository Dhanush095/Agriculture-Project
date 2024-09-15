import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('your_dataset.csv')

# Assuming the dataset has columns: Moisture, Humidity, Temperature, Time to Irrigate
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fitting Random Forest Regression to the Training set
regressor = RandomForestRegressor(n_estimators=10, random_state=0)  # You can adjust the number of estimators
regressor.fit(X_train, y_train)

# Taking user input for moisture, humidity, and temperature
moisture_input = float(input("Enter the moisture level: "))
humidity_input = float(input("Enter the humidity level: "))
temperature_input = float(input("Enter the temperature: "))

# Predicting the time for the user input
predicted_time = regressor.predict(np.array([[moisture_input, humidity_input, temperature_input]]))
print(f"For a moisture level of {moisture_input}, humidity of {humidity_input}, and temperature of {temperature_input}, the predicted time is: {predicted_time[0]} seconds")

# R-squared score for Random Forest Regression
rf_r2 = regressor.score(X_test, y_test)  # Or r2_score(y_test, regressor.predict(X_test))
rf_accuracy = rf_r2 * 100
print(f"R-squared score for Random Forest Regression: {rf_r2:.2f}")
print(f"Accuracy score for Random Forest Regression: {rf_accuracy:.2f}%")


# Visualizing the Training set results (for one variable - e.g., moisture)
plt.scatter(X_train[:, 0], y_train, color='red')  # Assuming moisture is the first column
plt.scatter(X_train[:, 0], regressor.predict(X_train), color='blue')  # Assuming moisture is the first column
plt.title('Time vs Moisture (Training set)')
plt.xlabel('Moisture')
plt.ylabel('Time to Irrigate (seconds)')
plt.show()