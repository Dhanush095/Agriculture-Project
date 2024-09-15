import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv('your_dataset.csv')

# Assuming the dataset has columns: Moisture, Humidity, Temperature, Time to Irrigate
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y.reshape(-1, 1)).flatten()

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=1/3, random_state=0)

# Fitting Support Vector Regression to the Training set
regressor = SVR(kernel='rbf')  # You can try different kernels like 'linear', 'poly', etc.
regressor.fit(X_train, y_train)

# Taking user input for moisture, humidity, and temperature
moisture_input = float(input("Enter the moisture level: "))
humidity_input = float(input("Enter the humidity level: "))
temperature_input = float(input("Enter the temperature: "))

# Scaling user input and predicting the scaled time
scaled_input = sc_X.transform(np.array([[moisture_input, humidity_input, temperature_input]]))
scaled_predicted_time = regressor.predict(scaled_input)
predicted_time = sc_y.inverse_transform(scaled_predicted_time.reshape(-1, 1)).flatten()
print(f"For a moisture level of {moisture_input}, humidity of {humidity_input}, and temperature of {temperature_input}, the predicted time is: {predicted_time[0]} seconds")
# R-squared score for Support Vector Regression
svm_r2 = regressor.score(X_test, y_test)  # Or r2_score(y_test, regressor.predict(X_test))
svm_accuracy = svm_r2 * 100
print(f"R-squared score for Support Vector Regression: {svm_r2:.2f}")
print(f"Accuracy score for Support Vector Regression: {svm_accuracy:.2f}%")

# Visualizing the Training set results (for one variable - e.g., moisture)
plt.scatter(X_train[:, 0], y_train, color='red')  # Assuming moisture is the first column
plt.scatter(X_train[:, 0], regressor.predict(X_train), color='blue')  # Assuming moisture is the first column
plt.title('Time vs Moisture (Training set)')
plt.xlabel('Moisture')
plt.ylabel('Time to Irrigate (seconds)')
plt.show()