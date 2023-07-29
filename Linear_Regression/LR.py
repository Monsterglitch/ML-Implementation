# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Sample values for testing
Epochs = 1500
gradient = 0
y_intercept = 0
Learning_rate = 0.0001 

# Reading data from the Sample_dataset 
def read_data():
    return pd.read_csv('Sample_dataset.csv')

df = read_data() # Storing it in df as a pandas dataframe

# Function to reduce the loss in the function
def gradient_descent(gradient, y_intercept, df, L):
    n = len(df) # Length of the dataframe
    new_gradient = 0
    new_y_intercept = 0
    for i in range(n):
        x = df.iloc[i].x # extracting values of x from the dataset
        y = df.iloc[i].y # extracting values of y from the dataset
        new_gradient = -(2 / n) * x * (y - (gradient * x + y_intercept)) # partial derivative of the loss function (E) with respect to gradient
        new_y_intercept = -(2 / n) * (y - (gradient * x + y_intercept)) # partial derivative of the loss function (E) with respect to y_intercept
    
    # Now the steepest ascent is used to find the steepest descent (reverse) using Learning_rate
    final_gradient = gradient - L * new_gradient 
    final_y_intercept = gradient - L * new_y_intercept
    return final_gradient, final_y_intercept

# Training for a 1500 Epochs
for i in range(Epochs):
    if Epochs % 50 == 0:
        print(f"For Epoch: {i}")
    gradient, y_intercept = gradient_descent(gradient, y_intercept, df, Learning_rate)
    
# Plotting the scatter plot
plt.scatter(df.x, df.y, color = "black")
plt.plot(list(range(1, 100)), [gradient * x + y_intercept for x in range(1, 100)], color = "red") # The values within the range() is the limit x lies in
plt.show()