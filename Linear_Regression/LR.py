import pandas as pd
import matplotlib.pyplot as plt

def read_data():
    return pd.read_csv('Sample_dataset.csv')

df = read_data()

def gradient_descent(gradient, y_intercept, df, L):
    n = len(df)
    new_gradient = 0
    new_y_intercept = 0
    for i in range(n):
        x = df.iloc[i].x
        y = df.iloc[i].y
        new_gradient = -(2 / n) * x * (y - (gradient * x + y_intercept))
        new_y_intercept = -(2 / n) * (y - (gradient * x + y_intercept))
    
    final_gradient = gradient - L * new_gradient
    final_y_intercept = gradient - L * new_y_intercept
    return final_gradient, final_y_intercept

Epochs = 1500
gradient = 0
y_intercept = 0
Learning_rate = 0.0001

for i in range(Epochs):
    if Epochs % 50 == 0:
        print(f"Epoch: {i}")
    gradient, y_intercept = gradient_descent(gradient, y_intercept, df, Learning_rate)
    
plt.scatter(df.x, df.y, color = "black")
plt.plot(list(range(1, 100)), [gradient * x + y_intercept for x in range(1, 100)], color = "red")
plt.show()