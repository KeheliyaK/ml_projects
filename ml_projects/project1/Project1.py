#----------------------------------
# Import the plugins
#__________________________________
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl as op

#----------------------------------
# Load and Process data
#__________________________________

data_path="proj1Dataset.xlsx" #define the path to the data
df=pd.read_excel(data_path, engine="openpyxl") #extract the data from the Excel file using openpyxl engine and save it
df1=df.dropna() #dropping the raws of data with null data

weight=df1["Weight"].to_numpy()
ones=np.ones((weight.shape[0],1),dtype=int) # create an array filled with number '1' using 'np.ones' function

# Normalize the Weight feature for better numerical stability in gradient descent.
mean_weight = np.mean(weight)
std_weight = np.std(weight)
weight_norm = (weight - mean_weight) / std_weight

X_matrix=np.hstack((weight_norm.reshape(-1,1),ones))#create X matrix appending ones and reshaping the array
t_matrix=df1["Horsepower"].to_numpy().reshape(-1,1)#create t matrix by extracting and reshaping

X_trns=X_matrix.T # taking the X transpose
X1=np.dot(X_trns,X_matrix)

#----------------------------------
# 1. Closed form solution
#__________________________________

X1_inv=np.linalg.inv(X1)
X2=np.dot(X1_inv,X_trns)
W_closed=np.dot(X2,t_matrix)

# Convert closed form parameters back to the original scale:
slope_closed = W_closed[0] / std_weight
intercept_closed = W_closed[1] - (W_closed[0] * mean_weight) / std_weight

y_closed=slope_closed*weight+intercept_closed


#----------------------------------
# 2. Gradient Descent method
#__________________________________

W_g=np.random.randn(2,1) #initializing random numbers for W vector
print('Initial_W_g= ',W_g)
rho=0.0001 #initializing a small value for learning rate
epsilon=1e-5 #Converge threshold
max_iterations=10000000 #Maximum iterations to prevent infinite loop

#Gradient Descent loop to calculate converged W values
iteration =0
m = X_matrix.shape[0]
while True:
    gradient = (1 / m) * X_matrix.T.dot(X_matrix.dot(W_g) - t_matrix)
    W_gradient=W_g-(rho*gradient) #update the weight function
    if np.linalg.norm(W_gradient-W_g)<epsilon:
        print(f"Converged at iteration{iteration}")
        break
    W_g=W_gradient #Update the W_g for next iteration
    iteration+=1

    # Stop if it takes too long (to prevent infinite loops)
    if iteration >= max_iterations:
        print("Reached max iterations, stopping optimization.")
        break

# Convert gradient descent parameters back to the original scale:
slope_g = W_gradient[0] / std_weight
intercept_g = W_gradient[1] - (W_gradient[0] * mean_weight) / std_weight
y_gradient=slope_g*weight+intercept_g

# Printing the results
print('slope from closed form', slope_closed)
print('intercept from closed form',intercept_closed)
print('slope from gradient descent', slope_g)
print('intercept from gradient descent', intercept_g)

# Plotting the results
plt.figure(figsize=(12, 5))

# Plot for Closed-form solution
plt.subplot(1, 2, 1)
plt.scatter(weight, t_matrix, color='red', marker='x')
plt.plot(weight, y_gradient, color='blue', label="Closed Form")
plt.xlabel("Weight")
plt.ylabel("Horsepower")
plt.title("Matlab's 'carbig' dataset ")
plt.legend()

# Plot for Gradient Descent
plt.subplot(1, 2, 2)
plt.scatter(weight, t_matrix, color='red', marker='x')
plt.plot(weight, y_closed, color='green', label="Gradient Descent")
plt.xlabel("Weight")
plt.ylabel("Horsepower")
plt.title("Matlab's 'carbig' dataset ")
plt.legend()

# Show plots
plt.show()
