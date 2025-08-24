import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read in matrix data CSV file
data = pd.read_csv("matrix_data.csv", header=None)

#Specify dimensions of matrix
xdim = 5
ydim = 5

#Allocate arrays for mesh grid based on matrix dimensions
x = np.linspace(-xdim, ydim, 5000)
y = np.linspace(-xdim, ydim, 5000)

#Instantiate list for 1D arrays to be converted to 2D matrix
arr_list = []

#Create mesh grid for matrix
x_1, y_1 = np.meshgrid(x, y)

#For each row in CSV file, append row to list of 1D arrays
for index, row in enumerate(data.iterrows()):
    arr_row = np.array(row[1].values[0:-1])
    arr_list.append(arr_row)

#Convert list of 1D arrays to matrix
matrix = np.vstack(arr_list)

#Plot results as a surface plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x_1, y_1, matrix, cmap='cool', edgecolor='blue')
ax.set_title('Surface Plot of Matrix Calculated on the GPU')
plt.show()