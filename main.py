#%%
import numpy as np
import csv

from input_handling import *
from mesh_generation import *
from finite_element import *
from finite_element_procedure import *
from line import *
from exportRes import *

#%%

'''
Numbering of Fem_Knots and Elements 

Global Node and Equation Numbers
From left top to right top, then next row to the right ( as one reads )
Local Node Numbers starting from bottom left in a counter-clockwise rotation

    1---------2---------3
    | 4     3 | 4     3 |
    |    1    |    2    |
    | 1     2 | 1     2 |
    4---------5---------6
    | 4     3 | 4     3 |
    |    3    |    4    |
    | 1     2 | 1     2 |
    7---------8---------9

    Outest Layer: Global Node Number
    Second Layer: Local Node Number
    Number in the Middle: Global Element Number

'''

#%%
# TODO 
width, height, order_num_int, amount_of_nodes_per_axis = getGeometryInputs_hard_coded()

line_start, line_end, line_value_function, amount_of_line_points = getLineInputs_hard_coded(width, height)

mat_tensor = getMaterialTensor_hard_coded()

boundary_conditions_ = getBCInputs_hard_coded()


# de drei zeilen kennen gelöscht werden
boundary_conditions = [[boundary_conditions_[0][0],boundary_conditions_[0][1]],
[boundary_conditions_[1][0],boundary_conditions_[1][1]],[boundary_conditions_[2][0],boundary_conditions_[2][1]],
[boundary_conditions_[3][0],boundary_conditions_[3][1]],[Type.Dirichlet.value, line_value_function]]



# creates the mesh with all the nodes
mesh_coords = createMesh(width, height, amount_of_nodes_per_axis)

# gets amount of coordinate pairs
array_size = mesh_coords.shape[0]

# get the coordinates of the Line
line_coords = []
line_coords = getLineCoordinates(line_start, line_end, mesh_coords, amount_of_line_points)

line_values = []
if line_coords:  # checks if list is not empty
    line_values = getLineValues(line_coords, line_value_function)


# creates the array containing the node-equations
NE_array = get_node_equation_array(array_size, mesh_coords, line_coords)

# creates the finite elements of the domain
finite_elements = element_generation(NE_array, amount_of_nodes_per_axis, height, width, boundary_conditions)
# System-matrix K

K = np.zeros([array_size, array_size])

rho = 1
rhs = np.zeros(array_size)
K, rhs = assembling_algorithm(finite_elements, 4, K, rhs, mat_tensor, order_num_int, rho)

#Resize the Arrays to the size we need them by cutting away the 0 entries
K = K[:array_size - len(line_values) - 36, :array_size - len(line_values) - 36]
rhs = rhs[:array_size - len(line_values) - 36]

u = np.linalg.solve(K, rhs)

# Add togheter the calculated values of the Nodes with Node-Equation-Numbers with the Boundaries
# Muas umgeschrieben werden, damit werte von Linie verwendet werden. 
# odr glabi spater brauchen mir den toal gor nit weil mirs in algorithmus schun richtig zommensetzen
counter = 0
out = []
for i in range(9):
    out.append(boundary_conditions[0][1])
for i in range(8):
     out.append(boundary_conditions[1][1])
     out.append(boundary_conditions[3][1])
     for j in range(8):
        out.append(u[counter]) 
        counter += 1
out.append(boundary_conditions[1][1])
out.append(boundary_conditions[3][1])
for i in range(9):
    out.append(boundary_conditions[2][1])

out = np.array(out)


# Node Connectivity Matrix
global_node_numbers_list = []
for j in range(0, 9):
     for i in range(0, 9):
          arr = np.array([amount_of_nodes_per_axis *(j+1) + i, amount_of_nodes_per_axis*(j+1) + (i+1), amount_of_nodes_per_axis *(j) + i, amount_of_nodes_per_axis *(j) + (i+1)])
          global_node_numbers_list.append(arr)
        
global_node_numbers_array = np.array(global_node_numbers_list)

# Export Writer
export_writer = EXPORT(4,                       # Nodes per Element
                       len(finite_elements),    # Amount of Elements
                       array_size,              # Amount of Nodes
                       2,                       # Dimension (2D)
                       out,                     # Result Vector
                       mesh_coords,             # Node Coordinates
                       global_node_numbers_array, # Node Connectivity Matrix
                       1)                       # Degree of Freedom per Node

export_writer.writeResults()


# Write values to a .csv file for the CI-CD System
filename = 'program_output.csv'

file = open(filename, 'w', newline='')

fields = ['coordinates', 'value']
writer = csv.DictWriter(file, fieldnames=fields, delimiter=';')
writer.writeheader()

for i, value in enumerate(out):
        writer.writerow({'coordinates': mesh_coords[i], 'value': out[i]})

# %%
