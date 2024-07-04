# %%
import numpy as np
import csv

from input_handling import *
from mesh_generation import *
from finite_element import *
from finite_element_procedure import *
from line import *
from exportRes import *

# %%

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

# %%


rho = 1
width, height, order_num_int, amount_of_nodes_per_axis = getGeometryInputs_hard_coded()
# width, height, order_num_int, amount_of_nodes_per_axis = getGeometryInputs()

# line_start, line_end, line_value_function, amount_of_line_points = getLineInputs(width, height)
line_start, line_end, line_value_function, amount_of_line_points = getLineInputs_hard_coded(width, height)

# mat_tensor = getMaterialTensor()
mat_tensor = getMaterialTensor_hard_coded()

boundary_conditions = getBCInputs_hard_coded()

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
finite_elements = element_generation(NE_array, amount_of_nodes_per_axis, height, width, amount_of_nodes_per_axis)
# System-matrix K

K = np.zeros([array_size, array_size])

rhs = np.zeros(array_size)
K, rhs = assembling_algorithm(finite_elements, 4, K, rhs, mat_tensor, order_num_int, rho)
K, rhs = bc.apply_boundary_conditions(K, rhs, boundary_conditions, bc.get_boundary_nodes(mesh_coords, width, height),
                                      width, height, amount_of_nodes_per_axis)
#K, rhs = apply_line_values(K, rhs, mesh_coords, line_coords, line_values)

u = np.linalg.solve(K, rhs)

# Node Connectivity Matrix
global_node_numbers_list = []
for j in range(0, amount_of_nodes_per_axis - 1):
    for i in range(0, amount_of_nodes_per_axis - 1):
        arr = np.array([amount_of_nodes_per_axis * (j + 1) + i, amount_of_nodes_per_axis * (j + 1) + (i + 1),
                        amount_of_nodes_per_axis * (j) + i, amount_of_nodes_per_axis * (j) + (i + 1)])
        global_node_numbers_list.append(arr)

global_node_numbers_array = np.array(global_node_numbers_list)

# Export Writer
export_writer = EXPORT(4,  # Nodes per Element
                       len(finite_elements),  # Amount of Elements
                       array_size,  # Amount of Nodes
                       2,  # Dimension (2D)
                       u,  # Result Vector
                       mesh_coords,  # Node Coordinates
                       global_node_numbers_array,  # Node Connectivity Matrix
                       1)  # Degree of Freedom per Node

export_writer.writeResults()

# Write values to a .csv file for the CI-CD System
filename = 'program_output.csv'

file = open(filename, 'w', newline='')

fields = ['coordinates', 'value']
writer = csv.DictWriter(file, fieldnames=fields, delimiter=';')
writer.writeheader()

for i, value in enumerate(u):
    writer.writerow({'coordinates': mesh_coords[i], 'value': u[i]})

# %%

# %%