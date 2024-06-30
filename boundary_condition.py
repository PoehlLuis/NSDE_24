# Class for boundary conditions containing value and type of bc(Dirilecht or Neumann)
import numpy as np
import mesh_generation


class BoundaryCondition:
    def __init__(self, value, bc_type):
        self.value = value
        self.bc_type = bc_type

    # Extracts the boundary nodes and returns them as 2D Array where row is for left,top,right,bottom


# and the col is another array with the corresponding nodes
# Of each side left side is the corner from inside perspective
''''
    left:    Node    Node    Node    Node ..... 
    top:    Node    Node    Node    Node ....
    right:  Node    ...
    bottom: Node    ...
'''


def get_boundary_nodes(mesh_coords, width, height):
    x_values = mesh_coords[:, 0]  # Extract all x-coordinates
    y_values = mesh_coords[:, 1]  # Extract all y-coordinates

    left_nodes = mesh_coords[(x_values == 0), :]
    right_nodes = mesh_coords[(x_values == width), :]
    top_nodes = mesh_coords[(x_values != 0) & (x_values != width) & (y_values == height), :]
    bottom_nodes = mesh_coords[(x_values != 0)& (x_values != width) & (y_values == 0), :]

    boundary_nodes = (left_nodes, top_nodes, right_nodes, bottom_nodes)
    return boundary_nodes


def apply_boundary_conditions(system_matrix, rhs, boundary_conditions, boundary_nodes, width, height,
                              amount_of_nodes_per_axis):
    for side, b_nodes in zip(boundary_conditions, boundary_nodes):
        for node in b_nodes:
            node_index = find_global_node_nr(node, width, height, amount_of_nodes_per_axis) - 1
            bc_value = side.value
            if side.bc_type == "Dirichlet":
                system_matrix[node_index, :] = 0
                system_matrix[:, node_index] = 0  # TODO: Maybe delete/change this?? # ToDo: We probably need this to maintein the tridiagonal structure
                system_matrix[node_index, node_index] = 1
                rhs[node_index] = bc_value
            elif side.bc_type == "Neumann":
                rhs[node_index] += bc_value
            else:
                raise ValueError("Invalid boundary condition type")
    return system_matrix, rhs


def find_global_node_nr(node, width, height, amount_of_nodes_per_axis):
    x_step_size = width / (amount_of_nodes_per_axis - 1)
    y_step_size = height / (amount_of_nodes_per_axis - 1)
    a = node[0]
    b = node[1]
    node_in_row = node[0] / x_step_size + 1
    node_in_col = amount_of_nodes_per_axis - node[1] / y_step_size

    return int(node_in_row + (node_in_col - 1) * amount_of_nodes_per_axis)