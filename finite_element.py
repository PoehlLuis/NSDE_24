import numpy as np

class FEM_Element:
    def __init__(self, global_element_number, global_node_numbers, global_cords):
        self.global_element_number = global_element_number
        self.global_node_numbers = global_node_numbers
        self.global_cords = global_cords

    def get_global_node_numbers(self):
        return self.global_node_numbers
    
    def get_global_element_number(self):
        return self.global_element_number

    def get_global_coords(self):
        return self.global_cords

# Generation of the finite elements in the domain
# Inputs: Array containing the the node equation numbers
#         amount of nodes per axis
# Output: Array containing the finite Elements
def element_generation(node_equation_array, dimension, height, width):
    elements = np.empty((dimension - 1)**2, dtype=FEM_Element)

    delta_x = width / (dimension - 1)
    delta_y = height / (dimension - 1)

    element_number = 1
    for y in range(dimension - 1): 
        for x in range(dimension - 1):
            node_numbers = np.array([node_equation_array[dimension * (y+1) + x], 
                                        node_equation_array[dimension * (y+1) + (x+1)], 
                                        node_equation_array[dimension * y + (x+1)], 
                                        node_equation_array[dimension * y + x]])

            global_coords = np.array([[delta_x*((element_number - 1) % (dimension-1)),height - y * delta_y],
                                     [delta_x*((element_number - 1) % (dimension-1)) + delta_x, height - y * delta_y],
                                     [delta_x*((element_number - 1) % (dimension-1)), height - y * delta_y - delta_y],
                                     [delta_x*((element_number - 1) % (dimension-1)) + delta_x, height - y * delta_y - delta_y]])

            elements[element_number - 1] = FEM_Element(element_number, node_numbers, global_coords)
            element_number += 1

    return elements

