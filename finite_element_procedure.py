import numpy as np
from Integration import *



# Knotengleichungsarray / Node-Equation-Array
# Inputs: amount of nodes in the domain
#         coordinates of the domain 
#         coordinates of the line in the domain with default empty array as default value
# Output: Array containing node-equation-numbers
def get_node_equation_array(array_size, mesh_coords, line_coords = []):

    node_equation_array = np.zeros(array_size, dtype=int)

    min_x = np.min(mesh_coords[:, 0])
    max_x = np.max(mesh_coords[:, 0])
    min_y = np.min(mesh_coords[:, 1])
    max_y = np.max(mesh_coords[:, 1])

    i, j = 0, 1  # i for indexing and j for value of globale gleichungsnummer
    for (x, y) in mesh_coords:
        #print(x, y)
        if(x != min_x and x != max_x and y != min_y and y != max_y):
            # check if this mesh_coord is one of the line_coordinates
            found = False
            for l in range(len(line_coords)):
                if((x, y) == line_coords[l]):
                    found = True
                    break

            #if((x, y) != line_coords[l] for l in range(len(line_coords))):
            if(found == False):
                node_equation_array[i] = j
                j += 1
                
        i += 1

    return node_equation_array

# Gleichungsarray / Equation-Array
# Inputs: array containing the finite elements of the domain
#         local node number [1 -4]
#         element number
# Output: global equation number
def EQ(finite_elements, local_number, element_number):
    return finite_elements[element_number - 1].get_global_node_numbers()[local_number - 1]


# Assembling-Algorithm from the VO
# Inputs: array of finite-element-objects
#         amount of element nodes
#         System-matrix K
# Output: System-matrix K
def assembling_algorithm(finite_elements, number_of_element_nodes, K, material_tensor, order):
    for e in range(1, finite_elements.size):
        global_coords = finite_elements[e].get_global_coords()
        for a in range(1, number_of_element_nodes):
            eq1 = EQ(finite_elements, a, e)
            if(eq1 > 0):
                for b in range(1, number_of_element_nodes):
                    eq2 = EQ(finite_elements, b, e)
                    if(eq2 > 0):
                        # Some funny things
                        K[eq1, eq2] += stiffnessMatrix(order,global_coords,material_tensor)
    return K

# Assembling-Algorithm from the VO
# Inputs: array of finite-element-objects
#         amount of element nodes
#         System-matrix K
# Output: System-matrix K
def assembling_algorithm2(finite_elements, number_of_element_nodes, K, Rhs, material_tensor, order, rho):
    for e in range(finite_elements.size):
        # Extract the global coordinates for the current element
        global_coords = finite_elements[e].get_global_coords()

        # Compute the element stiffness matrix
        element_stiffness_matrix = stiffnessMatrix(order, global_coords, material_tensor)
        #Compute the elemnts rhs matrix
        element_force_vector = rhs(order, global_coords, rho)

        for a in range(1, number_of_element_nodes):
            eq1 = EQ(finite_elements, a, e)
            if eq1 > 0:
                Rhs[eq1 - 1] += element_force_vector[a]
                for b in range(1, number_of_element_nodes):
                    eq2 = EQ(finite_elements, b, e)
                    if eq2 > 0:
                        # Add the appropriate component of the element stiffness matrix to K
                        K[eq1 - 1, eq2 - 1] += element_stiffness_matrix[a, b]
    return K, Rhs