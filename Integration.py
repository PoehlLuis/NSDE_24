import numpy as np

def stiffnessMatrix(order,coords,mat_tensor):
    """
     :parameter
     order: order of the stiffness matrix
     coords: global coordinates of the Elements
     mat_tensor: material tensor should be 2x2 ?
     :returns
     Ke: stiffness matrix of each element
     """
    points, weight = np.polynomial.legendre.leggauss(order)
    Ke = np.zeros((4, 4))
    for i in range(order):
        for j in range(order):
            for j in range(order):
                xi = points[i]
                eta = points[j]

                J = Jacobian(xi, eta, coords)
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)

                dN_dxi, dN_deta = dNi_(xi, eta)

                global_Na = np.zeros(4)
                global_Nb = np.zeros(4)
                for k in range(4):
                    global_Na[k] = invJ[0, 0] * dN_dxi[k] + invJ[0, 1] * dN_deta[k]
                    global_Nb[k] = invJ[1, 0] * dN_dxi[k] + invJ[1, 1] * dN_deta[k]
                B = np.array([global_Na, global_Nb])

                Ke += (B.T @ mat_tensor @ B) * detJ * weight[i] * weight[j]
    return Ke


def dNi_(xi, eta):
    dNi_dxi = np.array([-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)])
    dNi_deta = np.array([-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)])
    return dNi_dxi, dNi_deta

def Jacobian(xi, eta, glob_coords):
    dNi_dxi, dNi_deta = dNi_(xi, eta)
    J = np.zeros((2, 2))
    for i in range(4):
        J[0, 0] += dNi_dxi[i] * glob_coords[i][0]
        J[0, 1] += dNi_dxi[i] * glob_coords[i][1]
        J[1, 0] += dNi_deta[i] * glob_coords[i][0]
        J[1, 1] += dNi_deta[i] * glob_coords[i][1]
    return J

def Na(xi, eta, node):
    if node == 1:
        return (1 - xi) * (1 - eta) / 4
    elif node == 2:
        return (1 + xi) * (1 - eta) / 4
    elif node == 3:
        return (1 + xi) * (1 + eta) / 4
    elif node == 4:
        return (1 - xi) * (1 + eta) / 4
    return 0

def rhs(order, glob_coords, rho):
    points, weight = np.polynomial.legendre.leggauss(order)
    fe = np.zeros(4)
    total_detJ = 0.0
    for i in range(order):
        for j in range(order):
            xi = points[i]
            eta = points[j]
            J = Jacobian(xi, eta, glob_coords)
            detJ = np.linalg.det(J)
            total_detJ += detJ

            N = np.array([Na(xi, eta, 1), Na(xi, eta, 2), Na(xi, eta, 3), Na(xi, eta, 4)])
            fe += N * rho * detJ * weight[i] * weight[j]
            print(f"xi: {xi}, eta: {eta}, detJ: {detJ}, N: {N}, weight[i]: {weight[i]}, weight[j]: {weight[j]}, fe: {fe}")
    # Normalize by total determinant of Jacobian
    fe_normalized = fe / total_detJ
    return fe_normalized

# Example global coordinates
# glob_coords = np.array([[0, 0], [30, 0], [30, 30], [0, 30]])

# # Example parameters
# order = 2  # Using 2-point Gauss quadrature
# rho = 1.0  # Example density

# # Calculate the normalized RHS vector
# fe_normalized = rhs(order, glob_coords, rho)
# print(f"Normalized RHS vector: {fe_normalized}")

