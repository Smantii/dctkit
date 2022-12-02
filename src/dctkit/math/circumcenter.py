import numpy as np


def circumcenter(s, node_coord):
    """Compute the circumcenter of a given simplex s.

       (References: Bell, Hirani, PyDEC: Software and Algorithms
       for Discretization of Exterior Calculus, 2012, Section 10.1)

        Args:
            s (np.array): matrix containing the IDs of the nodes beloging to s.
            node_coord (np.array): coordinates of every node of the cell complex
                          in which s is defined.
        Returns:
            circumcenter (np.array): coordinates of the circumcenter of s.
    """
    # store the coordinates of the nodes in s
    simplex_coord = node_coord[s[:]]
    rows, cols = simplex_coord.shape

    assert (rows <= cols + 1)

    # construct the matrix A
    A = np.bmat([[2*np.dot(simplex_coord, simplex_coord.T), np.ones((rows, 1))],
                [np.ones((1, rows)) ,  np.zeros((1, 1))]])
    b = np.hstack((np.sum(simplex_coord * simplex_coord, axis=1), np.ones((1))))
    # barycentric coordinates x of the circumcenter are the solution
    # of the linear sistem Ax = b
    bary_coords = np.linalg.solve(A, b)
    bary_coords = bary_coords[:-1]
    # compute the coordinates of the circumcenter
    circumcenter = np.dot(bary_coords, simplex_coord)

    return circumcenter


def all_circumcenters(S, node_coord):
    """Compute the circumcenters of all the p-simplices S of a cell complex.

    Args:
        S (np.array): matrix of node tags in which any rows represents the tag
                      of a p-simplex.
        node_coord (np.array): coordinates of every node of the cell complex
                          in which s is defined.
    Returns:
        C (np.array): matrix in which rows are the coordinates of the circumcenter
                      of the relative rows of S.
    """
    C = np.empty((S.shape[0], node_coord.shape[1]))
    for i in range(S.shape[0]):
        C[i, :] = circumcenter(S[i, :], node_coord)
    return C