import numpy as np
from scipy.spatial import Delaunay

# Original square from the test
pts = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
tri = Delaunay(pts)
print("Scipy Delaunay triangulation of unit square:")
print(tri.simplices)
print("\nNote: Both [[0,1,2],[0,2,3]] and [[0,1,3],[1,2,3]] are valid Delaunay triangulations")
print("for a cocircular quad. Scipy picks one arbitrarily.")
