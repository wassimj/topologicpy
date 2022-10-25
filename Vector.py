from topologicpy import topologic
import numpy as np
import numpy.linalg as la
import math

class Vector():
    @staticmethod
    def angle_between(v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2' """
        n_v1=la.norm(v1)
        n_v2=la.norm(v2)
        if (abs(np.log10(n_v1/n_v2)) > 10):
            v1 = v1/n_v1
            v2 = v2/n_v2
        cosang = np.dot(v1, v2)
        sinang = la.norm(np.cross(v1, v2))
        return np.arctan2(sinang, cosang)

    def multiplyVector(vector, magnitude, tolerance):
        """ Returns a vector that multiplies the input vector by the input magnitude """
        oldMag = 0
        for value in vector:
            oldMag += value ** 2
        oldMag = oldMag ** 0.5
        if oldMag < tolerance:
            return [0,0,0]
        newVector = []
        for i in range(len(vector)):
            newVector.append(vector[i] * magnitude / oldMag)
        return newVector

    def unitizeVector(vector):
        """
        Description
        -----------
        Returns a unitized version of the input vector.

        Parameters
        ----------
        vector : list
            The input vector.

        Returns
        -------
        list
            The unitized vector.

        """
        mag = 0
        for value in vector:
            mag += value ** 2
        mag = mag ** 0.5
        unitVector = []
        for i in range(len(vector)):
            unitVector.append(vector[i] / mag)
        return unitVector

    def normalize(u):
      return u / np.linalg.norm(u)

    def get_normal(vertices):
      return normalize(np.cross(vertices[1] - vertices[0], vertices[-1] - vertices[0]))

# From https://gis.stackexchange.com/questions/387237/deleting-collinear-vertices-from-polygon-feature-class-using-arcpy
def are_collinear(v1, v2, v3, tolerance=0.5):
  e1 = topologic.EdgeUtility.ByVertices([v2, v1])
  e2 = topologic.EdgeUtility.ByVertices([v2, v3])
  rad = topologic.EdgeUtility.AngleBetween(e1, e2)

  return abs(math.sin(rad)) < math.sin(math.radians(tolerance))

def removeCollinearEdges(wire, angTol):
  vertices = getSubTopologies(wire, topologic.Vertex)

  indexes_of_vertices_to_remove = [
    idx for idx, vertex in enumerate(vertices)
    if are_collinear(vertices[idx-1], vertex, vertices[idx+1 if idx+1 < len(vertices) else 0], angTol)
  ]

  vertices_to_keep = [
    val for idx, val in enumerate(vertices)
    if idx not in indexes_of_vertices_to_remove
  ]

  return vertices_to_keep



def projectFace(face, other_face):
  normal = topologic.FaceUtility.NormalAtParameters(face, 0.5, 0.5)
  n = [normal[0], normal[1], normal[2]]
  point = topologic.FaceUtility.VertexAtParameters(face, 0.5, 0.5)
  d = np.dot(n, [point.X(), point.Y(), point.Z()])

  other_normal = topologic.FaceUtility.NormalAtParameters(other_face, 0.5, 0.5)
  if np.dot(n, [other_normal[0], other_normal[1], other_normal[2]]) + 1 > 1e-6:
    return [None, None]

  other_point = topologic.FaceUtility.VertexAtParameters(other_face, 0.5, 0.5)
  dist = -np.dot(n, [other_point.X(), other_point.Y(), other_point.Z()]) + d
  if dist < 1e-6:
    return [None, None]

  top_space_boundary = boolean(face, topologic.TopologyUtility.Translate(other_face, dist*normal[0], dist*normal[1], dist*normal[2]), "Intersect")
  if top_space_boundary is None:
    return [None, None]

  top_space_boundary = getSubTopologies(top_space_boundary, topologic.Face)
  if not top_space_boundary:
    return [None, None]

  return [dist, top_space_boundary[0]]