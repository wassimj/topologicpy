# Matrix Classes unit test

# importing libraries
import sys
sys.path.append("C:/Users/wassimj/Documents/GitHub")

import topologicpy
import topologic
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Cluster import Cluster
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Shell import Shell
from topologicpy.Cell import Cell
from topologicpy.Grid import Grid
from topologicpy.Matrix import Matrix


# Case 1 - ByRotation
# test 1
mat_rot = Matrix.ByRotation(rx=0, ry=0, rz=0, order='xyz')
assert isinstance(mat_rot, list), "Matrix.ByRotation. list"
assert len(mat_rot) == 4, "Matrix.ByRotation. List length should be 4"
assert len(mat_rot[0]) == 4, "Matrix.ByRotation. List length should be 4"
# test 2
mat_rot1 = Matrix.ByRotation(rx=30, ry=90, rz=0, order='xyz')
assert isinstance(mat_rot1, list), "Matrix.ByRotation. list"
assert len(mat_rot1) == 4, "Matrix.ByRotation. List length should be 4"
assert len(mat_rot1[0]) == 4, "Matrix.ByRotation. List length should be 4"
# test 3
mat_rot2 = Matrix.ByRotation(rx=30, ry=90, rz=10, order='xyz')
assert isinstance(mat_rot2, list), "Matrix.ByRotation. list"
assert len(mat_rot2) == 4, "Matrix.ByRotation. List length should be 4"
assert len(mat_rot2[0]) == 4, "Matrix.ByRotation. List length should be 4"


# Case 2 - ByScaling
# test 1
mat_scal = Matrix.ByScaling(sx=4, sy=4, sz=4)
assert isinstance(mat_scal, list), "Matrix.ByScaling. list"
assert len(mat_scal) == 4, "Matrix.ByRotation. List length should be 4"
assert len(mat_scal[0]) == 4, "Matrix.ByRotation. List length should be 4"
# test 2
mat_scal1 = Matrix.ByScaling(sx=4.5, sy=4.2, sz=4.1)
assert isinstance(mat_scal1, list), "Matrix.ByScaling. list"
assert len(mat_scal1) == 4, "Matrix.ByRotation. List length should be 4"
assert len(mat_scal1[0]) == 4, "Matrix.ByRotation. List length should be 4"

# Case 3 - Add
# test 1
mat_add = Matrix.Add(mat_rot,mat_scal)
assert isinstance(mat_add, list), "Matrix.Add. list"
assert len(mat_add) == 4, "Matrix.ByRotation. List length should be 4"
assert len(mat_add[0]) == 4, "Matrix.ByRotation. List length should be 4"
# test 2
mat_add1 = Matrix.Add(mat_rot,mat_rot1)
assert isinstance(mat_add1, list), "Matrix.Add. list"
assert len(mat_add1) == 4, "Matrix.ByRotation. List length should be 4"
assert len(mat_add1[0]) == 4, "Matrix.ByRotation. List length should be 4"
# test 3
mat_add2 = Matrix.Add(mat_rot2,mat_scal1)
assert isinstance(mat_add2, list), "Matrix.Add. list"
assert len(mat_add2) == 4, "Matrix.ByRotation. List length should be 4"
assert len(mat_add2[0]) == 4, "Matrix.ByRotation. List length should be 4"


# Case 4 - ByTranslation
# test 1
mat_tansl = Matrix.ByTranslation(tx=3, ty=4, tz=2)
assert isinstance(mat_tansl, list), "Matrix.ByTranslation. list"
assert len(mat_tansl) == 4, "Matrix.ByRotation. List length should be 4"
assert len(mat_tansl[0]) == 4, "Matrix.ByRotation. List length should be 4"
# test 2
mat_tansl1 = Matrix.ByTranslation(tx=-3, ty=-4, tz=-2)
assert isinstance(mat_tansl1, list), "Matrix.ByTranslation. list"
assert len(mat_tansl1) == 4, "Matrix.ByRotation. List length should be 4"
assert len(mat_tansl1[0]) == 4, "Matrix.ByRotation. List length should be 4"


# Case 5 - Multiply
# test 1
mat_mul = Matrix.Multiply(mat_add,mat_tansl)
assert isinstance(mat_mul, list), "Matrix.Multiply. list"
assert len(mat_mul) == 4, "Matrix.ByRotation. List length should be 4"
assert len(mat_mul[0]) == 4, "Matrix.ByRotation. List length should be 4"
# test 2
mat_mul1 = Matrix.Multiply(mat_rot,mat_tansl)
assert isinstance(mat_mul1, list), "Matrix.Multiply. list"
assert len(mat_mul1) == 4, "Matrix.ByRotation. List length should be 4"
assert len(mat_mul1[0]) == 4, "Matrix.ByRotation. List length should be 4"
# test 3
mat_mul2 = Matrix.Multiply(mat_rot,mat_scal1)
assert isinstance(mat_mul2, list), "Matrix.Multiply. list"
assert len(mat_mul2) == 4, "Matrix.ByRotation. List length should be 4"
assert len(mat_mul2[0]) == 4, "Matrix.ByRotation. List length should be 4"

# Case 6 - Subtract
# test 1
mat_sub = Matrix.Subtract(mat_scal,mat_rot)
assert isinstance(mat_sub, list), "Matrix.Subtract. list"
assert len(mat_sub) == 4, "Matrix.ByRotation. List length should be 4"
assert len(mat_sub[0]) == 4, "Matrix.ByRotation. List length should be 4"
# test 2
mat_sub1 = Matrix.Subtract(mat_scal1,mat_scal)
assert isinstance(mat_sub1, list), "Matrix.Subtract. list"
assert len(mat_sub1) == 4, "Matrix.ByRotation. List length should be 4"
assert len(mat_sub1[0]) == 4, "Matrix.ByRotation. List length should be 4"
# test 3
mat_sub2 = Matrix.Subtract(mat_rot2,mat_scal)
assert isinstance(mat_sub2, list), "Matrix.Subtract. list"
assert len(mat_sub2) == 4, "Matrix.ByRotation. List length should be 4"
assert len(mat_sub2[0]) == 4, "Matrix.ByRotation. List length should be 4"

# Case 7 - Transpose
# test 1
mat_transp = Matrix.Transpose(mat_sub)
assert isinstance(mat_transp, list), "Matrix.Transpose. list"
assert len(mat_transp) == 4, "Matrix.ByRotation. List length should be 4"
assert len(mat_transp[0]) == 4, "Matrix.ByRotation. List length should be 4"
# test 2
mat_transp1 = Matrix.Transpose(mat_rot)
assert isinstance(mat_transp1, list), "Matrix.Transpose. list"
assert len(mat_transp1) == 4, "Matrix.ByRotation. List length should be 4"
assert len(mat_transp1[0]) == 4, "Matrix.ByRotation. List length should be 4"
# test 3
mat_transp2 = Matrix.Transpose(mat_scal)
assert isinstance(mat_transp2, list), "Matrix.Transpose. list"
assert len(mat_transp2) == 4, "Matrix.ByRotation. List length should be 4"
assert len(mat_transp2[0]) == 4, "Matrix.ByRotation. List length should be 4"