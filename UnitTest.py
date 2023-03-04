# Vertex Unit testing
import topologicpy
import topologic
from topologicpy.Vertex import Vertex
from topologicpy.CellComplex import CellComplex

v = Vertex.ByCoordinates(10, 20, 30)
assert isinstance(v, topologic.Vertex), "Vertex.ByCoordinates. Should be topologic.Vertex"

v = Vertex.ByCoordinates(10.123, 20.456, 30.789)
x = Vertex.Coordinates(v, outputType="x", mantissa=3)
assert x == [10.123], "Vertex.Coordinates. Should be 10.123"
y = Vertex.Coordinates(v, outputType="y", mantissa=3)
assert y == [20.456], "Vertex.Coordinates. Should be 20.456"
z = Vertex.Coordinates(v, outputType="z", mantissa=3)
assert z == [30.789], "Vertex.Coordinates. Should be 30.789"
xyz = Vertex.Coordinates(v, outputType="xyz", mantissa=3)
assert xyz == [10.123, 20.456, 30.789], "Vertex.Coordinates. Should be 30.789"

v1 = Vertex.ByCoordinates(10, 20, 30)
v2 = Vertex.ByCoordinates(20, 30, 40)
d = Vertex.Distance(v1, v2, mantissa=3)
assert d == 17.321, "Vertex.Distance. Should be 17.321"

origin = Vertex.ByCoordinates(0,0,0)
cc = CellComplex.Prism(origin=origin, width=10, length=10, height=10, uSides=2, vSides=2, wSides=2,
                         direction=[0,0,1], originLocation="Center")
assert isinstance(cc, topologic.CellComplex), "Vertex.EnclosingCell. Should be topologic.CellComplex"
EnclosingCell(vertex, topology, exclusive=True, tolerance=0.0001)