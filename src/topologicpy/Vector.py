# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3.0 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence


class Vector(list):
    """A lightweight collection of vector utility methods used by TopologicPy."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_vector(value) -> bool:
        return isinstance(value, (list, tuple, Vector)) and len(value) > 0

    @staticmethod
    def _to_float_vector(value, length: Optional[int] = None):
        if not Vector._is_vector(value):
            return None
        try:
            result = [float(x) for x in value]
        except Exception:
            return None
        if length is not None and len(result) != int(length):
            return None
        return result

    @staticmethod
    def _to_3d(value):
        v = Vector._to_float_vector(value)
        if v is None:
            return None
        if len(v) == 2:
            return [v[0], v[1], 0.0]
        if len(v) == 3:
            return v
        return None

    @staticmethod
    def _round(value, mantissa: int = 6):
        if mantissa is None or mantissa < 0:
            return float(value)
        return round(float(value), int(mantissa))

    @staticmethod
    def _round_vector(vector, mantissa: int = 6):
        if vector is None:
            return None
        return [Vector._round(x, mantissa) for x in vector]

    @staticmethod
    def _dot_raw(vectorA, vectorB) -> float:
        return sum(a * b for a, b in zip(vectorA, vectorB))

    @staticmethod
    def _cross_raw(vectorA, vectorB):
        return [
            vectorA[1] * vectorB[2] - vectorA[2] * vectorB[1],
            vectorA[2] * vectorB[0] - vectorA[0] * vectorB[2],
            vectorA[0] * vectorB[1] - vectorA[1] * vectorB[0],
        ]

    @staticmethod
    def _magnitude_raw(vector) -> float:
        return math.sqrt(sum(x * x for x in vector))

    @staticmethod
    def _perpendicular_unit(vector, tolerance: float = 0.0001):
        v = Vector._to_3d(vector)
        if v is None:
            return None
        # Pick the global axis least aligned with v to avoid a near-zero cross product.
        axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        axis = min(axes, key=lambda a: abs(Vector._dot_raw(Vector.Normalize(v) or [0, 0, 0], a)))
        p = Vector._cross_raw(v, axis)
        if Vector._magnitude_raw(p) <= tolerance:
            p = Vector._cross_raw(v, [0.0, 0.0, 1.0])
        return Vector.Normalize(p)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    @staticmethod
    def Add(vectorA, vectorB):
        a = Vector._to_float_vector(vectorA)
        b = Vector._to_float_vector(vectorB)
        if a is None or b is None:
            return None
        return [x + y for x, y in zip(a, b)]

    @staticmethod
    def Subtract(vectorA, vectorB):
        a = Vector._to_float_vector(vectorA)
        b = Vector._to_float_vector(vectorB)
        if a is None or b is None:
            return None
        return [x - y for x, y in zip(a, b)]

    @staticmethod
    def Multiply(vector, magnitude, tolerance=0.0001):
        v = Vector._to_float_vector(vector)
        if v is None:
            return None
        try:
            m = float(magnitude)
        except Exception:
            return None
        if abs(m) <= float(tolerance):
            return [0.0] * len(v)
        return [x * m for x in v]

    @staticmethod
    def Sum(vectors: list):
        if not isinstance(vectors, list):
            print("Vector.Sum - Error: The input vectors parameter is not a valid list. Returning None.")
            return None
        valid = [Vector._to_float_vector(v) for v in vectors]
        valid = [v for v in valid if v is not None]
        if len(valid) < 1:
            print("Vector.Sum - Error: The input vectors parameter does not contain any valid vectors. Returning None.")
            return None
        dimensions = len(valid[0])
        valid = [v for v in valid if len(v) == dimensions]
        if len(valid) < 1:
            return None
        result = [0.0] * dimensions
        for vector in valid:
            for i in range(dimensions):
                result[i] += vector[i]
        return result

    @staticmethod
    def Average(vectors: list):
        if not isinstance(vectors, list):
            print("Vector.Average - Error: The input vectors parameter is not a valid list. Returning None.")
            return None
        total = Vector.Sum(vectors)
        if total is None:
            print("Vector.Average - Error: The input vectors parameter does not contain any valid vectors. Returning None.")
            return None
        valid_count = len([v for v in vectors if Vector._to_float_vector(v) is not None and len(Vector._to_float_vector(v)) == len(total)])
        if valid_count < 1:
            return None
        return [x / valid_count for x in total]

    @staticmethod
    def Reverse(vector):
        v = Vector._to_float_vector(vector)
        if v is None:
            return None
        return [-x for x in v]

    # ------------------------------------------------------------------
    # Construction and coordinate conversion
    # ------------------------------------------------------------------
    @staticmethod
    def ByCoordinates(x, y, z):
        return [x, y, z]

    @staticmethod
    def Coordinates(vector, outputType="xyz", mantissa: int = 6, silent: bool = False):
        v = Vector._to_3d(vector)
        if v is None:
            if not silent:
                print("Vector.Coordinates - Error: The input vector parameter is not a valid vector. Returning None.")
            return None
        x, y, z = Vector._round_vector(v, mantissa)
        matrix = [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]
        out = str(outputType or "xyz").lower()
        if out == "matrix":
            return matrix
        result = []
        for axis in out:
            if axis == "x":
                result.append(x)
            elif axis == "y":
                result.append(y)
            elif axis == "z":
                result.append(z)
        return result

    @staticmethod
    def ByAzimuthAltitude(azimuth: float, altitude: float, north: float = 0, reverse: bool = False, mantissa: int = 6, tolerance: float = 0.0001):
        """Returns a unit vector from azimuth/altitude angles without requiring TopologicCore."""
        try:
            az = math.radians(float(azimuth) + float(north))
            alt = math.radians(float(altitude))
        except Exception:
            return None
        horizontal = math.cos(alt)
        # 0 azimuth is +Y/North; 90 is +X/East.
        x = horizontal * math.sin(az)
        y = horizontal * math.cos(az)
        z = math.sin(alt)
        result = [x, y, z]
        if reverse:
            result = Vector.Reverse(result)
        return Vector._round_vector(result, mantissa)

    @staticmethod
    def ByVertices(*vertices, normalize: bool = True, mantissa: int = 6, silent: bool = False):
        try:
            from topologicpy.Vertex import Vertex
            from topologicpy.Topology import Topology
            from topologicpy.Helper import Helper
        except Exception:
            if not silent:
                print("Vector.ByVertices - Error: Required TopologicPy modules are unavailable. Returning None.")
            return None

        if len(vertices) == 0:
            if not silent:
                print("Vector.ByVertices - Error: The input vertices parameter is an empty list. Returning None.")
            return None
        if len(vertices) == 1 and isinstance(vertices[0], list):
            vertex_list = vertices[0]
        else:
            try:
                vertex_list = Helper.Flatten(list(vertices))
            except Exception:
                vertex_list = list(vertices)
        vertex_list = [x for x in vertex_list if Topology.IsInstance(x, "Vertex")]
        if len(vertex_list) < 2:
            if not silent:
                print("Vector.ByVertices - Error: The input parameters do not contain a minimum of two valid vertices. Returning None.")
            return None
        v1 = vertex_list[0]
        v2 = vertex_list[-1]
        result = [
            Vertex.X(v2, mantissa=mantissa) - Vertex.X(v1, mantissa=mantissa),
            Vertex.Y(v2, mantissa=mantissa) - Vertex.Y(v1, mantissa=mantissa),
            Vertex.Z(v2, mantissa=mantissa) - Vertex.Z(v1, mantissa=mantissa),
        ]
        return Vector.Normalize(result) if normalize else result

    # ------------------------------------------------------------------
    # Magnitudes and normalization
    # ------------------------------------------------------------------
    @staticmethod
    def Quadrance(vector, mantissa: int = 6):
        v = Vector._to_float_vector(vector)
        if v is None or len(v) == 0:
            return 0.0
        return Vector._round(sum(x * x for x in v), mantissa)

    @staticmethod
    def Magnitude(vector, mantissa: int = 6):
        v = Vector._to_float_vector(vector)
        if v is None:
            return None
        return Vector._round(Vector._magnitude_raw(v), mantissa)

    @staticmethod
    def Length(vector, mantissa: int = 6):
        return Vector.Magnitude(vector, mantissa=mantissa)

    @staticmethod
    def Normalize(vector):
        v = Vector._to_float_vector(vector)
        if v is None:
            return None
        mag = Vector._magnitude_raw(v)
        if mag <= 0.0:
            return None
        return [x / mag for x in v]

    @staticmethod
    def SetMagnitude(vector: list, magnitude: float) -> list:
        normalized = Vector.Normalize(vector)
        if normalized is None:
            return None
        return Vector.Multiply(vector=normalized, magnitude=magnitude)

    # ------------------------------------------------------------------
    # Products and angular measures
    # ------------------------------------------------------------------
    @staticmethod
    def Dot(vectorA, vectorB, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        a = Vector._to_float_vector(vectorA)
        b = Vector._to_float_vector(vectorB)
        if a is None:
            if not silent:
                print("Vector.Dot - Error: The input vectorA parameter is not a valid vector. Returning None.")
            return None
        if b is None:
            if not silent:
                print("Vector.Dot - Error: The input vectorB parameter is not a valid vector. Returning None.")
            return None
        if Vector._magnitude_raw(a) <= tolerance:
            if not silent:
                print("Vector.Dot - Error: The magnitude of the input vectorA parameter is less than the input tolerance parameter. Returning None.")
            return None
        if Vector._magnitude_raw(b) <= tolerance:
            if not silent:
                print("Vector.Dot - Error: The magnitude of the input vectorB parameter is less than the input tolerance parameter. Returning None.")
            return None
        return Vector._round(Vector._dot_raw(a, b), mantissa)

    @staticmethod
    def Cross(vectorA, vectorB, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        a = Vector._to_3d(vectorA)
        b = Vector._to_3d(vectorB)
        if a is None:
            if not silent:
                print("Vector.Cross - Error: The input vectorA parameter is not a valid vector. Returning None.")
            return None
        if b is None:
            if not silent:
                print("Vector.Cross - Error: The input vectorB parameter is not a valid vector. Returning None.")
            return None
        if Vector._magnitude_raw(a) <= tolerance:
            if not silent:
                print("Vector.Cross - Error: The magnitude of the input vectorA parameter is less than the input tolerance parameter. Returning None.")
            return None
        if Vector._magnitude_raw(b) <= tolerance:
            if not silent:
                print("Vector.Cross - Error: The magnitude of the input vectorB parameter is less than the input tolerance parameter. Returning None.")
            return None
        result = Vector._cross_raw(a, b)
        if Vector._magnitude_raw(result) <= tolerance:
            return [0, 0, 0]
        return Vector._round_vector(result, mantissa)

    @staticmethod
    def Spread(vectorA, vectorB, mantissa: int = 6, bracket: bool = False) -> float:
        a = Vector._to_3d(vectorA)
        b = Vector._to_3d(vectorB)
        if a is None:
            print("Vector.Spread - Error: The input vectorA is not a valid vector. Returning None.")
            return None
        if b is None:
            print("Vector.Spread - Error: The input vectorB is not a valid vector. Returning None.")
            return None
        uu = Vector._dot_raw(a, a)
        vv = Vector._dot_raw(b, b)
        if uu == 0.0 or vv == 0.0:
            return 0.0
        dot = Vector._dot_raw(a, b)
        spread = 1.0 - (dot * dot) / (uu * vv)
        spread = max(0.0, min(1.0, spread))
        if bracket:
            spread = min(spread, 1.0 - spread)
        return Vector._round(spread, mantissa)

    @staticmethod
    def Angle(vectorA, vectorB, mantissa: int = 6, bracket: bool = False):
        a = Vector._to_3d(vectorA)
        b = Vector._to_3d(vectorB)
        if a is None:
            print("Vector.Angle - Error: The input vectorA is not a valid vector. Returning None.")
            return None
        if b is None:
            print("Vector.Angle - Error: The input vectorB is not a valid vector. Returning None.")
            return None
        if Vector._dot_raw(a, a) == 0.0 or Vector._dot_raw(b, b) == 0.0:
            return 0.0
        spread = Vector.Spread(a, b, mantissa=15, bracket=False)
        if spread is None:
            return None
        spread = max(0.0, min(1.0, spread))
        acute = math.degrees(math.asin(math.sqrt(spread)))
        if bracket:
            angle = acute
        else:
            angle = 180.0 - acute if Vector._dot_raw(a, b) < 0.0 else acute
        return Vector._round(angle, mantissa)

    @staticmethod
    def CompassAngle(vectorA, vectorB, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        a = Vector._to_3d(vectorA)
        b = Vector._to_3d(vectorB)
        if a is None:
            if not silent:
                print("Vector.CompassAngle - Error: The input vectorA parameter is not a valid vector. Returning None.")
            return None
        if b is None:
            if not silent:
                print("Vector.CompassAngle - Error: The input vectorB parameter is not a valid vector. Returning None.")
            return None
        if abs(a[0]) <= tolerance and abs(a[1]) <= tolerance:
            if not silent:
                print("Vector.CompassAngle - Error: The input vectorA parameter is vertical in the Z Axis. Returning None.")
            return None
        if abs(b[0]) <= tolerance and abs(b[1]) <= tolerance:
            if not silent:
                print("Vector.CompassAngle - Error: The input vectorB parameter is vertical in the Z Axis. Returning None.")
            return None
        # Match the original TopologicPy convention: angle is measured clockwise
        # in the horizontal plane, with 0 along +Y/North and 90 along +X/East.
        ang1 = math.atan2(a[1], a[0])
        ang2 = math.atan2(b[1], b[0])
        return Vector._round(math.degrees((ang1 - ang2) % (2.0 * math.pi)), mantissa)

    @staticmethod
    def AzimuthAltitude(vector, mantissa: int = 6):
        v = Vector._to_3d(vector)
        if v is None:
            return None
        x, y, z = v
        if x == 0.0 and y == 0.0:
            if z > 0.0:
                return {"azimuth": 0, "altitude": 90}
            if z < 0.0:
                return {"azimuth": 0, "altitude": -90}
            return None
        azimuth = math.degrees(math.atan2(x, y)) % 360.0
        xy_distance = math.sqrt(x * x + y * y)
        altitude = math.degrees(math.atan2(z, xy_distance))
        return {"azimuth": Vector._round(azimuth, mantissa), "altitude": Vector._round(altitude, mantissa)}

    @staticmethod
    def Bisect(vectorA, vectorB):
        a = Vector.Normalize(Vector._to_3d(vectorA))
        b = Vector.Normalize(Vector._to_3d(vectorB))
        if a is None or b is None:
            return None
        dot = max(-1.0, min(1.0, Vector._dot_raw(a, b)))
        if abs(dot - 1.0) <= 1e-9:
            return a
        if abs(dot + 1.0) <= 1e-9:
            print("Vector.Bisect - Warning: The two vectors are anti-parallel and thus the bisecting vector is not uniquely defined.")
            return Vector._perpendicular_unit(a)
        return Vector.Normalize([a[i] + b[i] for i in range(3)])

    @staticmethod
    def IsSame(vectorA, vectorB, tolerance=0.0001):
        a = Vector._to_float_vector(vectorA)
        b = Vector._to_float_vector(vectorB)
        if a is None or b is None or len(a) != len(b):
            return False
        return all(abs(x - y) <= tolerance for x, y in zip(a, b))

    @staticmethod
    def IsParallel(vectorA, vectorB, tolerance=0.0001):
        a = Vector.Normalize(Vector._to_3d(vectorA))
        b = Vector.Normalize(Vector._to_3d(vectorB))
        if a is None or b is None:
            return False
        return abs(Vector._dot_raw(a, b) - 1.0) <= tolerance

    @staticmethod
    def IsAntiParallel(vectorA, vectorB):
        a = Vector.Normalize(Vector._to_3d(vectorA))
        b = Vector.Normalize(Vector._to_3d(vectorB))
        if a is None or b is None:
            return False
        return abs(Vector._dot_raw(a, b) + 1.0) <= 1e-6

    @staticmethod
    def IsCollinear(vectorA, vectorB, tolerance=0.0001):
        a = Vector.Normalize(Vector._to_3d(vectorA))
        b = Vector.Normalize(Vector._to_3d(vectorB))
        if a is None or b is None:
            return False
        return abs(abs(Vector._dot_raw(a, b)) - 1.0) <= tolerance

    # ------------------------------------------------------------------
    # Directions
    # ------------------------------------------------------------------
    @staticmethod
    def CompassDirection(vector, tolerance: float = 0.0001, silent: bool = False):
        v = Vector._to_3d(vector)
        if v is None:
            if not silent:
                print("Vector.CompassDirection - Error: The input vector parameter is not a valid vector. Returning None.")
            return None
        x, y, z = v
        mag = Vector._magnitude_raw(v)
        if mag <= tolerance:
            return "Origin"
        x, y, z = x / mag, y / mag, z / mag
        x = 0.0 if abs(x) <= tolerance else x
        y = 0.0 if abs(y) <= tolerance else y
        z = 0.0 if abs(z) <= tolerance else z
        if x == 0.0 and y > 0.0:
            horizontal = "North"
        elif x == 0.0 and y < 0.0:
            horizontal = "South"
        elif y == 0.0 and x > 0.0:
            horizontal = "East"
        elif y == 0.0 and x < 0.0:
            horizontal = "West"
        elif x > 0.0 and y > 0.0:
            horizontal = "Northeast"
        elif x < 0.0 and y > 0.0:
            horizontal = "Northwest"
        elif x < 0.0 and y < 0.0:
            horizontal = "Southwest"
        elif x > 0.0 and y < 0.0:
            horizontal = "Southeast"
        else:
            horizontal = ""
        if z > 0.0:
            return f"Up_{horizontal}" if horizontal else "Up"
        if z < 0.0:
            return f"Down_{horizontal}" if horizontal else "Down"
        return horizontal

    @staticmethod
    def CompassDirections():
        return [
            "Origin", "Up", "Down",
            "North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest",
            "Up_North", "Up_Northeast", "Up_East", "Up_Southeast", "Up_South", "Up_Southwest", "Up_West", "Up_Northwest",
            "Down_North", "Down_Northeast", "Down_East", "Down_Southeast", "Down_South", "Down_Southwest", "Down_West", "Down_Northwest",
        ]

    @staticmethod
    def North(): return [0, 1, 0]
    @staticmethod
    def South(): return [0, -1, 0]
    @staticmethod
    def East(): return [1, 0, 0]
    @staticmethod
    def West(): return [-1, 0, 0]
    @staticmethod
    def Up(): return [0, 0, 1]
    @staticmethod
    def Down(): return [0, 0, -1]
    @staticmethod
    def NorthEast(): return [1, 1, 0]
    @staticmethod
    def NorthWest(): return [-1, 1, 0]
    @staticmethod
    def SouthEast(): return [1, -1, 0]
    @staticmethod
    def SouthWest(): return [-1, -1, 0]
    @staticmethod
    def XAxis(): return [1, 0, 0]
    @staticmethod
    def YAxis(): return [0, 1, 0]
    @staticmethod
    def ZAxis(): return [0, 0, 1]

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------
    @staticmethod
    def TransformationMatrix(vectorA, vectorB):
        """Returns a 4x4 rotation matrix aligning vectorA with vectorB."""
        a = Vector.Normalize(Vector._to_3d(vectorA))
        b = Vector.Normalize(Vector._to_3d(vectorB))
        if a is None or b is None:
            return None
        dot = max(-1.0, min(1.0, Vector._dot_raw(a, b)))
        if abs(dot - 1.0) <= 1e-12:
            return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        if abs(dot + 1.0) <= 1e-12:
            axis = Vector._perpendicular_unit(a)
            angle = math.pi
        else:
            axis = Vector.Normalize(Vector._cross_raw(a, b))
            angle = math.acos(dot)
        if axis is None:
            return None
        x, y, z = axis
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1.0 - c
        return [
            [t*x*x + c,     t*x*y - s*z,   t*x*z + s*y,   0.0],
            [t*x*y + s*z,   t*y*y + c,     t*y*z - s*x,   0.0],
            [t*x*z - s*y,   t*y*z + s*x,   t*z*z + c,     0.0],
            [0.0,           0.0,           0.0,           1.0],
        ]


__all__ = ["Vector"]
