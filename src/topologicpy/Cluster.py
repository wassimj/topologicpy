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

from topologicpy.Core import Core
from typing import Callable


class Cluster:
    """Utility methods for creating, querying, and analysing Topologic Clusters."""

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _Flatten(values):
        """Flattens Python list/tuple containers without flattening Topologies."""
        result = []

        def walk(value):
            if isinstance(value, (list, tuple)):
                for item in value:
                    walk(item)
            else:
                result.append(value)

        walk(values)
        return result

    @staticmethod
    def _Query(cluster, methodName: str, silent: bool = False):
        """Executes a backend collection query using the canonical output-list contract."""
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
                print(f"Cluster.{methodName} - Error: The input cluster parameter is not a valid Cluster. Returning None.")
            return None

        output = []
        try:
            result = Core.InstanceCall(cluster, methodName, None, output)
        except Exception:
            try:
                result = Core.InstanceCall(cluster, methodName)
            except Exception:
                if not silent:
                    print(f"Cluster.{methodName} - Error: Could not query the backend. Returning None.")
                return None

        if output:
            return [item for item in output if Topology.IsInstance(item, "Topology")]
        if isinstance(result, list):
            return [item for item in result if Topology.IsInstance(item, "Topology")]
        if result in (0, None):
            return []

        if not silent:
            print(f"Cluster.{methodName} - Error: The backend returned an invalid result. Returning None.")
        return None

    @staticmethod
    def _IsSame(topologyA, topologyB) -> bool:
        """Backend-neutral topology identity comparison."""
        from topologicpy.Topology import Topology

        try:
            return bool(Topology.IsSame(topologyA, topologyB, silent=True))
        except TypeError:
            try:
                return bool(Topology.IsSame(topologyA, topologyB))
            except Exception:
                return topologyA is topologyB
        except Exception:
            return topologyA is topologyB

    @staticmethod
    def _ReconstructTopologies(cluster, silent: bool = False):
        """
        Reconstructs top-level constituents when a backend cannot expose a direct
        Cluster.Topologies query. This is a hierarchy/identity operation only; no
        geometric Boolean operations are used.
        """
        from topologicpy.Topology import Topology

        type_methods = [
            ("CellComplex", "CellComplexes", "cellcomplex"),
            ("Cell", "Cells", "cell"),
            ("Shell", "Shells", "shell"),
            ("Face", "Faces", "face"),
            ("Wire", "Wires", "wire"),
            ("Edge", "Edges", "edge"),
            ("Vertex", "Vertices", "vertex"),
        ]

        selected = []
        for type_name, method_name, sub_type in type_methods:
            candidates = Cluster._Query(cluster, method_name, silent=True)
            if candidates is None:
                continue

            for candidate in candidates:
                if not Topology.IsInstance(candidate, type_name):
                    continue

                contained = False
                for parent in selected:
                    try:
                        descendants = Topology.SubTopologies(
                            parent,
                            subTopologyType=sub_type,
                            silent=True,
                        ) or []
                    except TypeError:
                        try:
                            descendants = Topology.SubTopologies(
                                parent,
                                subTopologyType=sub_type,
                            ) or []
                        except Exception:
                            descendants = []
                    except Exception:
                        descendants = []

                    if any(Cluster._IsSame(candidate, descendant) for descendant in descendants):
                        contained = True
                        break

                if not contained and not any(Cluster._IsSame(candidate, item) for item in selected):
                    selected.append(candidate)

        if selected:
            return selected

        if not silent:
            print("Cluster.Topologies - Error: Could not determine the direct constituents of the Cluster. Returning None.")
        return None

    @staticmethod
    def _DirectOfType(cluster, typeName: str, tolerance: float = 0.0001, silent: bool = False):
        """Returns direct Cluster constituents of the requested Topologic type."""
        from topologicpy.Topology import Topology

        topologies = Cluster.Topologies(cluster, tolerance=tolerance, silent=silent)
        if topologies is None:
            return None
        return [topology for topology in topologies if Topology.IsInstance(topology, typeName)]

    # -------------------------------------------------------------------------
    # Constructors and grouping
    # -------------------------------------------------------------------------

    @staticmethod
    def ByFormula(
        formula: str,
        xRange=None,
        yRange=None,
        xString: str = "X",
        yString: str = "Y",
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Creates a Cluster of Vertices by evaluating a mathematical expression.

        If only ``xRange`` is supplied, X is the independent variable and the
        expression computes Y. If only ``yRange`` is supplied, Y is the
        independent variable and the expression computes X. If both ranges are
        supplied, the expression computes Z over the Cartesian product of X and Y.

        The expression is parsed and restricted to arithmetic, the independent
        variables, common mathematical functions, and mathematical constants. It
        does not execute arbitrary Python code.

        Parameters
        ----------
        formula : str
            The mathematical expression to evaluate, for example ``"X**2"`` or
            ``"cos(X) + sin(Y)"``.
        xRange : tuple or list, optional
            ``(start, end, step)`` for X. The endpoint is included. Default is None.
        yRange : tuple or list, optional
            ``(start, end, step)`` for Y. The endpoint is included. Default is None.
        xString : str, optional
            The identifier used for the X variable. It must not be lowercase.
            Default is ``"X"``.
        yString : str, optional
            The identifier used for the Y variable. It must not be lowercase.
            Default is ``"Y"``.
        tolerance : float, optional
            Numerical tolerance used when generating inclusive ranges and creating
            Vertices. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cluster or None
            The Cluster of generated Vertices, or None if the inputs or expression
            are invalid.
        """
        import ast
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            tolerance = 0.0001
        if tolerance <= 0:
            tolerance = 0.0001

        if not isinstance(formula, str) or not formula.strip():
            if not silent:
                print("Cluster.ByFormula - Error: The input formula parameter is not a valid string. Returning None.")
            return None
        if xRange is None and yRange is None:
            if not silent:
                print("Cluster.ByFormula - Error: xRange and yRange cannot both be None. Returning None.")
            return None
        if not isinstance(xString, str) or not xString.isidentifier() or xString.islower():
            if not silent:
                print("Cluster.ByFormula - Error: xString must be a valid non-lowercase Python identifier. Returning None.")
            return None
        if not isinstance(yString, str) or not yString.isidentifier() or yString.islower():
            if not silent:
                print("Cluster.ByFormula - Error: yString must be a valid non-lowercase Python identifier. Returning None.")
            return None
        if xString == yString:
            if not silent:
                print("Cluster.ByFormula - Error: xString and yString must be different identifiers. Returning None.")
            return None

        def build_range(value, name):
            if value is None:
                return []
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise ValueError(f"{name} must be a (start, end, step) tuple or list")
            start, end, step = map(float, value)
            if not all(math.isfinite(v) for v in (start, end, step)):
                raise ValueError(f"{name} contains a non-finite value")
            if abs(step) <= tolerance:
                raise ValueError(f"{name} step cannot be zero")
            delta = end - start
            if abs(delta) <= tolerance:
                return [start]
            if delta * step < 0:
                raise ValueError(f"{name} step has the wrong sign")

            values = []
            current = start
            max_count = 1_000_000
            for _ in range(max_count):
                if step > 0:
                    if current >= end - tolerance:
                        break
                else:
                    if current <= end + tolerance:
                        break
                values.append(current)
                current += step
            else:
                raise ValueError(f"{name} generated too many values")

            if not values or abs(values[-1] - end) > tolerance:
                values.append(end)
            else:
                values[-1] = end
            return values

        try:
            x_values = build_range(xRange, "xRange")
            y_values = build_range(yRange, "yRange")
        except Exception as error:
            if not silent:
                print(f"Cluster.ByFormula - Error: {error}. Returning None.")
            return None

        allowed_functions = {
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "atan2": math.atan2,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            "exp": math.exp,
            "log": math.log,
            "log10": math.log10,
            "floor": math.floor,
            "ceil": math.ceil,
            "degrees": math.degrees,
            "radians": math.radians,
            "pow": pow,
        }
        allowed_constants = {"pi": math.pi, "e": math.e, "tau": math.tau}
        allowed_names = set(allowed_functions) | set(allowed_constants) | {xString, yString}
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.UAdd,
            ast.USub,
        )

        try:
            tree = ast.parse(formula, mode="eval")
            for node in ast.walk(tree):
                if not isinstance(node, allowed_nodes):
                    raise ValueError(f"Unsupported expression element: {node.__class__.__name__}")
                if isinstance(node, ast.Name) and node.id not in allowed_names:
                    raise ValueError(f"Unsupported name: {node.id}")
                if isinstance(node, ast.Call):
                    if not isinstance(node.func, ast.Name) or node.func.id not in allowed_functions:
                        raise ValueError("Only supported mathematical functions can be called")
                    if node.keywords:
                        raise ValueError("Keyword arguments are not supported")
            code = compile(tree, "<Cluster.ByFormula>", "eval")
        except Exception as error:
            if not silent:
                print(f"Cluster.ByFormula - Error: Invalid formula ({error}). Returning None.")
            return None

        def evaluate(x=None, y=None):
            env = dict(allowed_functions)
            env.update(allowed_constants)
            if x is not None:
                env[xString] = x
            if y is not None:
                env[yString] = y
            value = eval(code, {"__builtins__": {}}, env)
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("Formula produced a non-finite value")
            return value

        vertices = []
        try:
            if x_values and y_values:
                for x in x_values:
                    for y in y_values:
                        z = evaluate(x=x, y=y)
                        vertex = Vertex.ByCoordinates(x, y, z)
                        if Topology.IsInstance(vertex, "Vertex"):
                            vertices.append(vertex)
            elif x_values:
                for x in x_values:
                    y = evaluate(x=x)
                    vertex = Vertex.ByCoordinates(x, y, 0.0)
                    if Topology.IsInstance(vertex, "Vertex"):
                        vertices.append(vertex)
            else:
                for y in y_values:
                    x = evaluate(y=y)
                    vertex = Vertex.ByCoordinates(x, y, 0.0)
                    if Topology.IsInstance(vertex, "Vertex"):
                        vertices.append(vertex)
        except Exception as error:
            if not silent:
                print(f"Cluster.ByFormula - Error: Could not evaluate the formula ({error}). Returning None.")
            return None

        if not vertices:
            if not silent:
                print("Cluster.ByFormula - Error: No valid Vertices were created. Returning None.")
            return None
        return Cluster.ByTopologies(vertices, silent=silent)

    @staticmethod
    def ByFunction(
        topologies: list,
        function: Callable,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Groups input topologies according to the value returned by a function.

        Numeric values are grouped using ``tolerance``. Boolean, string, None,
        and other comparable values are grouped by exact equality. Each group is
        returned as a Topologic Cluster.

        Parameters
        ----------
        topologies : list
            The input Topologies to group.
        function : callable
            A function called as ``function(topology, mantissa=..., tolerance=...)``.
        mantissa : int, optional
            Decimal precision applied to numeric function results. Default is 6.
        tolerance : float, optional
            Maximum numerical difference for two numeric values to belong to the
            same group. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list[topologic_core.Cluster] or None
            One Cluster per equivalence class, or None for invalid inputs.
        """
        import math
        from numbers import Number
        from topologicpy.Topology import Topology

        if not isinstance(topologies, list):
            if not silent:
                print("Cluster.ByFunction - Error: The input topologies parameter is not a valid list. Returning None.")
            return None
        if not callable(function):
            if not silent:
                print("Cluster.ByFunction - Error: The input function parameter is not callable. Returning None.")
            return None
        if len(topologies) == 0:
            return []

        try:
            mantissa = max(0, int(mantissa))
            tolerance = abs(float(tolerance))
        except Exception:
            if not silent:
                print("Cluster.ByFunction - Error: mantissa or tolerance is invalid. Returning None.")
            return None

        valid_topologies = [t for t in topologies if Topology.IsInstance(t, "Topology")]
        if not valid_topologies:
            if not silent:
                print("Cluster.ByFunction - Error: No valid Topologies were supplied. Returning None.")
            return None

        def is_numeric(value):
            return isinstance(value, Number) and not isinstance(value, bool)

        numeric_groups = []  # [representative, members]
        exact_groups = []    # [representative, members]

        for index, topology in enumerate(valid_topologies):
            try:
                value = function(topology, mantissa=mantissa, tolerance=tolerance)
            except Exception as error:
                if not silent:
                    print(f"Cluster.ByFunction - Warning: Function evaluation failed at index {index}; skipping topology ({error}).")
                continue

            if is_numeric(value):
                try:
                    value = round(float(value), mantissa)
                    if not math.isfinite(value):
                        raise ValueError
                except Exception:
                    if not silent:
                        print(f"Cluster.ByFunction - Warning: Non-finite numeric value at index {index}; skipping topology.")
                    continue

                placed = False
                for group in numeric_groups:
                    if abs(value - group[0]) <= tolerance:
                        group[1].append(topology)
                        placed = True
                        break
                if not placed:
                    numeric_groups.append([value, [topology]])
            else:
                placed = False
                for group in exact_groups:
                    try:
                        equal = value == group[0]
                        if not isinstance(equal, bool):
                            equal = bool(equal)
                    except Exception:
                        equal = repr(value) == repr(group[0])
                    if equal:
                        group[1].append(topology)
                        placed = True
                        break
                if not placed:
                    exact_groups.append([value, [topology]])

        groups = numeric_groups + exact_groups
        clusters = []
        for _, members in groups:
            cluster = Cluster.ByTopologies(members, silent=True)
            if Topology.IsInstance(cluster, "Cluster"):
                clusters.append(cluster)

        return clusters

    @staticmethod
    def ByTopologies(*topologies, transferDictionaries: bool = False, silent: bool = False):
        """
        Creates a Cluster from one or more Topologies.

        Python list/tuple containers may be nested arbitrarily. Unlike the legacy
        implementation, a single valid input Topology still produces a one-member
        Cluster; the constructor therefore has a stable return type.

        Parameters
        ----------
        *topologies : topologic_core.Topology or list
            One or more Topologies, optionally nested in lists or tuples.
        transferDictionaries : bool, optional
            If True, dictionaries from all valid input Topologies are merged and
            assigned to the resulting Cluster. Default is False.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cluster or None
            The created Cluster, or None if no valid Topologies are supplied.
        """
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        flat = Cluster._Flatten(list(topologies))
        topology_list = [item for item in flat if Topology.IsInstance(item, "Topology")]
        if not topology_list:
            if not silent:
                print("Cluster.ByTopologies - Error: The input parameters do not contain any valid Topologies. Returning None.")
            return None

        try:
            cluster = Core.Cluster.ByTopologies(topology_list, False)
        except Exception:
            try:
                cluster = Core.Cluster.ByTopologies(topology_list, transferDictionaries=False)
            except Exception:
                cluster = None

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
                print("Cluster.ByTopologies - Error: The backend could not create a Cluster. Returning None.")
            return None

        if transferDictionaries:
            dictionaries = []
            for topology in topology_list:
                try:
                    dictionary = Topology.Dictionary(topology, silent=True)
                except TypeError:
                    dictionary = Topology.Dictionary(topology)
                except Exception:
                    dictionary = None
                try:
                    keys = Dictionary.Keys(dictionary)
                except Exception:
                    keys = None
                if isinstance(keys, list) and keys:
                    dictionaries.append(dictionary)

            if dictionaries:
                try:
                    merged = Dictionary.ByMergedDictionaries(dictionaries, silent=True)
                except TypeError:
                    merged = Dictionary.ByMergedDictionaries(dictionaries)
                except Exception:
                    merged = None
                if merged is not None:
                    try:
                        cluster = Topology.SetDictionary(cluster, merged, silent=True)
                    except TypeError:
                        cluster = Topology.SetDictionary(cluster, merged)

        return cluster

    # -------------------------------------------------------------------------
    # Direct constituents and descendant accessors
    # -------------------------------------------------------------------------

    @staticmethod
    def Topologies(cluster, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns the direct, top-level constituent Topologies of a Cluster.

        This method does not return every descendant. For example, a Face inside
        a direct Shell constituent is not itself a direct constituent.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input Cluster.
        tolerance : float, optional
            Reserved for backend-neutral hierarchy fallbacks. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            Direct constituent Topologies, or None if the query fails.
        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
                print("Cluster.Topologies - Error: The input cluster parameter is not a valid Cluster. Returning None.")
            return None

        output = []
        try:
            result = Core.InstanceCall(cluster, "Topologies", None, output)
            if output:
                return [item for item in output if Topology.IsInstance(item, "Topology")]
            if isinstance(result, list):
                return [item for item in result if Topology.IsInstance(item, "Topology")]
            # A successful output-list call on an empty Cluster is still valid.
            if result == 0:
                return []
        except Exception:
            pass

        try:
            result = Core.InstanceCall(cluster, "Topologies")
            if isinstance(result, list):
                return [item for item in result if Topology.IsInstance(item, "Topology")]
        except Exception:
            pass

        return Cluster._ReconstructTopologies(cluster, silent=silent)

    @staticmethod
    def CellComplexes(cluster, silent: bool = False) -> list:
        """
        Returns all CellComplex descendants of the input Cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input Cluster.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            The CellComplex descendants, an empty list when none exist, or None if
            the input/query is invalid.
        """
        return Cluster._Query(cluster, "CellComplexes", silent=silent)

    @staticmethod
    def Cells(cluster, silent: bool = False) -> list:
        """
        Returns all Cell descendants of the input Cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input Cluster.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            The Cell descendants, an empty list when none exist, or None on failure.
        """
        return Cluster._Query(cluster, "Cells", silent=silent)

    @staticmethod
    def Edges(cluster, silent: bool = False) -> list:
        """
        Returns all Edge descendants of the input Cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input Cluster.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            The Edge descendants, an empty list when none exist, or None on failure.
        """
        return Cluster._Query(cluster, "Edges", silent=silent)

    @staticmethod
    def Faces(cluster, silent: bool = False) -> list:
        """
        Returns all Face descendants of the input Cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input Cluster.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            The Face descendants, an empty list when none exist, or None on failure.
        """
        return Cluster._Query(cluster, "Faces", silent=silent)

    @staticmethod
    def Shells(cluster, silent: bool = False) -> list:
        """
        Returns all Shell descendants of the input Cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input Cluster.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            The Shell descendants, an empty list when none exist, or None on failure.
        """
        return Cluster._Query(cluster, "Shells", silent=silent)

    @staticmethod
    def Vertices(cluster, silent: bool = False) -> list:
        """
        Returns all Vertex descendants of the input Cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input Cluster.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            The Vertex descendants, an empty list when none exist, or None on failure.
        """
        return Cluster._Query(cluster, "Vertices", silent=silent)

    @staticmethod
    def Wires(cluster, silent: bool = False) -> list:
        """
        Returns all Wire descendants of the input Cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input Cluster.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            The Wire descendants, an empty list when none exist, or None on failure.
        """
        return Cluster._Query(cluster, "Wires", silent=silent)

    # -------------------------------------------------------------------------
    # Free/direct topology queries
    # -------------------------------------------------------------------------

    @staticmethod
    def FreeCells(cluster, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns direct Cell constituents of the input Cluster.

        Direct constituents are not descendants of another direct Cluster member.
        This query is resolved from Cluster hierarchy and topology identity; it does
        not use geometric Boolean subtraction.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input Cluster.
        tolerance : float, optional
            Reserved for backend-neutral hierarchy fallbacks. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            The free Cells, an empty list when none exist, or None on failure.
        """
        return Cluster._DirectOfType(cluster, "Cell", tolerance=tolerance, silent=silent)

    @staticmethod
    def FreeShells(cluster, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns direct Shell constituents of the input Cluster.

        Direct constituents are not descendants of another direct Cluster member.
        This query is resolved from Cluster hierarchy and topology identity; it does
        not use geometric Boolean subtraction.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input Cluster.
        tolerance : float, optional
            Reserved for backend-neutral hierarchy fallbacks. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            The free Shells, an empty list when none exist, or None on failure.
        """
        return Cluster._DirectOfType(cluster, "Shell", tolerance=tolerance, silent=silent)

    @staticmethod
    def FreeFaces(cluster, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns direct Face constituents of the input Cluster.

        Direct constituents are not descendants of another direct Cluster member.
        This query is resolved from Cluster hierarchy and topology identity; it does
        not use geometric Boolean subtraction.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input Cluster.
        tolerance : float, optional
            Reserved for backend-neutral hierarchy fallbacks. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            The free Faces, an empty list when none exist, or None on failure.
        """
        return Cluster._DirectOfType(cluster, "Face", tolerance=tolerance, silent=silent)

    @staticmethod
    def FreeWires(cluster, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns direct Wire constituents of the input Cluster.

        Direct constituents are not descendants of another direct Cluster member.
        This query is resolved from Cluster hierarchy and topology identity; it does
        not use geometric Boolean subtraction.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input Cluster.
        tolerance : float, optional
            Reserved for backend-neutral hierarchy fallbacks. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            The free Wires, an empty list when none exist, or None on failure.
        """
        return Cluster._DirectOfType(cluster, "Wire", tolerance=tolerance, silent=silent)

    @staticmethod
    def FreeEdges(cluster, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns direct Edge constituents of the input Cluster.

        Direct constituents are not descendants of another direct Cluster member.
        This query is resolved from Cluster hierarchy and topology identity; it does
        not use geometric Boolean subtraction.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input Cluster.
        tolerance : float, optional
            Reserved for backend-neutral hierarchy fallbacks. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            The free Edges, an empty list when none exist, or None on failure.
        """
        return Cluster._DirectOfType(cluster, "Edge", tolerance=tolerance, silent=silent)

    @staticmethod
    def FreeVertices(cluster, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns direct Vertex constituents of the input Cluster.

        Direct constituents are not descendants of another direct Cluster member.
        This query is resolved from Cluster hierarchy and topology identity; it does
        not use geometric Boolean subtraction.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input Cluster.
        tolerance : float, optional
            Reserved for backend-neutral hierarchy fallbacks. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            The free Vertices, an empty list when none exist, or None on failure.
        """
        return Cluster._DirectOfType(cluster, "Vertex", tolerance=tolerance, silent=silent)

    @staticmethod
    def FreeTopologies(cluster, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns the direct constituent Topologies of the Cluster.

        For a heterogeneous Cluster, these are precisely the members that are not
        descendants of another direct constituent. No Boolean operations are used.
        """
        return Cluster.Topologies(cluster, tolerance=tolerance, silent=silent)

    # -------------------------------------------------------------------------
    # Boundary, type, and simplification
    # -------------------------------------------------------------------------

    @staticmethod
    def ExternalBoundary(cluster, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns a Cluster representing the exposed boundary of each free constituent.

        CellComplex and Cell constituents contribute their outer Shells. Free Shells,
        Wires, Edges, and Vertices are already boundary objects and are retained.
        Free Faces contribute their external boundary Wires.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input Cluster.
        tolerance : float, optional
            The desired tolerance. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cluster or None
            A Cluster of exposed boundary Topologies.
        """
        from topologicpy.Cell import Cell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
                print("Cluster.ExternalBoundary - Error: The input cluster parameter is not a valid Cluster. Returning None.")
            return None

        boundary = []
        for topology in Cluster.FreeTopologies(cluster, tolerance=tolerance, silent=True) or []:
            item = None
            if Topology.IsInstance(topology, "CellComplex"):
                item = CellComplex.ExternalBoundary(topology, silent=True)
            elif Topology.IsInstance(topology, "Cell"):
                item = Cell.ExternalBoundary(topology, silent=True)
            elif Topology.IsInstance(topology, "Face"):
                try:
                    item = Face.ExternalBoundary(topology, silent=True)
                except TypeError:
                    item = Face.ExternalBoundary(topology)
            elif Topology.IsInstance(topology, "Topology"):
                item = topology

            if Topology.IsInstance(item, "Topology"):
                boundary.append(item)

        if not boundary:
            if not silent:
                print("Cluster.ExternalBoundary - Error: No boundary Topologies could be created. Returning None.")
            return None
        return Cluster.ByTopologies(boundary, silent=silent)

    @staticmethod
    def HighestType(cluster, silent: bool = False) -> int:
        """
        Returns the type ID of the highest-dimensional topology present in a Cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input Cluster.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        int or None
            The highest Topologic type ID, or None for invalid/empty input.
        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
                print("Cluster.HighestType - Error: The input cluster parameter is not a valid Cluster. Returning None.")
            return None

        checks = [
            (Cluster.CellComplexes, "CellComplex"),
            (Cluster.Cells, "Cell"),
            (Cluster.Shells, "Shell"),
            (Cluster.Faces, "Face"),
            (Cluster.Wires, "Wire"),
            (Cluster.Edges, "Edge"),
            (Cluster.Vertices, "Vertex"),
        ]
        for method, type_name in checks:
            values = method(cluster, silent=True)
            if values:
                return Topology.TypeID(type_name)
        return None

    @staticmethod
    def Simplify(cluster, tolerance: float = 0.0001, silent: bool = False):
        """
        Simplifies a Cluster only when it has exactly one direct constituent.

        This method deliberately does not infer redundancy from vertex counts or
        geometric coincidence. A multi-member Cluster is returned unchanged.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input Cluster.
        tolerance : float, optional
            Reserved for backend-neutral hierarchy queries. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Topology or None
            The sole constituent when there is exactly one; otherwise the input Cluster.
        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
                print("Cluster.Simplify - Error: The input cluster parameter is not a valid Cluster. Returning None.")
            return None
        topologies = Cluster.Topologies(cluster, tolerance=tolerance, silent=True)
        if topologies is None:
            return None
        return topologies[0] if len(topologies) == 1 else cluster

    # -------------------------------------------------------------------------
    # Clustering algorithms
    # -------------------------------------------------------------------------

    @staticmethod
    def DBSCAN(
        topologies,
        selectors=None,
        keys=["x", "y", "z"],
        epsilon: float = 0.5,
        minSamples: int = 2,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Clusters Topologies using the DBSCAN density-based clustering algorithm.

        Coordinate features (``x``, ``y``, ``z``; case-insensitive) are read from
        selector Vertices. Additional features are read from the dictionaries of
        the corresponding Topologies and must be finite numeric values.

        Parameters
        ----------
        topologies : list
            The Topologies to cluster.
        selectors : list, optional
            Representative Vertices corresponding one-to-one with ``topologies``.
            Required when any input topology is not a Vertex. Default is None.
        keys : list, optional
            Feature keys. ``x``, ``y``, and ``z`` refer to selector coordinates;
            other keys refer to topology dictionary values. Default is
            ``["x", "y", "z"]``.
        epsilon : float, optional
            DBSCAN neighbourhood radius. Default is 0.5.
        minSamples : int, optional
            Minimum number of samples, including the sample itself, required for a
            core point. Default is 2.
        tolerance : float, optional
            Numerical allowance used at the epsilon boundary. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        tuple[list[topologic_core.Cluster], topologic_core.Cluster or None]
            The detected clusters and an optional Cluster containing noise points.
            ``(None, None)`` is returned for invalid input.
        """
        import math
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex

        if not isinstance(topologies, list) or not topologies:
            if not silent:
                print("Cluster.DBSCAN - Error: The input topologies parameter is not a valid non-empty list. Returning None.")
            return None, None
        if not all(Topology.IsInstance(t, "Topology") for t in topologies):
            if not silent:
                print("Cluster.DBSCAN - Error: Every item in topologies must be a valid Topology. Returning None.")
            return None, None
        if not isinstance(keys, list) or not keys:
            if not silent:
                print("Cluster.DBSCAN - Error: keys must be a valid non-empty list. Returning None.")
            return None, None

        try:
            epsilon = float(epsilon)
            tolerance = abs(float(tolerance))
            minSamples = int(minSamples)
        except Exception:
            if not silent:
                print("Cluster.DBSCAN - Error: epsilon, minSamples, or tolerance is invalid. Returning None.")
            return None, None
        if not math.isfinite(epsilon) or epsilon <= 0 or minSamples < 1 or minSamples > len(topologies):
            if not silent:
                print("Cluster.DBSCAN - Error: epsilon must be positive and minSamples must be between 1 and the number of Topologies. Returning None.")
            return None, None

        if selectors is None:
            if not all(Topology.IsInstance(t, "Vertex") for t in topologies):
                if not silent:
                    print("Cluster.DBSCAN - Error: selectors are required when topologies contain non-Vertex objects. Returning None.")
                return None, None
            selectors = topologies
        else:
            if not isinstance(selectors, list) or len(selectors) != len(topologies):
                if not silent:
                    print("Cluster.DBSCAN - Error: selectors must be a list with the same length as topologies. Returning None.")
                return None, None
            if not all(Topology.IsInstance(s, "Vertex") for s in selectors):
                if not silent:
                    print("Cluster.DBSCAN - Error: Every selector must be a valid Vertex. Returning None.")
                return None, None

        def numeric(value):
            try:
                value = float(value)
                return value if math.isfinite(value) else None
            except Exception:
                return None

        data = []
        for topology, selector in zip(topologies, selectors):
            try:
                dictionary = Topology.Dictionary(topology, silent=True)
            except TypeError:
                dictionary = Topology.Dictionary(topology)
            row = []
            for key in keys:
                key_lower = str(key).lower()
                if key_lower == "x":
                    value = Vertex.X(selector)
                elif key_lower == "y":
                    value = Vertex.Y(selector)
                elif key_lower == "z":
                    value = Vertex.Z(selector)
                else:
                    value = Dictionary.ValueAtKey(dictionary, key)
                value = numeric(value)
                if value is None:
                    if not silent:
                        print(f"Cluster.DBSCAN - Error: Feature '{key}' is missing or non-numeric. Returning None.")
                    return None, None
                row.append(value)
            data.append(row)

        n = len(data)
        limit2 = (epsilon + tolerance) ** 2

        def neighbours(index):
            source = data[index]
            result = []
            for j, target in enumerate(data):
                distance2 = sum((a - b) ** 2 for a, b in zip(source, target))
                if distance2 <= limit2:
                    result.append(j)
            return result

        UNVISITED = 0
        NOISE = -1
        labels = [UNVISITED] * n
        cluster_id = 0

        for i in range(n):
            if labels[i] != UNVISITED:
                continue
            seed_neighbours = neighbours(i)
            if len(seed_neighbours) < minSamples:
                labels[i] = NOISE
                continue

            cluster_id += 1
            labels[i] = cluster_id
            queue = list(seed_neighbours)
            queued = set(queue)
            cursor = 0
            while cursor < len(queue):
                j = queue[cursor]
                cursor += 1

                if labels[j] == NOISE:
                    labels[j] = cluster_id
                if labels[j] != UNVISITED:
                    continue

                labels[j] = cluster_id
                j_neighbours = neighbours(j)
                if len(j_neighbours) >= minSamples:
                    for neighbour in j_neighbours:
                        if neighbour not in queued:
                            queued.add(neighbour)
                            queue.append(neighbour)

        clusters = []
        for cid in range(1, cluster_id + 1):
            members = [topologies[i] for i, label in enumerate(labels) if label == cid]
            cluster = Cluster.ByTopologies(members, silent=True)
            if Topology.IsInstance(cluster, "Cluster"):
                clusters.append(cluster)

        noise_members = [topologies[i] for i, label in enumerate(labels) if label == NOISE]
        noise = Cluster.ByTopologies(noise_members, silent=True) if noise_members else None
        return clusters, noise

    @staticmethod
    def KMeans(
        topologies,
        selectors=None,
        keys=["x", "y", "z"],
        k=4,
        maxIterations=100,
        centroidKey="k_centroid",
        distanceMeasure: str = "euclidean",
        init: str = "kmeans++",
        nInit: int = 10,
        tol: float = 1e-6,
        standardize: bool = False,
        normalize: bool = False,
        randomSeed: int = None,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Clusters Topologies using a K-Means-like partitioning algorithm.

        Supported distance measures are ``euclidean``, ``sqeuclidean``,
        ``manhattan``, ``chebyshev``, ``cosine``, and ``mahalanobis``. Euclidean
        variants use arithmetic means; Manhattan uses coordinate-wise medians;
        Chebyshev uses coordinate-wise midranges; Cosine uses normalized means;
        Mahalanobis uses arithmetic means with a global inverse covariance metric.

        Parameters
        ----------
        topologies : list
            The Topologies to cluster.
        selectors : list, optional
            Representative Vertices corresponding one-to-one with ``topologies``.
            Required when any topology is not a Vertex. Default is None.
        keys : list, optional
            Feature keys. Coordinate keys are read from selectors; other keys are
            read from topology dictionaries. Default is ["x", "y", "z"].
        k : int, optional
            Number of clusters. Default is 4.
        maxIterations : int, optional
            Maximum iterations per restart. Default is 100.
        centroidKey : str, optional
            Dictionary key used to store each cluster centroid. Default is
            ``"k_centroid"``.
        distanceMeasure : str, optional
            Distance metric. Default is ``"euclidean"``.
        init : str, optional
            ``"kmeans++"`` or ``"random"``. Default is ``"kmeans++"``.
        nInit : int, optional
            Number of restarts. The solution with the lowest objective is retained.
            Default is 10.
        tol : float, optional
            Convergence tolerance. Default is 1e-6.
        standardize : bool, optional
            If True, z-score standardizes features before clustering. Default False.
        normalize : bool, optional
            If True, L2-normalizes feature rows. Default False.
        randomSeed : int, optional
            Random seed. Default is None.
        mantissa : int, optional
            Decimal precision used when storing centroids. Default is 6.
        tolerance : float, optional
            Topologic tolerance retained for API consistency. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list[topologic_core.Cluster] or None
            The resulting Clusters, each carrying its centroid in its dictionary.
        """
        import math
        try:
            import numpy as np
        except Exception:
            if not silent:
                print("Cluster.KMeans - Error: NumPy is required for KMeans. Returning None.")
            return None

        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex

        if not isinstance(topologies, list) or not topologies:
            if not silent:
                print("Cluster.KMeans - Error: topologies must be a valid non-empty list. Returning None.")
            return None
        if not all(Topology.IsInstance(t, "Topology") for t in topologies):
            if not silent:
                print("Cluster.KMeans - Error: Every item in topologies must be a valid Topology. Returning None.")
            return None
        if not isinstance(keys, list) or not keys:
            if not silent:
                print("Cluster.KMeans - Error: keys must be a valid non-empty list. Returning None.")
            return None

        if selectors is None:
            if not all(Topology.IsInstance(t, "Vertex") for t in topologies):
                if not silent:
                    print("Cluster.KMeans - Error: selectors are required when topologies contain non-Vertex objects. Returning None.")
                return None
            selectors = topologies
        else:
            if not isinstance(selectors, list) or len(selectors) != len(topologies):
                if not silent:
                    print("Cluster.KMeans - Error: selectors must have the same length as topologies. Returning None.")
                return None
            if not all(Topology.IsInstance(s, "Vertex") for s in selectors):
                if not silent:
                    print("Cluster.KMeans - Error: Every selector must be a valid Vertex. Returning None.")
                return None

        try:
            k = int(k)
            maxIterations = int(maxIterations)
            nInit = int(nInit)
            tol = abs(float(tol))
            mantissa = max(0, int(mantissa))
            tolerance = abs(float(tolerance))
        except Exception:
            if not silent:
                print("Cluster.KMeans - Error: One or more numerical parameters are invalid. Returning None.")
            return None

        n = len(topologies)
        if k < 1 or k > n or maxIterations < 1 or nInit < 1:
            if not silent:
                print("Cluster.KMeans - Error: Require 1 <= k <= number of Topologies, maxIterations >= 1, and nInit >= 1. Returning None.")
            return None

        distanceMeasure = str(distanceMeasure or "euclidean").strip().lower()
        allowed_metrics = {"euclidean", "sqeuclidean", "manhattan", "chebyshev", "cosine", "mahalanobis"}
        if distanceMeasure not in allowed_metrics:
            if not silent:
                print(f"Cluster.KMeans - Error: Unsupported distanceMeasure '{distanceMeasure}'. Returning None.")
            return None
        init = str(init or "kmeans++").strip().lower()
        if init not in {"kmeans++", "random"}:
            if not silent:
                print("Cluster.KMeans - Error: init must be 'kmeans++' or 'random'. Returning None.")
            return None

        def safe_float(value):
            try:
                value = float(value)
                return value if math.isfinite(value) else None
            except Exception:
                return None

        rows = []
        for topology, selector in zip(topologies, selectors):
            try:
                dictionary = Topology.Dictionary(topology, silent=True)
            except TypeError:
                dictionary = Topology.Dictionary(topology)
            row = []
            for key in keys:
                lower = str(key).lower()
                if lower == "x":
                    value = Vertex.X(selector)
                elif lower == "y":
                    value = Vertex.Y(selector)
                elif lower == "z":
                    value = Vertex.Z(selector)
                else:
                    value = Dictionary.ValueAtKey(dictionary, key)
                value = safe_float(value)
                if value is None:
                    if not silent:
                        print(f"Cluster.KMeans - Warning: Feature '{key}' is missing or non-numeric; using 0.0.")
                    value = 0.0
                row.append(value)
            rows.append(row)

        X = np.asarray(rows, dtype=float)
        if X.ndim != 2 or X.shape[0] != n or X.shape[1] < 1:
            if not silent:
                print("Cluster.KMeans - Error: Could not construct a valid feature matrix. Returning None.")
            return None

        X_work = X.copy()
        mu = None
        sigma = None
        if standardize:
            mu = X_work.mean(axis=0)
            sigma = X_work.std(axis=0)
            sigma[sigma == 0] = 1.0
            X_work = (X_work - mu) / sigma
        if normalize:
            norms = np.linalg.norm(X_work, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X_work = X_work / norms

        inv_cov = None
        if distanceMeasure == "mahalanobis":
            cov = np.atleast_2d(np.cov(X_work, rowvar=False))
            regularized = cov + 1e-9 * np.eye(cov.shape[0])
            try:
                inv_cov = np.linalg.inv(regularized)
            except Exception:
                inv_cov = np.linalg.pinv(regularized)

        def pairwise_distances(A, C):
            if distanceMeasure in {"euclidean", "sqeuclidean"}:
                diff = A[:, None, :] - C[None, :, :]
                d2 = np.sum(diff * diff, axis=2)
                return np.sqrt(d2) if distanceMeasure == "euclidean" else d2
            if distanceMeasure == "manhattan":
                return np.sum(np.abs(A[:, None, :] - C[None, :, :]), axis=2)
            if distanceMeasure == "chebyshev":
                return np.max(np.abs(A[:, None, :] - C[None, :, :]), axis=2)
            if distanceMeasure == "cosine":
                an = np.linalg.norm(A, axis=1, keepdims=True)
                cn = np.linalg.norm(C, axis=1, keepdims=True).T
                an[an == 0] = 1.0
                cn[cn == 0] = 1.0
                similarity = np.clip((A @ C.T) / (an * cn), -1.0, 1.0)
                return 1.0 - similarity
            diff = A[:, None, :] - C[None, :, :]
            d2 = np.einsum("nkd,df,nkf->nk", diff, inv_cov, diff)
            return np.sqrt(np.maximum(d2, 0.0))

        def update_centroids(A, labels):
            C = np.zeros((k, A.shape[1]), dtype=float)
            counts = np.zeros(k, dtype=int)
            for j in range(k):
                members = A[labels == j]
                counts[j] = members.shape[0]
                if counts[j] == 0:
                    continue
                if distanceMeasure in {"euclidean", "sqeuclidean", "mahalanobis"}:
                    C[j] = members.mean(axis=0)
                elif distanceMeasure == "manhattan":
                    C[j] = np.median(members, axis=0)
                elif distanceMeasure == "chebyshev":
                    C[j] = 0.5 * (members.min(axis=0) + members.max(axis=0))
                else:  # cosine
                    centroid = members.mean(axis=0)
                    norm = np.linalg.norm(centroid)
                    C[j] = centroid / (norm if norm > 0 else 1.0)
            return C, counts

        rng = np.random.default_rng(randomSeed)

        def initialize(A):
            if init == "random":
                indices = rng.choice(A.shape[0], size=k, replace=False)
                return A[indices].copy()

            chosen = [int(rng.integers(0, A.shape[0]))]
            C = [A[chosen[0]].copy()]
            closest_d2 = np.sum((A - C[0]) ** 2, axis=1)
            for _ in range(1, k):
                available = [i for i in range(A.shape[0]) if i not in chosen]
                if not available:
                    break
                weights = closest_d2.copy()
                weights[chosen] = 0.0
                total = float(weights.sum())
                if total <= 0:
                    index = int(rng.choice(available))
                else:
                    index = int(rng.choice(A.shape[0], p=weights / total))
                    if index in chosen:
                        index = int(rng.choice(available))
                chosen.append(index)
                C.append(A[index].copy())
                closest_d2 = np.minimum(closest_d2, np.sum((A - A[index]) ** 2, axis=1))
            return np.vstack(C)

        def objective(A, C, labels):
            if distanceMeasure in {"euclidean", "sqeuclidean"}:
                diff = A - C[labels]
                return float(np.sum(diff * diff))
            distances = pairwise_distances(A, C)
            return float(distances[np.arange(A.shape[0]), labels].sum())

        def solve_once(A):
            C = initialize(A)
            previous = None
            labels = np.zeros(A.shape[0], dtype=int)

            for _ in range(maxIterations):
                distances = pairwise_distances(A, C)
                labels = np.argmin(distances, axis=1)
                C_new, counts = update_centroids(A, labels)

                if np.any(counts == 0):
                    assigned_distance = distances[np.arange(A.shape[0]), labels]
                    farthest = np.argsort(-assigned_distance)
                    used = set()
                    for empty in np.where(counts == 0)[0]:
                        pick = next((int(i) for i in farthest if int(i) not in used), int(farthest[0]))
                        used.add(pick)
                        C_new[empty] = A[pick]

                shift = float(np.linalg.norm(C_new - C))
                C = C_new
                current = objective(A, C, labels)
                if shift <= tol or (previous is not None and abs(previous - current) <= tol * (abs(previous) + tol)):
                    break
                previous = current

            distances = pairwise_distances(A, C)
            labels = np.argmin(distances, axis=1)

            # Guarantee non-empty output clusters by moving a farthest donor point
            # from a cluster that has more than one member into each empty cluster.
            counts = np.bincount(labels, minlength=k)
            for empty in np.where(counts == 0)[0]:
                donors = [j for j in range(k) if counts[j] > 1]
                if not donors:
                    break
                donor = max(donors, key=lambda j: counts[j])
                donor_indices = np.where(labels == donor)[0]
                donor_distances = distances[donor_indices, donor]
                pick = int(donor_indices[int(np.argmax(donor_distances))])
                labels[pick] = int(empty)
                counts[donor] -= 1
                counts[empty] += 1

            C, _ = update_centroids(A, labels)
            return labels, C, objective(A, C, labels)

        best = None
        best_objective = float("inf")
        for _ in range(nInit):
            labels, centroids, score = solve_once(X_work)
            if score < best_objective:
                best_objective = score
                best = (labels.copy(), centroids.copy())

        if best is None:
            return None
        labels, C_work = best

        if normalize:
            C_store = C_work.copy()
        elif standardize:
            C_store = C_work * sigma + mu
        else:
            C_store = C_work.copy()
        C_store = np.round(C_store.astype(float), mantissa).tolist()

        result = []
        for j in range(k):
            indices = np.where(labels == j)[0].tolist()
            if not indices:
                continue
            members = [topologies[i] for i in indices]
            cluster = Cluster.ByTopologies(members, silent=True)
            if not Topology.IsInstance(cluster, "Cluster"):
                continue
            dictionary = Dictionary.ByKeysValues([centroidKey], [C_store[j]])
            cluster = Topology.SetDictionary(cluster, dictionary, silent=True)
            result.append(cluster)
        return result

    # -------------------------------------------------------------------------
    # Cell grouping and geometric helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def MergeCells(cells, tolerance: float = 0.0001, silent: bool = False):
        """
        Groups face-adjacent Cells into CellComplexes and retains isolated Cells.

        Connected components are found with a complete breadth-first traversal.
        Shared Face queries are attempted first. A backend merge test is used only
        as a compatibility fallback for independently constructed Cells whose
        coincident interface is not represented by shared topological identity.

        Parameters
        ----------
        cells : list
            The input Cells.
        tolerance : float, optional
            The desired tolerance. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cluster or None
            A Cluster containing one CellComplex per connected component of two or
            more Cells plus any isolated Cells.
        """
        from collections import deque
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Topology import Topology

        if not isinstance(cells, list):
            if not silent:
                print("Cluster.MergeCells - Error: cells must be a valid list. Returning None.")
            return None
        valid_cells = [cell for cell in cells if Topology.IsInstance(cell, "Cell")]
        if len(valid_cells) != len(cells) or not valid_cells:
            if not silent:
                print("Cluster.MergeCells - Error: cells must contain one or more valid Cells only. Returning None.")
            return None

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            tolerance = 0.0001

        adjacency_cache = {}

        def adjacent(i, j):
            key = (min(i, j), max(i, j))
            if key in adjacency_cache:
                return adjacency_cache[key]
            cell_a = valid_cells[i]
            cell_b = valid_cells[j]

            shared = None
            try:
                shared = Topology.SharedFaces(cell_a, cell_b, silent=True)
            except TypeError:
                try:
                    shared = Topology.SharedFaces(cell_a, cell_b)
                except Exception:
                    shared = None
            except Exception:
                shared = None
            if shared:
                adjacency_cache[key] = True
                return True

            # Compatibility fallback for coincident but independently-created faces.
            try:
                merged = Topology.Merge(cell_a, cell_b, tolerance=tolerance, silent=True)
            except TypeError:
                try:
                    merged = Topology.Merge(cell_a, cell_b, tolerance=tolerance)
                except Exception:
                    merged = None
            except Exception:
                merged = None
            value = Topology.IsInstance(merged, "CellComplex")
            adjacency_cache[key] = bool(value)
            return bool(value)

        remaining = set(range(len(valid_cells)))
        components = []
        while remaining:
            seed = remaining.pop()
            component = [seed]
            queue = deque([seed])
            while queue:
                current = queue.popleft()
                neighbours = [index for index in list(remaining) if adjacent(current, index)]
                for index in neighbours:
                    remaining.remove(index)
                    component.append(index)
                    queue.append(index)
            components.append(component)

        output = []
        for component in components:
            component_cells = [valid_cells[index] for index in component]
            if len(component_cells) == 1:
                output.append(component_cells[0])
                continue
            cell_complex = CellComplex.ByCells(component_cells, tolerance=tolerance, silent=True)
            if Topology.IsInstance(cell_complex, "CellComplex"):
                output.append(cell_complex)
            else:
                # Never lose input Cells if a backend cannot assemble one component.
                output.extend(component_cells)

        return Cluster.ByTopologies(output, silent=silent)

    @staticmethod
    def MysticRose(
        wire=None,
        origin=None,
        radius: float = 0.5,
        sides: int = 16,
        perimeter: bool = True,
        direction: list = [0, 0, 1],
        placement: str = "center",
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Creates a Mystic Rose as a Cluster of chords between non-adjacent vertices.

        Parameters
        ----------
        wire : topologic_core.Wire, optional
            Closed source Wire. If None, a polygonal circle is created. Default None.
        origin : topologic_core.Vertex, optional
            Origin used when a source Wire is generated. Default is the global origin.
        radius : float, optional
            Radius of the generated source circle. Default is 0.5.
        sides : int, optional
            Number of source circle sides. Default is 16.
        perimeter : bool, optional
            If True, includes the source Wire perimeter Edges. Default is True.
        direction : list, optional
            Source circle normal. Default is [0, 0, 1].
        placement : str, optional
            Source circle placement. Default is "center".
        tolerance : float, optional
            The desired tolerance. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cluster or None
            The created Edge Cluster.
        """
        from itertools import combinations
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Wire import Wire

        if wire is None:
            try:
                radius = abs(float(radius))
                sides = int(sides)
            except Exception:
                if not silent:
                    print("Cluster.MysticRose - Error: radius or sides is invalid. Returning None.")
                return None
            if radius <= tolerance or sides < 3:
                if not silent:
                    print("Cluster.MysticRose - Error: radius must exceed tolerance and sides must be at least 3. Returning None.")
                return None
            wire = Wire.Circle(
                origin=origin,
                radius=radius,
                sides=sides,
                fromAngle=0,
                toAngle=360,
                close=True,
                direction=direction,
                placement=placement,
                tolerance=tolerance,
                silent=True,
            )

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Cluster.MysticRose - Error: wire is not a valid Wire. Returning None.")
            return None
        try:
            closed = Wire.IsClosed(wire, tolerance=tolerance, silent=True)
        except TypeError:
            try:
                closed = Wire.IsClosed(wire)
            except Exception:
                closed = False
        if not closed:
            if not silent:
                print("Cluster.MysticRose - Error: wire must be closed. Returning None.")
            return None

        vertices = Topology.Vertices(wire, silent=True) or []
        if len(vertices) < 3:
            return None

        edges = list(Wire.Edges(wire) or []) if perimeter else []
        n = len(vertices)
        for a, b in combinations(range(n), 2):
            if abs(a - b) in {1, n - 1}:
                continue
            edge = Edge.ByVertices([vertices[a], vertices[b]], tolerance=tolerance, silent=True)
            if Topology.IsInstance(edge, "Edge"):
                edges.append(edge)
        return Cluster.ByTopologies(edges, silent=silent)

    @staticmethod
    def Tripod(
        size: float = 1.0,
        radius: float = 0.03,
        sides: int = 4,
        faceColorKey="faceColor",
        xColor="red",
        yColor="green",
        zColor="blue",
        matrix=None,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Creates a three-axis colour-coded XYZ tripod.

        Parameters
        ----------
        size : float, optional
            Overall axis length. Default is 1.0.
        radius : float, optional
            Shaft radius. Default is 0.03.
        sides : int, optional
            Number of radial sides used by shafts and arrow heads. Default is 4.
        faceColorKey : str, optional
            Dictionary key used to store Face colours. Default is "faceColor".
        xColor, yColor, zColor : optional
            Values stored for X, Y, and Z axis Face colours.
        matrix : list, optional
            Optional 4x4 transformation matrix applied to the resulting Cluster.
        tolerance : float, optional
            The desired tolerance. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cluster or None
            The created tripod.
        """
        from topologicpy.Cell import Cell
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        try:
            size = abs(float(size))
            radius = abs(float(radius))
            sides = int(sides)
            tolerance = abs(float(tolerance))
        except Exception:
            if not silent:
                print("Cluster.Tripod - Error: size, radius, sides, or tolerance is invalid. Returning None.")
            return None
        if size <= tolerance or radius <= tolerance or sides < 3:
            if not silent:
                print("Cluster.Tripod - Error: size and radius must exceed tolerance and sides must be at least 3. Returning None.")
            return None

        shaft_height = size * 0.7
        head_height = size - shaft_height
        cylinder = Cell.Cylinder(radius=radius, height=shaft_height, uSides=sides, placement="bottom", tolerance=tolerance)
        cone = Cell.Cone(baseRadius=radius * 2.25, height=head_height, placement="bottom", uSides=sides, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(cylinder, "Cell") or not Topology.IsInstance(cone, "Cell"):
            return None
        cone = Topology.Translate(cone, 0, 0, shaft_height, silent=True)
        z_arrow = Topology.Union(cylinder, cone, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(z_arrow, "Topology"):
            return None
        x_arrow = Topology.Rotate(z_arrow, axis=[0, 1, 0], angle=90, tolerance=tolerance, silent=True)
        y_arrow = Topology.Rotate(z_arrow, axis=[1, 0, 0], angle=-90, tolerance=tolerance, silent=True)

        for arrow, color in ((x_arrow, xColor), (y_arrow, yColor), (z_arrow, zColor)):
            for face in Topology.Faces(arrow, silent=True) or []:
                dictionary = Dictionary.ByKeyValue(faceColorKey, color)
                try:
                    Topology.SetDictionary(face, dictionary, silent=True)
                except Exception:
                    pass

        cluster = Cluster.ByTopologies(x_arrow, y_arrow, z_arrow, silent=silent)
        if not Topology.IsInstance(cluster, "Cluster"):
            return None
        if matrix is not None:
            cluster = Topology.Transform(cluster, matrix=matrix, tolerance=tolerance, silent=silent)
        return cluster
