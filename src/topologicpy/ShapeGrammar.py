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

import copy
from typing import Any, Dict, List, Optional, Tuple


class ShapeGrammar:
    """A lightweight topology shape-grammar container.

    A ``ShapeGrammar`` stores transformation rules. Each rule consists of an
    input topology pattern, an optional output topology, an operation, and an
    optional 4x4 matrix that prepares the rule output relative to the rule
    input. ``ApplicableRules`` can then compute matching transformations, and
    ``ApplyRule`` applies a selected rule to a target topology.
    """

    _OPERATION_SPECS = (
        ("Replace", "Replaces the input topology with the output topology.", None, None, None),
        ("Transform", "Transforms the input topology using the specified matrix.", None, None, None),
        ("Union", "Unions the input topology and the output topology.", None, None, None),
        ("Difference", "Subtracts the output topology from the input topology.", None, None, None),
        ("Symmetric Difference", "Calculates the symmetrical difference of the input topology and the output topology.", None, None, None),
        ("Intersect", "Intersects the input topology and the output topology.", None, None, None),
        ("Merge", "Merges the input topology and the output topology.", None, None, None),
        ("Slice", "Slices the input topology using the output topology.", None, None, None),
        ("Impose", "Imposes the output topology on the input topology.", None, None, None),
        ("Imprint", "Imprints the output topology on the input topology.", None, None, None),
        ("Divide", "Divides the input topology along the x, y, and z axes using the specified number of sides (uSides, vSides, wSides).", 2, 2, 2),
    )

    def __init__(self):
        self.title = "Untitled"
        self.description = ""
        self.rules = []
        self.operations = [
            {"title": title, "description": description, "uSides": u, "vSides": v, "wSides": w}
            for title, description, u, v, w in self._OPERATION_SPECS
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_4x4_matrix(matrix) -> bool:
        if not isinstance(matrix, (list, tuple)) or len(matrix) != 4:
            return False
        for row in matrix:
            if not isinstance(row, (list, tuple)) or len(row) != 4:
                return False
            for value in row:
                try:
                    float(value)
                except Exception:
                    return False
        return True

    @staticmethod
    def _copy_matrix(matrix):
        if matrix is None:
            return None
        return [[float(value) for value in row] for row in matrix]

    @staticmethod
    def _op_title(operation) -> str:
        if isinstance(operation, dict):
            return str(operation.get("title", ""))
        if operation is None:
            return ""
        return str(operation)

    def _normalise_operation(self, operation, silent: bool = False):
        """Returns a canonical operation dictionary or None if invalid.

        ``operation=None`` intentionally means the documented default operation:
        ``Replace``.
        """
        if operation is None:
            return copy.deepcopy(self.OperationByTitle("Replace"))
        if operation in self.operations:
            return copy.deepcopy(operation)
        if isinstance(operation, str):
            op = self.OperationByTitle(operation)
            if op is not None:
                return copy.deepcopy(op)
        if isinstance(operation, dict):
            op = self.OperationByTitle(operation.get("title"))
            if op is not None:
                normalised = copy.deepcopy(op)
                # Preserve explicit divide-side overrides supplied in the rule.
                for key in ("uSides", "vSides", "wSides"):
                    if key in operation and operation[key] is not None:
                        normalised[key] = operation[key]
                return normalised
        if not silent:
            print("ShapeGrammar - Error: The operation parameter is not a valid operation. Returning None.")
        return None

    @staticmethod
    def _is_topology(obj) -> bool:
        try:
            from topologicpy.Topology import Topology
            return bool(Topology.IsInstance(obj, "Topology"))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Operations and rule storage
    # ------------------------------------------------------------------
    def OperationTitles(self):
        """
        Returns the list of available operation titles.

        Returns
        -------
        list
            The requested list of operation titles.
        """
        return [op["title"] for op in self.operations]

    def OperationByTitle(self, title):
        """
        Returns the operation matching the input title string.

        Parameters
        ----------
        title : str
            The input operation title. See ``OperationTitles`` for available
            operations. Matching is case-insensitive; exact matches are preferred
            over substring matches.

        Returns
        -------
        dict or None
            The requested operation dictionary, or None if no operation matches.
        """
        if not isinstance(title, str) or title.strip() == "":
            return None
        query = title.strip().lower()
        for op in self.operations:
            if query == str(op.get("title", "")).lower():
                return op
        for op in self.operations:
            op_title = str(op.get("title", ""))
            if query in op_title.lower():
                return op
        return None

    def AddRule(self,
                input,
                output,
                title: str = "Untitled Rule",
                description: str = "",
                operation: dict = None,
                matrix: list = None,
                silent: bool = False):
        """
        Adds a rule to the topology grammar.

        Parameters
        ----------
        input : topologic_core.Topology
            The input topology of the rule.
        output : topologic_core.Topology
            The output topology of the rule. This may be None for transform-like
            rules that only use the input topology.
        title : str, optional
            The title of the rule. Default is ``"Untitled Rule"``.
        description : str, optional
            The description of the rule. Default is ``""``.
        operation : dict or str, optional
            The desired rule operation. If set to None, the replacement rule is
            applied. Default is None.
        matrix : list, optional
            The 4x4 transformation matrix that transforms the output topology to
            the input topology. If set to None, no rule-preparation transform is
            applied. Default is None.
        silent : bool, optional
            If set to True, error and warning messages are suppressed. Default is
            False.

        Returns
        -------
        None
            This method does not return a value.
        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(input, "Topology"):
            if not silent:
                print("ShapeGrammar.AddRule - Error: The input parameter is not a valid topology. Returning None.")
            return None
        if output is not None and not Topology.IsInstance(output, "Topology"):
            if not silent:
                print("ShapeGrammar.AddRule - Error: The output parameter is not a valid topology. Returning None.")
            return None

        op = self._normalise_operation(operation, silent=silent)
        if op is None:
            return None

        if matrix is not None and not self._is_4x4_matrix(matrix):
            if not silent:
                print("ShapeGrammar.AddRule - Error: The matrix parameter is not a valid 4x4 numeric matrix. Returning None.")
            return None

        self.rules.append({
            "input": input,
            "output": output,
            "title": str(title) if title is not None else "Untitled Rule",
            "description": str(description) if description is not None else "",
            "operation": op,
            "matrix": self._copy_matrix(matrix),
        })
        return None

    def ApplicableRules(self, topology, keys: list = None, silent: bool = False):
        """
        Returns rules applicable to the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        keys : list, optional
            The dictionary keys to semantically match. Default is None, meaning
            dictionaries are not considered.
        silent : bool, optional
            If set to True, error and warning messages are suppressed. Default is
            False.

        Returns
        -------
        tuple
            ``(rules, matrices)`` where ``rules`` is the list of applicable rules
            and ``matrices`` is the corresponding list of similarity transforms.
            Returns None if the input topology is invalid.
        """
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("ShapeGrammar.ApplicableRules - Error: The topology parameter is not a valid topology. Returning None.")
            return None

        ap_rules = []
        ap_trans = []
        d = Topology.Dictionary(topology)
        match_keys = [k for k in keys if isinstance(k, str)] if isinstance(keys, list) else []

        for rule in self.rules:
            if not isinstance(rule, dict) or "input" not in rule:
                continue
            dict_status = True
            rule_input = rule.get("input")
            if match_keys:
                d_input = Topology.Dictionary(rule_input)
                for key in match_keys:
                    if Dictionary.ValueAtKey(d, key, None) != Dictionary.ValueAtKey(d_input, key, None):
                        dict_status = False
                        break
            if not dict_status:
                continue

            try:
                similar = Topology.IsSimilar(rule_input, topology)
            except Exception:
                similar = (False, None)
            if isinstance(similar, (list, tuple)) and len(similar) >= 2:
                topology_status, mat = bool(similar[0]), similar[1]
            else:
                topology_status, mat = bool(similar), None
            if topology_status:
                ap_rules.append(rule)
                ap_trans.append(mat)
        return ap_rules, ap_trans

    # ------------------------------------------------------------------
    # Rule application
    # ------------------------------------------------------------------
    def ApplyRule(self, topology, rule: dict = None, matrix: list = None, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Applies a rule to the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        rule : dict, optional
            The desired rule to apply. If None, the method returns the topology
            transformed by ``matrix`` when supplied, otherwise the input topology.
        matrix : list, optional
            The 4x4 transformation matrix that transforms the rule result to the
            target topology. If set to None, no final transformation is applied.
        mantissa : int, optional
            Decimal precision. Default is 6.
        tolerance : float, optional
            The desired tolerance. Default is 0.0001.
        silent : bool, optional
            If set to True, error and warning messages are suppressed. Default is
            False.

        Returns
        -------
        topologic_core.Topology or None
            The transformed topology, or None if validation fails.
        """
        from topologicpy.Topology import Topology

        def _divide_by_sides(base_topology, uSides, vSides, wSides):
            from topologicpy.Cluster import Cluster
            from topologicpy.Face import Face
            from topologicpy.Vertex import Vertex

            def _bb(topo):
                vertices = Topology.Vertices(topo, silent=True) or []
                if not vertices:
                    return None
                xs, ys, zs = [], [], []
                for v in vertices:
                    xs.append(Vertex.X(v, mantissa=mantissa))
                    ys.append(Vertex.Y(v, mantissa=mantissa))
                    zs.append(Vertex.Z(v, mantissa=mantissa))
                return [min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)]

            try:
                uSides = int(uSides)
                vSides = int(vSides)
                wSides = int(wSides)
            except Exception:
                return base_topology
            if uSides < 1 or vSides < 1 or wSides < 1:
                return base_topology

            bounds = _bb(base_topology)
            if bounds is None:
                return base_topology
            x_min, y_min, z_min, maxX, maxY, maxZ = bounds
            dx, dy, dz = maxX - x_min, maxY - y_min, maxZ - z_min
            if abs(dx) <= float(tolerance) and abs(dy) <= float(tolerance) and abs(dz) <= float(tolerance):
                return base_topology

            centroid = Vertex.ByCoordinates(x_min + dx * 0.5, y_min + dy * 0.5, z_min + dz * 0.5)
            faces = []
            if wSides > 1 and abs(dz) > float(tolerance):
                w_origin = Vertex.ByCoordinates(Vertex.X(centroid, mantissa=mantissa), Vertex.Y(centroid, mantissa=mantissa), z_min)
                w_face = Face.Rectangle(origin=w_origin, width=max(dx * 1.1, tolerance), length=max(dy * 1.1, tolerance))
                offset = dz / float(wSides)
                for i in range(wSides - 1):
                    faces.append(Topology.Translate(w_face, 0, 0, offset * (i + 1)))
            if uSides > 1 and abs(dx) > float(tolerance):
                u_origin = Vertex.ByCoordinates(x_min, Vertex.Y(centroid, mantissa=mantissa), Vertex.Z(centroid, mantissa=mantissa))
                u_face = Face.Rectangle(origin=u_origin, width=max(dz * 1.1, tolerance), length=max(dy * 1.1, tolerance), direction=[1, 0, 0])
                offset = dx / float(uSides)
                for i in range(uSides - 1):
                    faces.append(Topology.Translate(u_face, offset * (i + 1), 0, 0))
            if vSides > 1 and abs(dy) > float(tolerance):
                v_origin = Vertex.ByCoordinates(Vertex.X(centroid, mantissa=mantissa), y_min, Vertex.Z(centroid, mantissa=mantissa))
                v_face = Face.Rectangle(origin=v_origin, width=max(dx * 1.1, tolerance), length=max(dz * 1.1, tolerance), direction=[0, 1, 0])
                offset = dy / float(vSides)
                for i in range(vSides - 1):
                    faces.append(Topology.Translate(v_face, 0, offset * (i + 1), 0))
            if not faces:
                return base_topology
            return Topology.Slice(base_topology, Cluster.ByTopologies(faces), tolerance=tolerance)

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("ShapeGrammar.ApplyRule - Error: The topology parameter is not a valid topology. Returning None.")
            return None
        if matrix is not None and not self._is_4x4_matrix(matrix):
            if not silent:
                print("ShapeGrammar.ApplyRule - Error: The matrix parameter is not a valid 4x4 numeric matrix. Returning None.")
            return None

        result_output = topology
        if rule is not None:
            if not isinstance(rule, dict):
                if not silent:
                    print("ShapeGrammar.ApplyRule - Error: The rule parameter is not a valid rule. Returning None.")
                return None
            rule_input = rule.get("input", topology)
            output = rule.get("output", None)
            r_matrix = rule.get("matrix", None)
            if r_matrix is not None and not self._is_4x4_matrix(r_matrix):
                if not silent:
                    print("ShapeGrammar.ApplyRule - Error: The rule matrix is not a valid 4x4 numeric matrix. Returning None.")
                return None
            operation = self._normalise_operation(rule.get("operation", None), silent=silent)
            if operation is None:
                return None

            if not Topology.IsInstance(rule_input, "Topology"):
                if not silent:
                    print("ShapeGrammar.ApplyRule - Error: The rule input is not a valid topology. Returning None.")
                return None
            if output is not None and not Topology.IsInstance(output, "Topology"):
                if not silent:
                    print("ShapeGrammar.ApplyRule - Error: The rule output is not a valid topology. Returning None.")
                return None

            op_title = str(operation.get("title", "Replace")).strip().lower()
            temp_output = output
            if r_matrix is not None and output is not None:
                temp_output = Topology.Transform(output, self._copy_matrix(r_matrix))

            # IMPORTANT: Symmetric Difference must be tested before Difference.
            if "symmetric difference" in op_title or "symmetrical difference" in op_title:
                result_output = Topology.SymmetricDifference(rule_input, temp_output)
            elif "replace" in op_title:
                result_output = temp_output
            elif "transform" in op_title:
                result_output = Topology.Transform(topology, self._copy_matrix(r_matrix)) if r_matrix is not None else topology
            elif "union" in op_title:
                result_output = Topology.Union(rule_input, temp_output)
            elif "difference" in op_title:
                result_output = Topology.Difference(rule_input, temp_output)
            elif "intersect" in op_title:
                result_output = Topology.Intersect(rule_input, temp_output)
            elif "merge" in op_title:
                result_output = Topology.Merge(rule_input, temp_output)
            elif "slice" in op_title:
                result_output = Topology.Slice(rule_input, temp_output)
            elif "impose" in op_title:
                result_output = Topology.Impose(rule_input, temp_output)
            elif "imprint" in op_title:
                result_output = Topology.Imprint(rule_input, temp_output)
            elif "divide" in op_title:
                result_output = _divide_by_sides(rule_input, operation.get("uSides"), operation.get("vSides"), operation.get("wSides"))
            else:
                result_output = topology

        if matrix is not None and result_output is not None:
            result_output = Topology.Transform(result_output, self._copy_matrix(matrix))
        return result_output

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------
    def ClusterByInputOutput(self, input, output, silent: bool = False):
        """
        Returns a cluster containing the input topology, output topology, and a
        small arrow between them.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Cluster import Cluster

        if not Topology.IsInstance(input, "Topology"):
            if not silent:
                print("ShapeGrammar.ClusterByInputOutput - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        if not Topology.IsInstance(output, "Topology"):
            if not silent:
                print("ShapeGrammar.ClusterByInputOutput - Error: The output topology parameter is not a valid topology. Returning None.")
            return None

        def _scaled_copy(topo, x_offset):
            bb = Topology.BoundingBox(topo)
            if bb is None:
                return None
            centroid = Topology.Centroid(bb)
            d = Topology.Dictionary(bb)
            xmin = Dictionary.ValueAtKey(d, "xmin", 0)
            ymin = Dictionary.ValueAtKey(d, "ymin", 0)
            zmin = Dictionary.ValueAtKey(d, "zmin", 0)
            xmax = Dictionary.ValueAtKey(d, "xmax", xmin)
            ymax = Dictionary.ValueAtKey(d, "ymax", ymin)
            zmax = Dictionary.ValueAtKey(d, "zmax", zmin)
            extent = max(float(xmax) - float(xmin), float(ymax) - float(ymin), float(zmax) - float(zmin))
            sf = 1.0 / extent if extent > 0 else 1.0
            temp = Topology.Translate(topo, -Vertex.X(centroid), -Vertex.Y(centroid), -Vertex.Z(centroid))
            temp = Topology.Scale(temp, x=sf, y=sf, z=sf)
            return Topology.Translate(temp, x_offset, 0, 0)

        temp_input = _scaled_copy(input, 0.5)
        temp_output = _scaled_copy(output, 2.5)
        if temp_input is None or temp_output is None:
            return None

        cyl = Cell.Cylinder(radius=0.04, height=0.4, placement="bottom")
        cyl = Topology.Rotate(cyl, axis=[0, 1, 0], angle=90)
        cyl = Topology.Translate(cyl, 1.25, 0, 0)

        cone = Cell.Cone(baseRadius=0.1, topRadius=0, height=0.15, placement="bottom")
        cone = Topology.Rotate(cone, axis=[0, 1, 0], angle=90)
        cone = Topology.Translate(cone, 1.65, 0, 0)
        cluster = Cluster.ByTopologies([temp_input, temp_output, cyl, cone])
        return Topology.Place(cluster, originA=Topology.Centroid(cluster), originB=Vertex.Origin())

    def ClusterByRule(self, rule, silent: bool = False):
        """Returns a cluster visualising the input rule."""
        if not isinstance(rule, dict) or "input" not in rule:
            if not silent:
                print("ShapeGrammar.ClusterByRule - Error: The rule parameter is not a valid rule. Returning None.")
            return None
        input_topology = rule.get("input")
        output_topology = self.ApplyRule(input_topology, rule, silent=silent)
        if output_topology is None:
            return None
        return self.ClusterByInputOutput(input_topology, output_topology, silent=silent)

    def FigureByInputOutput(self, input, output, silent: bool = False):
        """Returns a Plotly figure of the input and output topologies as a rule."""
        from topologicpy.Topology import Topology
        from topologicpy.Plotly import Plotly

        if not Topology.IsInstance(input, "Topology"):
            if not silent:
                print("ShapeGrammar.FigureByInputOutput - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        if not Topology.IsInstance(output, "Topology"):
            if not silent:
                print("ShapeGrammar.FigureByInputOutput - Error: The output topology parameter is not a valid topology. Returning None.")
            return None

        cluster = self.ClusterByInputOutput(input, output, silent=silent)
        if cluster is None:
            return None
        data = Plotly.DataByTopology(cluster)
        return Plotly.FigureByData(data)

    def FigureByRule(self, rule, silent: bool = False):
        """Returns a Plotly figure visualising the input rule."""
        if not isinstance(rule, dict) or "input" not in rule:
            if not silent:
                print("ShapeGrammar.FigureByRule - Error: The rule parameter is not a valid rule. Returning None.")
            return None
        input_topology = rule.get("input")
        output_topology = self.ApplyRule(input_topology, rule, silent=silent)
        if output_topology is None:
            return None
        return self.FigureByInputOutput(input_topology, output_topology, silent=silent)


__all__ = ["ShapeGrammar"]
