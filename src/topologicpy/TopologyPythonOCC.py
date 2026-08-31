# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.

"""
PythonOCC implementation of the public Topology class.

This class progressively replaces the legacy dual-backend implementation with
clean PythonOCC-first methods. During migration it temporarily inherits methods
that have not yet been rewritten from ``TopologyLegacy.Topology``.
"""

from __future__ import annotations

from typing import Any

from topologicpy.Core import Core
import topologicpy.TopologyLegacy as _topology_legacy_module


# Preserve the frozen legacy class before rebinding the legacy module-global
# ``Topology`` symbol at the end of this module.
TopologyLegacy = _topology_legacy_module.Topology


class Topology(TopologyLegacy):
    """
    Provides PythonOCC-first topology operations.

    Notes
    -----
    Methods not yet implemented in this class are temporarily inherited from
    ``TopologyLegacy.Topology``. This inheritance is a migration mechanism and
    will be removed when the PythonOCC implementation is complete.
    """

    _TYPE_IDS = {
        "vertex": 1,
        "edge": 2,
        "wire": 4,
        "face": 8,
        "shell": 16,
        "cell": 32,
        "cellcomplex": 64,
        "cluster": 128,
        "aperture": 256,
        "context": 512,
        "dictionary": 1024,
        "content": 8192,
        "graph": 2048,
        "tgraph": 2048,
        "topology": 4096,
    }

    _TYPE_RANKS = {
        "vertex": 0,
        "edge": 1,
        "wire": 2,
        "face": 3,
        "shell": 4,
        "cell": 5,
        "cellcomplex": 6,
        "cluster": 7,
    }

    _SUBTOPOLOGY_METHODS = {
        "vertex": "Vertices",
        "edge": "Edges",
        "wire": "Wires",
        "face": "Faces",
        "shell": "Shells",
        "cell": "Cells",
        "cellcomplex": "CellComplexes",
        "cluster": "Clusters",
        "aperture": "Apertures",
    }

    @staticmethod
    def _IsTGraph(value: Any) -> bool:
        """Returns True if the input object is a TGraph."""
        try:
            from topologicpy.TGraph import TGraph
            return isinstance(value, TGraph)
        except Exception:
            return False

    @staticmethod
    def _Deduplicate(topologies: list) -> list:
        """Returns identity-unique topologies while preserving input order."""
        result = []

        for topology in topologies or []:
            duplicate = False

            for existing in result:
                try:
                    if Topology.IsSame(topology, existing, silent=True):
                        duplicate = True
                        break
                except Exception:
                    if topology is existing:
                        duplicate = True
                        break

            if not duplicate:
                result.append(topology)

        return result

    # ---------------------------------------------------------------------
    # Type system
    # ---------------------------------------------------------------------

    @staticmethod
    def IsInstance(topology, type: str, silent: bool = True):
        """
        Returns True if the input object is an instance of the requested type.

        Parameters
        ----------
        topology : object
            The input object.
        type : str
            The requested type. Valid values are ``"vertex"``, ``"edge"``,
            ``"wire"``, ``"face"``, ``"shell"``, ``"cell"``,
            ``"cellcomplex"``, ``"cluster"``, ``"topology"``, ``"aperture"``,
            ``"context"``, ``"content"``, ``"dictionary"``, ``"graph"``, and ``"tgraph"``.
            The comparison is case insensitive.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is True.

        Returns
        -------
        bool
            True if the input object is an instance of the requested type.
            Returns None if ``type`` is invalid.
        """
        if not isinstance(type, str):
            if not silent:
                print(
                    "Topology.IsInstance - Error: The input type parameter is not "
                    "a valid string. Returning None."
                )
            return None

        requested = type.strip().lower()

        if requested not in Topology._TYPE_IDS:
            if not silent:
                print(
                    "Topology.IsInstance - Error: The input type parameter is not "
                    "a recognized type. Returning None."
                )
            return None

        # Content, Aperture, and Context are semantic-layer objects on the
        # PythonOCC path. They are intentionally not kernel topology wrappers.
        if requested in ("content", "aperture", "context"):
            try:
                from topologicpy.Content import Content
                from topologicpy.Aperture import Aperture
                from topologicpy.Context import Context

                if requested == "content" and isinstance(topology, Content):
                    return True
                if requested == "aperture" and isinstance(topology, Aperture):
                    return True
                if requested == "context" and isinstance(topology, Context):
                    return True
            except Exception:
                pass

            # Retain recognition of backend-native objects during the transition.
            namespace_name = {
                "aperture": "Aperture",
                "context": "Context",
            }.get(requested)
            if namespace_name is None:
                return False
            try:
                return isinstance(topology, Core.Namespace(namespace_name))
            except Exception:
                return False

        is_tgraph = Topology._IsTGraph(topology)

        if requested == "tgraph":
            return is_tgraph

        if requested == "graph":
            if is_tgraph:
                return True
            try:
                return isinstance(topology, Core.Namespace("Graph"))
            except Exception:
                return False

        namespace_names = {
            "vertex": "Vertex",
            "edge": "Edge",
            "wire": "Wire",
            "face": "Face",
            "shell": "Shell",
            "cell": "Cell",
            "cellcomplex": "CellComplex",
            "cluster": "Cluster",
            "topology": "Topology",
            "aperture": "Aperture",
            "context": "Context",
            "dictionary": "Dictionary",
        }

        namespace_name = namespace_names.get(requested)

        if namespace_name is None:
            return False

        try:
            return isinstance(topology, Core.Namespace(namespace_name))
        except Exception:
            return False

    @staticmethod
    def Type(topology, silent: bool = False):
        """
        Returns the numeric type identifier of the input topology.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        int
            The numeric topology type identifier, or None if the input is
            invalid.
        """
        if Topology.IsInstance(topology, "Aperture"):
            return Topology._TYPE_IDS["aperture"]
        if Topology.IsInstance(topology, "Context"):
            return Topology._TYPE_IDS["context"]
        if Topology.IsInstance(topology, "Content"):
            return Topology._TYPE_IDS["content"]

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.Type - Error: The input object is not a valid "
                    "topology or semantic object. Returning None."
                )
            return None

        try:
            result = Core.InstanceCall(topology, "Type")
        except Exception:
            result = None

        if isinstance(result, int):
            return result

        if not silent:
            print(
                "Topology.Type - Error: Could not determine the topology type. "
                "Returning None."
            )
        return None

    @staticmethod
    def TypeAsString(topology, silent: bool = False):
        """
        Returns the type of the input topology or graph as a string.

        Parameters
        ----------
        topology : topologicpy.Topology, topologicpy.Graph, or topologicpy.TGraph
            The input topology or graph.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        str
            The type name, or None if the input is invalid.
        """
        if Topology._IsTGraph(topology):
            return "TGraph"

        if Topology.IsInstance(topology, "Graph"):
            return "Graph"

        if Topology.IsInstance(topology, "Aperture"):
            return "Aperture"
        if Topology.IsInstance(topology, "Context"):
            return "Context"
        if Topology.IsInstance(topology, "Content"):
            return "Content"

        if Topology.IsInstance(topology, "Topology"):
            try:
                result = Core.InstanceCall(topology, "GetTypeAsString")
            except Exception:
                result = None

            if isinstance(result, str) and result:
                return result

        if not silent:
            print(
                "Topology.TypeAsString - Error: The input topology parameter is "
                "not a valid topology or graph. Returning None."
            )
        return None

    @staticmethod
    def TypeID(name: str = None, silent: bool = False) -> int:
        """
        Returns the numeric type identifier associated with the input type name.

        Parameters
        ----------
        name : str , optional
            The input type name. Valid values are ``"vertex"``, ``"edge"``,
            ``"wire"``, ``"face"``, ``"shell"``, ``"cell"``,
            ``"cellcomplex"``, ``"cluster"``, ``"aperture"``, ``"context"``,
            ``"dictionary"``, ``"content"``, ``"graph"``, ``"tgraph"``, and ``"topology"``.
            The comparison is case insensitive. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        int
            The numeric type identifier, or None if ``name`` is invalid.
        """
        if not isinstance(name, str):
            if not silent:
                print(
                    "Topology.TypeID - Error: The input name parameter is not a "
                    "valid string. Returning None."
                )
            return None

        result = Topology._TYPE_IDS.get(name.strip().lower())

        if result is None and not silent:
            print(
                "Topology.TypeID - Error: The input name parameter is not a "
                "recognized type. Returning None."
            )

        return result

    # ---------------------------------------------------------------------
    # OCCT shape conversion
    # ---------------------------------------------------------------------

    @staticmethod
    def ByOCCTShape(occtShape, ontology: bool = False, silent: bool = False):
        """
        Creates a topology from the input Open CASCADE shape.

        Parameters
        ----------
        occtShape : OCC.Core.TopoDS.TopoDS_Shape
            The input Open CASCADE shape.
        ontology : bool , optional
            If set to True, ontology metadata is added to the returned topology.
            Default is False.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        topologicpy.Topology
            The created topology, or None if the shape cannot be wrapped.
        """
        try:
            topology = Core.Call("Topology", "ByOcctShape", occtShape)
        except Exception:
            topology = None

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.ByOCCTShape - Error: Could not create a topology "
                    "from the input OCCT shape. Returning None."
                )
            return None

        return Topology._OntologyAnnotate(
            topology,
            ontology=ontology,
            generatedBy="Topology.ByOCCTShape",
            annotateSubtopologies=True,
            silent=True,
        )

    @staticmethod
    def OCCTShape(topology, silent: bool = False):
        """
        Returns the Open CASCADE shape of the input topology.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        OCC.Core.TopoDS.TopoDS_Shape
            The underlying Open CASCADE shape, or None if the input is invalid.
        """
        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.OCCTShape - Error: The input topology parameter is "
                    "not a valid topology. Returning None."
                )
            return None

        try:
            return Core.InstanceCall(topology, "GetOcctShape")
        except Exception:
            if not silent:
                print(
                    "Topology.OCCTShape - Error: Could not retrieve the OCCT "
                    "shape. Returning None."
                )
            return None

    # ---------------------------------------------------------------------
    # Identity
    # ---------------------------------------------------------------------

    @staticmethod
    def IsSame(topologyA, topologyB, silent: bool = False):
        """
        Returns True if the input objects represent the same topological entity.

        For PythonOCC topologies, identity is based on the underlying OCCT
        ``TopoDS_Shape.IsSame`` semantics. TGraphs use Python object identity.

        Parameters
        ----------
        topologyA : topologicpy.Topology, topologicpy.Graph, or topologicpy.TGraph
            The first input object.
        topologyB : topologicpy.Topology, topologicpy.Graph, or topologicpy.TGraph
            The second input object.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        bool
            True if the two inputs represent the same entity. False otherwise.
            Returns None if either input is invalid.
        """
        a_is_tgraph = Topology._IsTGraph(topologyA)
        b_is_tgraph = Topology._IsTGraph(topologyB)

        if a_is_tgraph or b_is_tgraph:
            if a_is_tgraph and b_is_tgraph:
                return topologyA is topologyB
            return False

        valid_types = (
            "Topology",
            "Aperture",
            "Graph",
        )

        if not any(
            Topology.IsInstance(topologyA, type_name)
            for type_name in valid_types
        ):
            if not silent:
                print(
                    "Topology.IsSame - Error: The input topologyA parameter is "
                    "not a valid topology or graph. Returning None."
                )
            return None

        if not any(
            Topology.IsInstance(topologyB, type_name)
            for type_name in valid_types
        ):
            if not silent:
                print(
                    "Topology.IsSame - Error: The input topologyB parameter is "
                    "not a valid topology or graph. Returning None."
                )
            return None

        try:
            return bool(Core.Call("Topology", "IsSame", topologyA, topologyB))
        except Exception:
            if not silent:
                print(
                    "Topology.IsSame - Error: Could not compare the input "
                    "topologies. Returning None."
                )
            return None

    # ---------------------------------------------------------------------
    # Subtopology accessors
    # ---------------------------------------------------------------------

    @staticmethod
    def Vertices(topology, silent: bool = True):
        """
        Returns the vertices of the input topology or graph.

        Parameters
        ----------
        topology : topologicpy.Topology, topologicpy.Graph, or topologicpy.TGraph
            The input topology or graph.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is True.

        Returns
        -------
        list
            The list of vertices, or None if the input is invalid or the query
            fails.
        """
        return Topology.SubTopologies(
            topology,
            subTopologyType="vertex",
            silent=silent,
        )

    @staticmethod
    def Edges(topology, silent: bool = False):
        """
        Returns the edges of the input topology or graph.

        Parameters
        ----------
        topology : topologicpy.Topology, topologicpy.Graph, or topologicpy.TGraph
            The input topology or graph.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        list
            The list of edges, or None if the input is invalid or the query
            fails.
        """
        return Topology.SubTopologies(
            topology,
            subTopologyType="edge",
            silent=silent,
        )

    @staticmethod
    def Wires(topology, silent: bool = False):
        """
        Returns the wires of the input topology.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        list
            The list of wires, or None if the input is invalid or the query
            fails.
        """
        return Topology.SubTopologies(
            topology,
            subTopologyType="wire",
            silent=silent,
        )

    @staticmethod
    def Faces(topology, silent: bool = False):
        """
        Returns the faces of the input topology.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        list
            The list of faces, or None if the input is invalid or the query
            fails.
        """
        return Topology.SubTopologies(
            topology,
            subTopologyType="face",
            silent=silent,
        )

    @staticmethod
    def Shells(topology, silent: bool = False):
        """
        Returns the shells of the input topology.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        list
            The list of shells, or None if the input is invalid or the query
            fails.
        """
        return Topology.SubTopologies(
            topology,
            subTopologyType="shell",
            silent=silent,
        )

    @staticmethod
    def Cells(topology, silent: bool = False):
        """
        Returns the cells of the input topology.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        list
            The list of cells, or None if the input is invalid or the query
            fails.
        """
        return Topology.SubTopologies(
            topology,
            subTopologyType="cell",
            silent=silent,
        )

    @staticmethod
    def CellComplexes(topology, silent: bool = False):
        """
        Returns the cell complexes of the input topology.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        list
            The list of cell complexes, or None if the input is invalid or the
            query fails.
        """
        return Topology.SubTopologies(
            topology,
            subTopologyType="cellcomplex",
            silent=silent,
        )

    @staticmethod
    def Clusters(topology, silent: bool = False):
        """
        Returns the clusters of the input topology.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        list
            The list of clusters, or None if the input is invalid or the query
            fails.
        """
        return Topology.SubTopologies(
            topology,
            subTopologyType="cluster",
            silent=silent,
        )

    @staticmethod
    def SubTopologies(
        topology,
        subTopologyType: str = "vertex",
        silent: bool = False,
    ):
        """
        Returns subtopologies of the requested type.

        Parameters
        ----------
        topology : topologicpy.Topology, topologicpy.Graph, or topologicpy.TGraph
            The input topology or graph.
        subTopologyType : str , optional
            The requested subtopology type. Valid values are ``"vertex"``,
            ``"edge"``, ``"wire"``, ``"face"``, ``"shell"``, ``"cell"``,
            ``"cellcomplex"``, ``"cluster"``, and ``"aperture"``. The
            comparison is case insensitive. Default is ``"vertex"``.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        list
            The requested subtopologies. A topology is considered a subtopology
            of itself. Returns an empty list when the requested dimensional type
            cannot occur below the input topology. Returns None if the input is
            invalid or the backend query fails.
        """
        if not isinstance(subTopologyType, str):
            if not silent:
                print(
                    "Topology.SubTopologies - Error: The input subTopologyType "
                    "parameter is not a valid string. Returning None."
                )
            return None

        requested = subTopologyType.strip().lower()

        if requested not in Topology._SUBTOPOLOGY_METHODS:
            if not silent:
                print(
                    "Topology.SubTopologies - Error: The input subTopologyType "
                    f"parameter '{subTopologyType}' is not recognized. Returning None."
                )
            return None

        # TGraph has only vertex and edge topology projections.
        if Topology._IsTGraph(topology):
            try:
                from topologicpy.TGraph import TGraph

                if requested == "vertex":
                    return TGraph.Vertices(topology)
                if requested == "edge":
                    return TGraph.Edges(topology)
            except Exception:
                if not silent:
                    print(
                        "Topology.SubTopologies - Error: Could not query the input "
                        "TGraph. Returning None."
                    )
                return None

            if not silent:
                print(
                    "Topology.SubTopologies - Error: The requested type is not a "
                    "valid TGraph subtopology. Returning None."
                )
            return None

        # Legacy Graph remains a public TopologicPy data type under PythonOCC.
        if Topology.IsInstance(topology, "Graph"):
            try:
                from topologicpy.Graph import Graph

                if requested == "vertex":
                    return Graph.Vertices(topology)
                if requested == "edge":
                    return Graph.Edges(topology)
            except Exception:
                if not silent:
                    print(
                        "Topology.SubTopologies - Error: Could not query the input "
                        "Graph. Returning None."
                    )
                return None

            if not silent:
                print(
                    "Topology.SubTopologies - Error: The requested type is not a "
                    "valid Graph subtopology. Returning None."
                )
            return None

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.SubTopologies - Error: The input topology parameter "
                    "is not a valid topology or graph. Returning None."
                )
            return None

        topology_type = Topology.TypeAsString(topology, silent=True)

        if not isinstance(topology_type, str):
            if not silent:
                print(
                    "Topology.SubTopologies - Error: Could not determine the type "
                    "of the input topology. Returning None."
                )
            return None

        topology_type = topology_type.strip().lower()

        if topology_type == requested:
            return [topology]

        # A lower-dimensional topology cannot contain a higher-dimensional one.
        if (
            topology_type in Topology._TYPE_RANKS
            and requested in Topology._TYPE_RANKS
            and Topology._TYPE_RANKS[topology_type]
            < Topology._TYPE_RANKS[requested]
        ):
            return []

        method_name = Topology._SUBTOPOLOGY_METHODS[requested]

        # Preserve TopologicPy's ordered Face boundary contract for Edges and
        # Vertices. OCCT's general explorer order is not a boundary walk order.
        if topology_type == "face" and requested in ("vertex", "edge"):
            try:
                wires = Core.InstanceCall(topology, "Wires")
            except Exception:
                wires = None

            if not isinstance(wires, list):
                if not silent:
                    print(
                        "Topology.SubTopologies - Error: Could not retrieve the "
                        "boundary wires of the input Face. Returning None."
                    )
                return None

            result = []

            for wire in wires:
                try:
                    items = Core.InstanceCall(wire, method_name)
                except Exception:
                    items = None

                if not isinstance(items, list):
                    if not silent:
                        print(
                            "Topology.SubTopologies - Error: Could not retrieve "
                            f"{requested}s from a Face boundary. Returning None."
                        )
                    return None

                result.extend(items)

            return result

        try:
            result = Core.InstanceCall(topology, method_name)
        except Exception:
            result = None

        if isinstance(result, list):
            return result

        if not silent:
            print(
                "Topology.SubTopologies - Error: The backend query failed. "
                "Returning None."
            )
        return None

    @staticmethod
    def SuperTopologies(
        topology,
        hostTopology,
        topologyType: str = None,
        silent: bool = False,
    ) -> list:
        """
        Returns supertopologies of the input topology within a host topology.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology whose ancestors are requested.
        hostTopology : topologicpy.Topology
            The host topology in which to search.
        topologyType : str , optional
            The requested supertopology type. Valid values are ``"edge"``,
            ``"wire"``, ``"face"``, ``"shell"``, ``"cell"``,
            ``"cellcomplex"``, and ``"cluster"``. If set to None, the next type
            in the standard topology hierarchy is used. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        list
            The matching supertopologies. Returns an empty list when no matching
            ancestors exist, or None if the inputs or requested type are invalid.
        """
        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.SuperTopologies - Error: The input topology parameter "
                    "is not a valid topology. Returning None."
                )
            return None

        if not Topology.IsInstance(hostTopology, "Topology"):
            if not silent:
                print(
                    "Topology.SuperTopologies - Error: The input hostTopology "
                    "parameter is not a valid topology. Returning None."
                )
            return None

        source_type = Topology.TypeAsString(topology, silent=True)
        host_type = Topology.TypeAsString(hostTopology, silent=True)

        if not isinstance(source_type, str) or not isinstance(host_type, str):
            if not silent:
                print(
                    "Topology.SuperTopologies - Error: Could not determine the "
                    "input topology types. Returning None."
                )
            return None

        source_type = source_type.strip().lower()
        host_type = host_type.strip().lower()

        if source_type not in Topology._TYPE_RANKS:
            if not silent:
                print(
                    "Topology.SuperTopologies - Error: The input topology type is "
                    "not supported. Returning None."
                )
            return None

        if topologyType is None:
            source_rank = Topology._TYPE_RANKS[source_type]
            target_rank = source_rank + 1
            target_type = next(
                (
                    name
                    for name, rank in Topology._TYPE_RANKS.items()
                    if rank == target_rank
                ),
                None,
            )
        else:
            if not isinstance(topologyType, str):
                if not silent:
                    print(
                        "Topology.SuperTopologies - Error: The input topologyType "
                        "parameter is not a valid string. Returning None."
                    )
                return None
            target_type = topologyType.strip().lower()

        if target_type not in Topology._TYPE_RANKS or target_type == "vertex":
            if not silent:
                print(
                    "Topology.SuperTopologies - Error: The input topologyType "
                    "parameter is not a valid supertopology type. Returning None."
                )
            return None

        source_rank = Topology._TYPE_RANKS[source_type]
        target_rank = Topology._TYPE_RANKS[target_type]

        if target_rank <= source_rank:
            if not silent:
                print(
                    "Topology.SuperTopologies - Error: The requested topologyType "
                    "is not higher-dimensional than the input topology. Returning None."
                )
            return None

        host_rank = Topology._TYPE_RANKS.get(host_type)

        if host_rank is not None and target_rank > host_rank:
            return []

        try:
            result = Core.InstanceCall(
                topology,
                "SuperTopologies",
                hostTopology,
                target_type,
            )
        except Exception:
            result = None

        if isinstance(result, list):
            return result

        if not silent:
            print(
                "Topology.SuperTopologies - Error: The backend query failed. "
                "Returning None."
            )
        return None

    # ---------------------------------------------------------------------
    # Dictionaries and relationships
    # ---------------------------------------------------------------------

    @staticmethod
    def Dictionary(topology, silent: bool = False):
        """
        Returns the dictionary of the input topology or graph.

        Parameters
        ----------
        topology : topologicpy.Topology, topologicpy.Graph, or topologicpy.TGraph
            The input topology or graph.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        topologicpy.Dictionary
            The dictionary of the input topology or graph, or None if the input
            is invalid.
        """
        if Topology._IsTGraph(topology):
            try:
                from topologicpy.Dictionary import Dictionary
                from topologicpy.TGraph import TGraph

                dictionary = TGraph.Dictionary(topology)

                if isinstance(dictionary, dict):
                    if len(dictionary) == 0:
                        return None
                    return Dictionary.ByKeysValues(
                        list(dictionary.keys()),
                        list(dictionary.values()),
                    )

                return dictionary
            except Exception:
                if not silent:
                    print(
                        "Topology.Dictionary - Error: Could not retrieve the "
                        "dictionary of the input TGraph. Returning None."
                    )
                return None

        if not (
            Topology.IsInstance(topology, "Topology")
            or Topology.IsInstance(topology, "Graph")
        ):
            if not silent:
                print(
                    "Topology.Dictionary - Error: The input topology parameter is "
                    "not a valid topology or graph. Returning None."
                )
            return None

        try:
            return Core.InstanceCall(topology, "GetDictionary")
        except Exception:
            if not silent:
                print(
                    "Topology.Dictionary - Error: Could not retrieve the input "
                    "dictionary. Returning None."
                )
            return None

    @staticmethod
    def SetDictionary(topology, dictionary, silent: bool = False):
        """
        Sets the dictionary of the input topology or graph.

        On the PythonOCC topology path, dictionaries are stored canonically as
        plain Python dictionaries in the shape-keyed backend AttributeManager.
        This avoids the lossy conversion through legacy Core attribute objects
        and preserves nested dictionaries, booleans, tuples/lists, and other
        supported Python semantic values exactly.

        Parameters
        ----------
        topology : topologicpy.Topology, topologicpy.Graph, topologicpy.TGraph, or dict
            The input topology, graph, TGraph, TGraph vertex record, or TGraph
            edge record.
        dictionary : topologicpy.Dictionary or dict
            The dictionary to assign.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        object
            The input object with the dictionary assigned. Returns None if the
            input object is invalid. If ``dictionary`` is invalid, the original
            input object is returned unchanged.
        """
        from topologicpy.Dictionary import Dictionary

        def dictionary_to_python(value):
            if isinstance(value, dict):
                return dict(value)

            try:
                from topologicpy.TGraph import TGraph
                result = TGraph._DictionaryToPython(value)
                if isinstance(result, dict):
                    return result
            except Exception:
                pass

            try:
                result = Dictionary.PythonDictionary(value, silent=True)
                if isinstance(result, dict):
                    return result
            except TypeError:
                try:
                    result = Dictionary.PythonDictionary(value)
                    if isinstance(result, dict):
                        return result
                except Exception:
                    pass
            except Exception:
                pass

            try:
                keys = Dictionary.Keys(value)
                if isinstance(keys, list):
                    return {
                        key: Dictionary.ValueAtKey(value, key, None)
                        for key in keys
                    }
            except TypeError:
                try:
                    keys = Dictionary.Keys(value)
                    if isinstance(keys, list):
                        return {
                            key: Dictionary.ValueAtKey(value, key)
                            for key in keys
                        }
                except Exception:
                    pass
            except Exception:
                pass

            return None

        def is_tgraph_vertex_record(value):
            return (
                isinstance(value, dict)
                and "index" in value
                and "src" not in value
                and "dst" not in value
            )

        def is_tgraph_edge_record(value):
            return (
                isinstance(value, dict)
                and "index" in value
                and "src" in value
                and "dst" in value
            )

        python_dictionary = dictionary_to_python(dictionary)

        if not isinstance(python_dictionary, dict):
            if not silent:
                print(
                    "Topology.SetDictionary - Warning: The input dictionary "
                    "parameter is not valid. Returning the original input."
                )
            return topology

        if Topology._IsTGraph(topology):
            try:
                topology.SetDictionary(python_dictionary)
                return topology
            except Exception:
                try:
                    topology._dictionary = dict(python_dictionary)
                    topology._invalidate_cache()
                    return topology
                except Exception:
                    if not silent:
                        print(
                            "Topology.SetDictionary - Error: Could not set the "
                            "dictionary of the input TGraph. Returning the original input."
                        )
                    return topology

        if is_tgraph_vertex_record(topology) or is_tgraph_edge_record(topology):
            topology["dictionary"] = dict(python_dictionary)

            topology["dictionary"].setdefault(
                "index",
                topology.get("index"),
            )

            if "active" in topology:
                topology["dictionary"].setdefault(
                    "active",
                    topology.get("active", True),
                )

            if is_tgraph_edge_record(topology):
                topology["dictionary"].setdefault("src", topology.get("src"))
                topology["dictionary"].setdefault("dst", topology.get("dst"))
                topology["dictionary"].setdefault("srcId", topology.get("src"))
                topology["dictionary"].setdefault("dstId", topology.get("dst"))
                topology["dictionary"].setdefault(
                    "directed",
                    topology.get("directed", False),
                )

            return topology

        is_topology = Topology.IsInstance(topology, "Topology")
        is_graph = Topology.IsInstance(topology, "Graph")

        if not (is_topology or is_graph):
            if not silent:
                print(
                    "Topology.SetDictionary - Error: The input topology parameter "
                    "is not a valid topology, graph, TGraph, or TGraph record. "
                    "Returning None."
                )
            return None

        # PythonOCC Topology wrappers already accept arbitrary Python dictionary
        # payloads and persist them by OCCT shape identity. Keep that exact Python
        # representation instead of round-tripping through legacy attribute classes.
        if is_topology:
            try:
                result = Core.InstanceCall(
                    topology,
                    "SetDictionary",
                    dict(python_dictionary),
                )
            except Exception:
                result = None

            if result is None:
                if not silent:
                    print(
                        "Topology.SetDictionary - Error: Could not set the dictionary. "
                        "Returning the original input."
                    )
                return topology

            return result

        # Graph remains on its existing backend dictionary contract for now.
        backend_dictionary = dictionary

        if isinstance(dictionary, dict):
            try:
                backend_dictionary = Dictionary.ByPythonDictionary(
                    dictionary,
                    silent=True,
                )
            except TypeError:
                try:
                    backend_dictionary = Dictionary.ByPythonDictionary(dictionary)
                except Exception:
                    backend_dictionary = None
            except Exception:
                backend_dictionary = None

            if backend_dictionary is None:
                try:
                    backend_dictionary = Dictionary.ByKeysValues(
                        list(dictionary.keys()),
                        list(dictionary.values()),
                    )
                except Exception:
                    backend_dictionary = None

        if backend_dictionary is None:
            if not silent:
                print(
                    "Topology.SetDictionary - Warning: Could not construct a "
                    "backend dictionary. Returning the original input."
                )
            return topology

        try:
            result = Core.InstanceCall(
                topology,
                "SetDictionary",
                backend_dictionary,
            )
        except Exception:
            result = None

        if result is None:
            if not silent:
                print(
                    "Topology.SetDictionary - Error: Could not set the dictionary. "
                    "Returning the original input."
                )
            return topology

        return result

    # ---------------------------------------------------------------------
    # Semantic Content / Context relationships
    # ---------------------------------------------------------------------

    @staticmethod
    def _RelationshipCandidates(topology, subTopologyType, allowed, silent=False):
        """Returns candidate host topologies for semantic attachment."""
        if subTopologyType is None:
            requested = "self"
        elif isinstance(subTopologyType, str):
            requested = subTopologyType.strip().lower() or "self"
        else:
            if not silent:
                print(
                    "Topology - Error: The input subTopologyType parameter is "
                    "not a valid string. Returning None."
                )
            return None

        if requested not in allowed:
            if not silent:
                print(
                    "Topology - Error: The input subTopologyType parameter is "
                    "not recognized. Returning None."
                )
            return None

        if requested == "self":
            return [topology]

        return Topology.SubTopologies(
            topology,
            subTopologyType=requested,
            silent=silent,
        )

    @staticmethod
    def _RelationshipHostForContent(content, candidates, tolerance=0.0001):
        """Returns the first candidate that geometrically contains content."""
        from topologicpy.Vertex import Vertex

        try:
            selector = Topology.InternalVertex(
                content, tolerance=tolerance, silent=True
            )
        except Exception:
            selector = None

        if not Topology.IsInstance(selector, "Vertex"):
            try:
                selector = Topology.Centroid(content, silent=True)
            except Exception:
                selector = None

        if not Topology.IsInstance(selector, "Vertex"):
            return None

        for candidate in candidates or []:
            try:
                if Vertex.IsInternal(
                    selector,
                    candidate,
                    tolerance=tolerance,
                    silent=True,
                ):
                    return candidate
            except Exception:
                continue
        return None

    @staticmethod
    def AddContent(
        topology,
        contents=None,
        subTopologyType: str = None,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Adds Content relationships to the input topology.

        Content topology identity is preserved: attaching a topology does not copy
        it. The same Content can therefore participate in multiple Contexts.
        Apertures are Contents and are included by :meth:`Topology.Contents`.
        """
        from topologicpy.SemanticManager import SemanticManager

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.AddContent - Error: The input topology parameter is "
                    "not a valid topology. Returning None."
                )
            return None

        if contents is None:
            if not silent:
                print(
                    "Topology.AddContent - Warning: The input contents parameter "
                    "is empty. Returning the input topology unmodified."
                )
            return topology

        if not isinstance(contents, list):
            contents = [contents]
        contents = [
            content for content in contents
            if Topology.IsInstance(content, "Topology")
        ]
        if not contents:
            if not silent:
                print(
                    "Topology.AddContent - Warning: The input contents parameter "
                    "does not contain valid topologies. Returning the input topology "
                    "unmodified."
                )
            return topology

        candidates = Topology._RelationshipCandidates(
            topology,
            subTopologyType,
            allowed=(
                "self", "cellcomplex", "cell", "shell",
                "face", "wire", "edge", "vertex",
            ),
            silent=silent,
        )
        if candidates is None:
            return None

        manager = SemanticManager.GetInstance()
        direct = (subTopologyType is None or (
            isinstance(subTopologyType, str)
            and (subTopologyType.strip() == "" or subTopologyType.strip().lower() == "self")
        ))

        for content in contents:
            host = candidates[0] if direct else Topology._RelationshipHostForContent(
                content, candidates, tolerance=tolerance
            )
            if host is None:
                continue
            manager.register(content, host, parameters=None)

        return topology

    @staticmethod
    def AddApertures(
        topology,
        apertures,
        exclusive=False,
        subTopologyType=None,
        tolerance=0.001,
        silent: bool = False,
    ):
        """Adds Aperture Content relationships to the input topology.

        Aperture is a specialised Content. The same represented topology can have
        several Contexts, for example one to a host Face and one to a room Cell.
        """
        from topologicpy.Dictionary import Dictionary
        from topologicpy.SemanticManager import SemanticManager

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.AddApertures - Error: The input topology parameter is "
                    "not a valid topology. Returning None."
                )
            return None
        if not apertures:
            if not silent:
                print(
                    "Topology.AddApertures - Warning: The input apertures parameter "
                    "is empty. Returning the input topology."
                )
            return topology
        if not isinstance(apertures, list):
            if not silent:
                print(
                    "Topology.AddApertures - Error: The input apertures parameter "
                    "is not a list. Returning None."
                )
            return None

        apertures = [
            aperture for aperture in apertures
            if Topology.IsInstance(aperture, "Topology")
        ]
        if not apertures:
            return topology

        candidates = Topology._RelationshipCandidates(
            topology,
            subTopologyType,
            allowed=("self", "cell", "face", "edge", "vertex"),
            silent=silent,
        )
        if candidates is None:
            return None

        manager = SemanticManager.GetInstance()
        direct = (subTopologyType is None or (
            isinstance(subTopologyType, str)
            and (subTopologyType.strip() == "" or subTopologyType.strip().lower() == "self")
        ))

        for aperture in apertures:
            # Preserve the long-standing public marker on the represented topology.
            dictionary = Topology.Dictionary(aperture, silent=True)
            try:
                dictionary = Dictionary.SetValueAtKey(dictionary, "type", "Aperture")
                marked = Topology.SetDictionary(aperture, dictionary, silent=True)
                if Topology.IsInstance(marked, "Topology"):
                    aperture = marked
            except Exception:
                pass

            if direct:
                host = candidates[0]
                if bool(exclusive) and manager.aperture_topologies_for_host(host):
                    continue
            else:
                eligible = [
                    candidate for candidate in candidates
                    if not bool(exclusive)
                    or not manager.aperture_topologies_for_host(candidate)
                ]
                host = Topology._RelationshipHostForContent(
                    aperture, eligible, tolerance=tolerance
                )

            if host is None:
                continue
            manager.register(
                aperture,
                host,
                aperture=True,
                parameters=None,
            )

        return topology

    @staticmethod
    def Contents(topology, silent: bool = False):
        """Returns Content topologies hosted by ``topology``.

        Apertures are a specialised Content and are therefore included.
        """
        from topologicpy.SemanticManager import SemanticManager

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.Contents - Error: The input topology parameter is "
                    "not a valid topology. Returning None."
                )
            return None
        return Topology._Deduplicate(
            SemanticManager.GetInstance().content_topologies_for_host(topology)
        )

    @staticmethod
    def Contexts(topology, silent: bool = False):
        """Returns all Context relationships of a Content topology."""
        from topologicpy.Content import Content
        from topologicpy.SemanticManager import SemanticManager

        if isinstance(topology, Content):
            return SemanticManager.GetInstance().contexts_for_content(topology)

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.Contexts - Error: The input topology parameter is "
                    "not a valid topology or Content. Returning None."
                )
            return None
        return SemanticManager.GetInstance().contexts_for_content(topology)

    @staticmethod
    def Apertures(topology, subTopologyType=None, silent: bool = False):
        """Returns Aperture Content topologies hosted by ``topology``."""
        from topologicpy.SemanticManager import SemanticManager

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.Apertures - Error: The input topology parameter is "
                    "not a valid topology. Returning None."
                )
            return None

        manager = SemanticManager.GetInstance()

        if subTopologyType is None:
            return Topology._Deduplicate(
                manager.aperture_topologies_for_host(topology)
            )

        if not isinstance(subTopologyType, str):
            if not silent:
                print(
                    "Topology.Apertures - Error: The input subTopologyType "
                    "parameter is not a valid string. Returning None."
                )
            return None

        requested = subTopologyType.strip().lower()
        if requested not in ("vertex", "edge", "face", "cell", "all"):
            if not silent:
                print(
                    "Topology.Apertures - Error: The input subTopologyType "
                    "parameter is not recognized. Returning None."
                )
            return None

        result = []
        if requested == "all":
            result.extend(manager.aperture_topologies_for_host(topology))
            requested_types = ("vertex", "edge", "face", "cell")
        else:
            requested_types = (requested,)

        for type_name in requested_types:
            subtopologies = Topology.SubTopologies(
                topology,
                subTopologyType=type_name,
                silent=True,
            )
            if subtopologies is None:
                if not silent:
                    print(
                        "Topology.Apertures - Error: Could not retrieve requested "
                        "subtopologies. Returning None."
                    )
                return None
            for subtopology in subtopologies:
                result.extend(manager.aperture_topologies_for_host(subtopology))

        return Topology._Deduplicate(result)

    @staticmethod
    def ApertureTopologies(
        topology,
        subTopologyType: str = None,
        silent: bool = False,
    ):
        """Compatibility alias for :meth:`Topology.Apertures`."""
        return Topology.Apertures(
            topology,
            subTopologyType=subTopologyType,
            silent=silent,
        )

    @staticmethod
    def RemoveContent(topology, contents, silent: bool = False):
        """Removes Contexts linking the specified Contents to ``topology``."""
        from topologicpy.Content import Content
        from topologicpy.SemanticManager import SemanticManager

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.RemoveContent - Error: The input topology parameter "
                    "is not a valid topology. Returning None."
                )
            return None
        if contents is None:
            return topology
        if not isinstance(contents, list):
            contents = [contents]
        contents = [
            item for item in contents
            if isinstance(item, Content) or Topology.IsInstance(item, "Topology")
        ]
        if not contents:
            return topology

        SemanticManager.GetInstance().remove(topology, contents=contents)
        return topology


    # ---------------------------------------------------------------------
    # Copy and transformations
    # ---------------------------------------------------------------------

    @staticmethod
    def Copy(topology, deep: bool = False, silent: bool = False):
        """
        Returns an independent copy of the input topology.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        deep : bool , optional
            If set to True, the native geometry and dictionaries attached to
            native subtopologies are deep-copied. If set to False, the native
            topology and its parent dictionary are copied. Default is False.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        topologicpy.Topology
            The copied topology, or None if the operation fails.
        """
        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.Copy - Error: The input topology parameter is not "
                    "a valid topology. Returning None."
                )
            return None

        method_name = "DeepCopy" if deep else "Copy"

        try:
            result = Core.InstanceCall(topology, method_name)
        except Exception as error:
            if not silent:
                print(
                    f"Topology.Copy - Error: The PythonOCC backend could not perform "
                    f"the {'deep' if deep else 'shallow'} copy. Returning None."
                )
                print("Error:", error)
            return None

        if not Topology.IsInstance(result, "Topology"):
            if not silent:
                print(
                    "Topology.Copy - Error: The PythonOCC backend returned an "
                    "invalid topology. Returning None."
                )
            return None

        source_type = Topology.TypeAsString(topology, silent=True)
        result_type = Topology.TypeAsString(result, silent=True)

        if source_type != result_type:
            if not silent:
                print(
                    "Topology.Copy - Error: The copied topology changed type from "
                    f"{source_type} to {result_type}. Returning None."
                )
            return None

        try:
            from topologicpy.SemanticManager import SemanticManager
            SemanticManager.GetInstance().transfer_topology(topology, result)
        except Exception:
            pass

        return result

    @staticmethod
    def DeepCopy(topology, silent: bool = False):
        """
        Returns an independent deep copy of the input topology.

        The native geometry is duplicated and dictionaries attached to native
        subtopologies are transferred to their exact copied counterparts.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        topologicpy.Topology
            The deep-copied topology, or None if the operation fails.
        """
        return Topology.Copy(topology, deep=True, silent=silent)

    @staticmethod
    def _ValidateAffineMatrix(matrix, tolerance: float = 0.0001, silent: bool = False):
        """Validates and returns a numeric 4x4 affine transformation matrix."""
        if (
            not isinstance(matrix, (list, tuple))
            or len(matrix) != 4
            or any(
                not isinstance(row, (list, tuple)) or len(row) != 4
                for row in matrix
            )
        ):
            if not silent:
                print(
                    "Topology.Transform - Error: The input matrix parameter is not "
                    "a valid 4x4 matrix. Returning None."
                )
            return None

        try:
            result = [
                [float(matrix[i][j]) for j in range(4)]
                for i in range(4)
            ]
            tol = abs(float(tolerance))
        except Exception:
            if not silent:
                print(
                    "Topology.Transform - Error: The input matrix or tolerance "
                    "parameter is not numeric. Returning None."
                )
            return None

        if (
            abs(result[3][0]) > tol
            or abs(result[3][1]) > tol
            or abs(result[3][2]) > tol
            or abs(result[3][3] - 1.0) > tol
        ):
            if not silent:
                print(
                    "Topology.Transform - Error: The input matrix is not a valid "
                    "affine transformation matrix. Returning None."
                )
            return None

        return result

    @staticmethod
    def _IsIdentityMatrix(matrix, tolerance: float = 0.0001) -> bool:
        """Returns True if the input 4x4 matrix is identity within tolerance."""
        identity = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        tol = abs(float(tolerance))
        return all(
            abs(float(matrix[i][j]) - identity[i][j]) <= tol
            for i in range(4)
            for j in range(4)
        )

    @staticmethod
    def _MultiplyMatrices(matrixA, matrixB):
        """Returns the row-major product of two 4x4 matrices."""
        return [
            [
                sum(float(matrixA[i][k]) * float(matrixB[k][j]) for k in range(4))
                for j in range(4)
            ]
            for i in range(4)
        ]

    @staticmethod
    def _RotationMatrix(origin, axis, angle: float, tolerance: float = 0.0001):
        """Returns a 4x4 axis-angle rotation matrix about the input origin."""
        import math
        from topologicpy.Vertex import Vertex

        try:
            ax, ay, az = [float(value) for value in axis]
            ox, oy, oz = [float(value) for value in Vertex.Coordinates(origin)]
            angle_radians = math.radians(float(angle))
        except Exception:
            return None

        magnitude = math.sqrt(ax * ax + ay * ay + az * az)
        if magnitude <= max(abs(float(tolerance)), 1.0e-15):
            return None

        ax /= magnitude
        ay /= magnitude
        az /= magnitude

        c = math.cos(angle_radians)
        s = math.sin(angle_radians)
        one_minus_c = 1.0 - c

        r00 = c + ax * ax * one_minus_c
        r01 = ax * ay * one_minus_c - az * s
        r02 = ax * az * one_minus_c + ay * s

        r10 = ay * ax * one_minus_c + az * s
        r11 = c + ay * ay * one_minus_c
        r12 = ay * az * one_minus_c - ax * s

        r20 = az * ax * one_minus_c - ay * s
        r21 = az * ay * one_minus_c + ax * s
        r22 = c + az * az * one_minus_c

        tx = ox - (r00 * ox + r01 * oy + r02 * oz)
        ty = oy - (r10 * ox + r11 * oy + r12 * oz)
        tz = oz - (r20 * ox + r21 * oy + r22 * oz)

        return [
            [r00, r01, r02, tx],
            [r10, r11, r12, ty],
            [r20, r21, r22, tz],
            [0.0, 0.0, 0.0, 1.0],
        ]

    @staticmethod
    def _FinalizeTransformation(
        source,
        result,
        operation: str = "Transform",
        silent: bool = False,
    ):
        """Validates a transformation result and preserves topology type."""
        if not Topology.IsInstance(result, "Topology"):
            if not silent:
                print(
                    f"Topology.{operation} - Error: The PythonOCC backend returned "
                    "an invalid topology. Returning None."
                )
            return None

        source_type = Topology.TypeAsString(source, silent=True)
        result_type = Topology.TypeAsString(result, silent=True)

        if source_type != result_type:
            if not silent:
                print(
                    f"Topology.{operation} - Error: The operation changed topology "
                    f"type from {source_type} to {result_type}. Returning None."
                )
            return None

        try:
            from topologicpy.SemanticManager import SemanticManager
            SemanticManager.GetInstance().transfer_topology(source, result)
        except Exception:
            pass

        return result

    @staticmethod
    def Transform(
        topology,
        matrix: list,
        angTolerance: float = 0.001,
        transferDictionaries: bool = True,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Transforms the input topology using a 4x4 affine transformation matrix.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        matrix : list
            A numeric 4x4 affine transformation matrix in row-major form.
        angTolerance : float , optional
            Retained for API compatibility. PythonOCC applies the complete
            affine matrix directly and does not decompose it into Euler
            rotations. Default is 0.001.
        transferDictionaries : bool , optional
            If set to True, dictionaries attached to the topology and its native
            subtopologies are transferred through OCCT shape history. Default
            is True.
        tolerance : float , optional
            The tolerance used when validating the affine matrix and identity
            transformation. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        topologicpy.Topology
            The transformed topology, or None if the operation fails.
        """
        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.Transform - Error: The input topology parameter is "
                    "not a valid topology. Returning None."
                )
            return None

        numeric_matrix = Topology._ValidateAffineMatrix(
            matrix,
            tolerance=tolerance,
            silent=silent,
        )
        if numeric_matrix is None:
            return None

        if Topology._IsIdentityMatrix(numeric_matrix, tolerance=tolerance):
            return topology

        try:
            result = Core.TopologyUtility.Transform(
                topology,
                numeric_matrix,
                bool(transferDictionaries),
            )
        except Exception as error:
            if not silent:
                print(
                    "Topology.Transform - Error: The PythonOCC affine "
                    "transformation failed. Returning None."
                )
                print("Error:", error)
            return None

        return Topology._FinalizeTransformation(
            topology,
            result,
            operation="Transform",
            silent=silent,
        )

    @staticmethod
    def Translate(
        topology,
        x=0,
        y=0,
        z=0,
        transferDictionaries: bool = True,
        silent: bool = False,
    ):
        """
        Translates the input topology.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        x : float , optional
            The translation distance along the X axis. Default is 0.
        y : float , optional
            The translation distance along the Y axis. Default is 0.
        z : float , optional
            The translation distance along the Z axis. Default is 0.
        transferDictionaries : bool , optional
            If set to True, dictionaries attached to the topology and its native
            subtopologies are transferred. Default is True.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        topologicpy.Topology
            The translated topology, or None if the operation fails.
        """
        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.Translate - Error: The input topology parameter is "
                    "not a valid topology. Returning None."
                )
            return None

        try:
            dx = float(x)
            dy = float(y)
            dz = float(z)
        except Exception:
            if not silent:
                print(
                    "Topology.Translate - Error: The x, y, and z parameters must "
                    "be numeric. Returning None."
                )
            return None

        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            return topology

        try:
            result = Core.TopologyUtility.Translate(
                topology,
                dx,
                dy,
                dz,
                bool(transferDictionaries),
            )
        except Exception as error:
            if not silent:
                print(
                    "Topology.Translate - Error: The PythonOCC translation "
                    "failed. Returning None."
                )
                print("Error:", error)
            return None

        return Topology._FinalizeTransformation(
            topology,
            result,
            operation="Translate",
            silent=silent,
        )

    @staticmethod
    def Move(
        topology,
        x=0,
        y=0,
        z=0,
        transferDictionaries: bool = True,
        silent: bool = False,
    ):
        """
        Moves the input topology by the specified Cartesian offsets.

        This method is an alias of :meth:`Topology.Translate`.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        x : float , optional
            The translation distance along the X axis. Default is 0.
        y : float , optional
            The translation distance along the Y axis. Default is 0.
        z : float , optional
            The translation distance along the Z axis. Default is 0.
        transferDictionaries : bool , optional
            If set to True, dictionaries attached to the topology and its native
            subtopologies are transferred. Default is True.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        topologicpy.Topology
            The moved topology, or None if the operation fails.
        """
        return Topology.Translate(
            topology,
            x=x,
            y=y,
            z=z,
            transferDictionaries=transferDictionaries,
            silent=silent,
        )

    @staticmethod
    def Rotate(
        topology,
        origin=None,
        axis=None,
        angle: float = 0,
        angTolerance: float = 0.001,
        transferDictionaries: bool = True,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Rotates the input topology about an arbitrary axis.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        origin : topologicpy.Vertex , optional
            The rotation origin. If set to None, the world origin is used.
            Default is None.
        axis : list , optional
            The axis of rotation as ``[x, y, z]``. Default is ``[0, 0, 1]``.
        angle : float , optional
            The rotation angle in degrees. Default is 0.
        angTolerance : float , optional
            Angles with absolute value below this threshold are treated as a
            no-op. Default is 0.001 degrees.
        transferDictionaries : bool , optional
            If set to True, dictionaries attached to the topology and its native
            subtopologies are transferred. Default is True.
        tolerance : float , optional
            The tolerance used to validate the rotation axis. Default is
            0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        topologicpy.Topology
            The rotated topology, or None if the operation fails.
        """
        import math
        from topologicpy.Vertex import Vertex

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.Rotate - Error: The input topology parameter is not "
                    "a valid topology. Returning None."
                )
            return None

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()

        if axis is None:
            axis = [0.0, 0.0, 1.0]

        if not isinstance(axis, (list, tuple)) or len(axis) != 3:
            if not silent:
                print(
                    "Topology.Rotate - Error: The input axis parameter must contain "
                    "exactly three numeric values. Returning None."
                )
            return None

        try:
            ax, ay, az = [float(value) for value in axis]
            angle_value = float(angle)
            angle_tolerance = abs(float(angTolerance))
            tol = abs(float(tolerance))
        except Exception:
            if not silent:
                print(
                    "Topology.Rotate - Error: The axis, angle, angTolerance, and "
                    "tolerance parameters must be numeric. Returning None."
                )
            return None

        if math.sqrt(ax * ax + ay * ay + az * az) <= max(tol, 1.0e-15):
            if not silent:
                print(
                    "Topology.Rotate - Error: The input axis has zero magnitude. "
                    "Returning None."
                )
            return None

        if abs(angle_value) < angle_tolerance:
            return topology

        try:
            result = Core.TopologyUtility.Rotate(
                topology,
                origin,
                ax,
                ay,
                az,
                angle_value,
                bool(transferDictionaries),
            )
        except Exception as error:
            if not silent:
                print(
                    "Topology.Rotate - Error: The PythonOCC rotation failed. "
                    "Returning None."
                )
                print("Error:", error)
            return None

        return Topology._FinalizeTransformation(
            topology,
            result,
            operation="Rotate",
            silent=silent,
        )

    @staticmethod
    def Scale(
        topology,
        origin=None,
        x=1,
        y=1,
        z=1,
        transferDictionaries: bool = True,
        silent: bool = False,
    ):
        """
        Scales the input topology about the specified origin.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        origin : topologicpy.Vertex , optional
            The scaling origin. If set to None, the world origin is used.
            Default is None.
        x : float , optional
            The X scale factor. Default is 1.
        y : float , optional
            The Y scale factor. Default is 1.
        z : float , optional
            The Z scale factor. Default is 1.
        transferDictionaries : bool , optional
            If set to True, dictionaries attached to the topology and its native
            subtopologies are transferred. Default is True.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        topologicpy.Topology
            The scaled topology, or None if the operation fails.
        """
        from topologicpy.Vertex import Vertex

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.Scale - Error: The input topology parameter is not "
                    "a valid topology. Returning None."
                )
            return None

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()

        try:
            sx = float(x)
            sy = float(y)
            sz = float(z)
        except Exception:
            if not silent:
                print(
                    "Topology.Scale - Error: The x, y, and z scale factors must "
                    "be numeric. Returning None."
                )
            return None

        if sx == 1.0 and sy == 1.0 and sz == 1.0:
            return topology

        for name, value in (("x", sx), ("y", sy), ("z", sz)):
            if abs(value) <= 1.0e-5 and not silent:
                print(
                    f"Topology.Scale - Warning: The {name} scale factor is close "
                    "to zero and may produce a degenerate topology."
                )

        try:
            result = Core.TopologyUtility.Scale(
                topology,
                origin,
                sx,
                sy,
                sz,
                bool(transferDictionaries),
            )
        except Exception as error:
            if not silent:
                print(
                    "Topology.Scale - Error: The PythonOCC scaling operation "
                    "failed. Returning None."
                )
                print("Error:", error)
            return None

        return Topology._FinalizeTransformation(
            topology,
            result,
            operation="Scale",
            silent=silent,
        )

    @staticmethod
    def Place(
        topology,
        originA=None,
        originB=None,
        mantissa: int = 6,
        silent: bool = False,
    ):
        """
        Places the input topology by mapping originA to originB.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        originA : topologicpy.Vertex , optional
            The source location. If set to None, the centroid of the topology is
            used. Default is None.
        originB : topologicpy.Vertex , optional
            The target location. If set to None, the world origin is used.
            Default is None.
        mantissa : int , optional
            The number of decimal places used when reading the origin
            coordinates. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        topologicpy.Topology
            The placed topology, or None if the operation fails.
        """
        from topologicpy.Vertex import Vertex

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.Place - Error: The input topology parameter is not "
                    "a valid topology. Returning None."
                )
            return None

        if not Topology.IsInstance(originA, "Vertex"):
            originA = Topology.Centroid(topology)
        if not Topology.IsInstance(originA, "Vertex"):
            if not silent:
                print(
                    "Topology.Place - Error: Could not determine originA. "
                    "Returning None."
                )
            return None

        if not Topology.IsInstance(originB, "Vertex"):
            originB = Vertex.Origin()

        try:
            dx = Vertex.X(originB, mantissa=mantissa) - Vertex.X(originA, mantissa=mantissa)
            dy = Vertex.Y(originB, mantissa=mantissa) - Vertex.Y(originA, mantissa=mantissa)
            dz = Vertex.Z(originB, mantissa=mantissa) - Vertex.Z(originA, mantissa=mantissa)
        except Exception:
            if not silent:
                print(
                    "Topology.Place - Error: Could not determine origin coordinates. "
                    "Returning None."
                )
            return None

        return Topology.Translate(topology, x=dx, y=dy, z=dz, silent=silent)

    @staticmethod
    def Orient(
        topology,
        origin=None,
        dirA=None,
        dirB=None,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Orients the input topology such that dirA is aligned with dirB.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        origin : topologicpy.Vertex , optional
            The rotation origin. If set to None, the centroid of the topology is
            used. Default is None.
        dirA : list , optional
            The source direction vector. Default is ``[0, 0, 1]``.
        dirB : list , optional
            The target direction vector. Default is ``[0, 0, 1]``.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        topologicpy.Topology
            The oriented topology, or None if the operation fails.
        """
        return Topology.OrientAndPlace(
            topology,
            originA=origin,
            originB=origin,
            dirA=[0.0, 0.0, 1.0] if dirA is None else dirA,
            dirB=[0.0, 0.0, 1.0] if dirB is None else dirB,
            transferDictionaries=True,
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def OrientAndPlace(
        topology,
        originA=None,
        originB=None,
        dirA=None,
        dirB=None,
        transferDictionaries: bool = True,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Orients and places the input topology using one affine transformation.

        The source origin is mapped exactly to the target origin while dirA is
        aligned with dirB.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        originA : topologicpy.Vertex , optional
            The source origin. If set to None, the centroid of the topology is
            used. Default is None.
        originB : topologicpy.Vertex , optional
            The target origin. If set to None, originA is retained. Default is
            None.
        dirA : list , optional
            The source direction vector. Default is ``[0, 0, 1]``.
        dirB : list , optional
            The target direction vector. Default is ``[0, 0, 1]``.
        transferDictionaries : bool , optional
            If set to True, dictionaries are transferred through OCCT shape
            history. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        topologicpy.Topology
            The oriented and placed topology, or None if the operation fails.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Vector import Vector

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.OrientAndPlace - Error: The input topology parameter "
                    "is not a valid topology. Returning None."
                )
            return None

        if dirA is None:
            dirA = [0.0, 0.0, 1.0]
        if dirB is None:
            dirB = [0.0, 0.0, 1.0]

        if not Topology.IsInstance(originA, "Vertex"):
            originA = Topology.Centroid(topology)
        if not Topology.IsInstance(originA, "Vertex"):
            if not silent:
                print(
                    "Topology.OrientAndPlace - Error: Could not determine originA. "
                    "Returning None."
                )
            return None

        if not Topology.IsInstance(originB, "Vertex"):
            originB = originA

        try:
            source = [float(value) for value in dirA]
            target = [float(value) for value in dirB]
            if len(source) != 3 or len(target) != 3:
                raise ValueError
            px, py, pz = [float(value) for value in Vertex.Coordinates(originA)]
            qx, qy, qz = [float(value) for value in Vertex.Coordinates(originB)]
        except Exception:
            if not silent:
                print(
                    "Topology.OrientAndPlace - Error: The direction vectors or "
                    "origin coordinates are invalid. Returning None."
                )
            return None

        rotation = Vector.TransformationMatrix(source, target)
        if (
            not isinstance(rotation, (list, tuple))
            or len(rotation) != 4
            or any(not isinstance(row, (list, tuple)) or len(row) != 4 for row in rotation)
        ):
            if not silent:
                print(
                    "Topology.OrientAndPlace - Error: Could not compute the "
                    "orientation matrix. Returning None."
                )
            return None

        try:
            r00, r01, r02 = [float(value) for value in rotation[0][:3]]
            r10, r11, r12 = [float(value) for value in rotation[1][:3]]
            r20, r21, r22 = [float(value) for value in rotation[2][:3]]
        except Exception:
            return None

        tx = qx - (r00 * px + r01 * py + r02 * pz)
        ty = qy - (r10 * px + r11 * py + r12 * pz)
        tz = qz - (r20 * px + r21 * py + r22 * pz)

        matrix = [
            [r00, r01, r02, tx],
            [r10, r11, r12, ty],
            [r20, r21, r22, tz],
            [0.0, 0.0, 0.0, 1.0],
        ]

        return Topology.Transform(
            topology,
            matrix,
            transferDictionaries=transferDictionaries,
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def Flatten(
        topology,
        origin=None,
        direction=None,
        transferDictionaries: bool = True,
        mantissa: int = 6,
        silent: bool = False,
    ):
        """
        Flattens the input topology to the XY plane.

        The input origin is mapped to the world origin and the input direction
        is aligned with the positive Z axis using one affine transformation.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        origin : topologicpy.Vertex , optional
            The origin to map to ``[0, 0, 0]``. If set to None, the centroid of
            the topology is used. Default is None.
        direction : list , optional
            The direction to align with the positive Z axis. Default is
            ``[0, 0, 1]``.
        transferDictionaries : bool , optional
            If set to True, dictionaries are transferred. Default is True.
        mantissa : int , optional
            The number of decimal places used when reading the origin
            coordinates. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        topologicpy.Topology
            The flattened topology, or None if the operation fails.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Vector import Vector

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.Flatten - Error: The input topology parameter is not "
                    "a valid topology. Returning None."
                )
            return None

        if direction is None:
            direction = [0.0, 0.0, 1.0]

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Topology.Centroid(topology)
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print(
                    "Topology.Flatten - Error: Could not determine the flattening "
                    "origin. Returning None."
                )
            return None

        rotation = Vector.TransformationMatrix(direction, Vector.Up())
        if not isinstance(rotation, (list, tuple)) or len(rotation) != 4:
            if not silent:
                print(
                    "Topology.Flatten - Error: Could not compute the flattening "
                    "matrix. Returning None."
                )
            return None

        try:
            px = float(Vertex.X(origin, mantissa=mantissa))
            py = float(Vertex.Y(origin, mantissa=mantissa))
            pz = float(Vertex.Z(origin, mantissa=mantissa))
            r = [[float(value) for value in row] for row in rotation]
        except Exception:
            return None

        # R * T(-origin): translation is -R*origin.
        r[0][3] = -(r[0][0] * px + r[0][1] * py + r[0][2] * pz)
        r[1][3] = -(r[1][0] * px + r[1][1] * py + r[1][2] * pz)
        r[2][3] = -(r[2][0] * px + r[2][1] * py + r[2][2] * pz)
        r[3] = [0.0, 0.0, 0.0, 1.0]

        return Topology.Transform(
            topology,
            r,
            transferDictionaries=transferDictionaries,
            silent=silent,
        )

    @staticmethod
    def Unflatten(
        topology,
        origin=None,
        direction=None,
        transferDictionaries: bool = True,
        silent: bool = False,
    ):
        """
        Unflattens the input topology.

        The positive Z axis is aligned with the input direction and the world
        origin is mapped to the input origin using one affine transformation.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        origin : topologicpy.Vertex , optional
            The target origin. If set to None, the world origin is used. Default
            is None.
        direction : list , optional
            The direction with which to align the positive Z axis. Default is
            ``[0, 0, 1]``.
        transferDictionaries : bool , optional
            If set to True, dictionaries are transferred. Default is True.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        topologicpy.Topology
            The unflattened topology, or None if the operation fails.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Vector import Vector

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.Unflatten - Error: The input topology parameter is "
                    "not a valid topology. Returning None."
                )
            return None

        if direction is None:
            direction = [0.0, 0.0, 1.0]

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()

        rotation = Vector.TransformationMatrix(Vector.Up(), direction)
        if not isinstance(rotation, (list, tuple)) or len(rotation) != 4:
            if not silent:
                print(
                    "Topology.Unflatten - Error: Could not compute the "
                    "unflattening matrix. Returning None."
                )
            return None

        try:
            matrix = [[float(value) for value in row] for row in rotation]
            ox, oy, oz = [float(value) for value in Vertex.Coordinates(origin)]
        except Exception:
            return None

        matrix[0][3] = ox
        matrix[1][3] = oy
        matrix[2][3] = oz
        matrix[3] = [0.0, 0.0, 0.0, 1.0]

        return Topology.Transform(
            topology,
            matrix,
            transferDictionaries=transferDictionaries,
            silent=silent,
        )

    @staticmethod
    def RotateByEulerAngles(
        topology,
        origin=None,
        roll: float = 0,
        pitch: float = 0,
        yaw: float = 0,
        transferDictionaries: bool = True,
        angTolerance: float = 0.001,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Rotates the input topology using Euler angles.

        Roll, pitch, and yaw are rotations about the X, Y, and Z axes,
        respectively. They are applied in that order about the same origin and
        combined into one affine transformation.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        origin : topologicpy.Vertex , optional
            The rotation origin. If set to None, the world origin is used.
            Default is None.
        roll : float , optional
            Rotation about the X axis in degrees. Default is 0.
        pitch : float , optional
            Rotation about the Y axis in degrees. Default is 0.
        yaw : float , optional
            Rotation about the Z axis in degrees. Default is 0.
        transferDictionaries : bool , optional
            If set to True, dictionaries are transferred. Default is True.
        angTolerance : float , optional
            Rotation magnitudes below this threshold are ignored. Default is
            0.001 degrees.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        topologicpy.Topology
            The rotated topology, or None if the operation fails.
        """
        from topologicpy.Vertex import Vertex

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.RotateByEulerAngles - Error: The input topology "
                    "parameter is not a valid topology. Returning None."
                )
            return None

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()

        try:
            angles = [float(roll), float(pitch), float(yaw)]
        except Exception:
            if not silent:
                print(
                    "Topology.RotateByEulerAngles - Error: The roll, pitch, and "
                    "yaw parameters must be numeric. Returning None."
                )
            return None

        matrices = []
        for axis, angle_value in zip(
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            angles,
        ):
            if abs(angle_value) < abs(float(angTolerance)):
                continue
            matrix = Topology._RotationMatrix(
                origin,
                axis,
                angle_value,
                tolerance=tolerance,
            )
            if matrix is None:
                return None
            matrices.append(matrix)

        if len(matrices) == 0:
            return topology

        combined = matrices[0]
        for matrix in matrices[1:]:
            # Sequential application M1 then M2 -> M2 * M1.
            combined = Topology._MultiplyMatrices(matrix, combined)

        return Topology.Transform(
            topology,
            combined,
            transferDictionaries=transferDictionaries,
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def RotateByQuaternion(
        topology,
        origin=None,
        quaternion=None,
        transferDictionaries: bool = False,
        angTolerance: float = 0.001,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Rotates the input topology using a quaternion.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        origin : topologicpy.Vertex , optional
            The rotation origin. If set to None, the world origin is used.
            Default is None.
        quaternion : list , optional
            The quaternion in ``[x, y, z, w]`` order. Default is
            ``[0, 0, 0, 1]``.
        transferDictionaries : bool , optional
            If set to True, dictionaries are transferred. Default is False.
        angTolerance : float , optional
            Rotation magnitudes below this threshold are ignored. Default is
            0.001 degrees.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default
            is False.

        Returns
        -------
        topologicpy.Topology
            The rotated topology, or None if the operation fails.
        """
        import math
        from topologicpy.Vertex import Vertex

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.RotateByQuaternion - Error: The input topology "
                    "parameter is not a valid topology. Returning None."
                )
            return None

        if quaternion is None:
            quaternion = [0.0, 0.0, 0.0, 1.0]

        if not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4:
            if not silent:
                print(
                    "Topology.RotateByQuaternion - Error: The quaternion must "
                    "contain exactly four numeric values. Returning None."
                )
            return None

        try:
            qx, qy, qz, qw = [float(value) for value in quaternion]
        except Exception:
            return None

        magnitude = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if magnitude <= max(abs(float(tolerance)), 1.0e-15):
            if not silent:
                print(
                    "Topology.RotateByQuaternion - Error: The quaternion has zero "
                    "magnitude. Returning None."
                )
            return None

        qx /= magnitude
        qy /= magnitude
        qz /= magnitude
        qw /= magnitude

        angle = 2.0 * math.degrees(math.acos(max(-1.0, min(1.0, qw))))
        sin_half = math.sqrt(max(0.0, 1.0 - qw * qw))

        if angle < abs(float(angTolerance)):
            return topology

        if sin_half <= 1.0e-15:
            axis = [1.0, 0.0, 0.0]
        else:
            axis = [qx / sin_half, qy / sin_half, qz / sin_half]

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()

        matrix = Topology._RotationMatrix(
            origin,
            axis,
            angle,
            tolerance=tolerance,
        )
        if matrix is None:
            return None

        return Topology.Transform(
            topology,
            matrix,
            transferDictionaries=transferDictionaries,
            tolerance=tolerance,
            silent=silent,
        )

    # -------------------------------------------------------------------------
    # Boolean operations
    # -------------------------------------------------------------------------

    @staticmethod
    def _Boolean(
        topologyA,
        topologyB,
        operation: str = "union",
        tranDict: bool = False,
        ontology: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Executes a Boolean operation using the PythonOCC backend.

        This internal dispatcher keeps the established backend Boolean
        algorithms and result normalization unchanged. Public callers should
        use Union, Difference, Intersect, SymmetricDifference, Merge, Slice,
        Impose, or Imprint.

        Parameters
        ----------
        topologyA : topologicpy.Topology
            The first input topology.
        topologyB : topologicpy.Topology
            The second input topology.
        operation : str , optional
            The Boolean operation. Valid values are "union", "difference",
            "intersect", "symdif", "merge", "slice", "impose", and
            "imprint". Default is "union".
        tranDict : bool , optional
            If set to True, dictionaries are transferred to the result.
            Default is False.
        ontology : bool , optional
            If set to True, ontology metadata is added to the result.
            Default is False.
        tolerance : float , optional
            The desired tolerance. It is retained for API compatibility and
            dictionary-transfer matching. The established PythonOCC Boolean
            kernels currently use their validated native tolerances. Default
            is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologicpy.Topology
            The Boolean result, or None if the operation has no result or
            fails.
        """
        if not Topology.IsInstance(topologyA, "Topology"):
            if not silent:
                print(
                    "Topology._Boolean - Error: The input topologyA parameter "
                    "is not a valid topology. Returning None."
                )
            return None

        if not Topology.IsInstance(topologyB, "Topology"):
            if not silent:
                print(
                    "Topology._Boolean - Error: The input topologyB parameter "
                    "is not a valid topology. Returning None."
                )
            return None

        if not isinstance(operation, str):
            if not silent:
                print(
                    "Topology._Boolean - Error: The input operation parameter "
                    "is not a valid string. Returning None."
                )
            return None

        operation = operation.strip().lower()
        aliases = {
            "intersection": "intersect",
            "xor": "symdif",
            "symmetricdifference": "symdif",
            "symmetric_difference": "symdif",
        }
        operation = aliases.get(operation, operation)

        backend_methods = {
            "union": "Union",
            "difference": "Difference",
            "intersect": "Intersect",
            "symdif": "XOR",
            "merge": "Merge",
            "slice": "Slice",
            "impose": "Impose",
            "imprint": "Imprint",
        }

        method_name = backend_methods.get(operation)
        if method_name is None:
            if not silent:
                print(
                    "Topology._Boolean - Error: The input operation parameter "
                    "is not recognized. Returning None."
                )
            return None

        if not isinstance(tranDict, bool):
            if not silent:
                print(
                    f"Topology.{method_name} - Error: The input tranDict "
                    "parameter is not a valid boolean. Returning None."
                )
            return None

        # Preserve the existing, fully tested dictionary-transfer and ontology
        # semantics while those two concerns are migrated separately. The
        # normal Boolean path below is entirely PythonOCC-native and no longer
        # passes through the legacy 629-line dispatcher.
        if tranDict or ontology:
            return TopologyLegacy._Boolean(
                topologyA,
                topologyB,
                operation=operation,
                tranDict=tranDict,
                ontology=ontology,
                tolerance=tolerance,
                silent=silent,
            )

        if topologyA == topologyB:
            if operation in ("difference", "symdif"):
                return None
            return topologyA

        try:
            return Core.InstanceCall(
                topologyA,
                method_name,
                topologyB,
                False,
            )
        except Exception as error:
            if not silent:
                print(
                    f"Topology.{method_name} - Error: The PythonOCC Boolean "
                    f"operation failed ({error}). Returning None."
                )
            return None

    @staticmethod
    def Union(
        topologyA,
        topologyB,
        tranDict: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Returns the Boolean union of the input topologies."""
        valid_a = Topology.IsInstance(topologyA, "Topology")
        valid_b = Topology.IsInstance(topologyB, "Topology")

        if not valid_a and not valid_b:
            if not silent:
                print(
                    "Topology.Union - Error: The input topologyA and topologyB "
                    "parameters are not valid topologies. Returning None."
                )
            return None
        if not valid_a:
            if not silent:
                print(
                    "Topology.Union - Warning: The input topologyA parameter is "
                    "not a valid topology. Returning topologyB."
                )
            return topologyB
        if not valid_b:
            if not silent:
                print(
                    "Topology.Union - Warning: The input topologyB parameter is "
                    "not a valid topology. Returning topologyA."
                )
            return topologyA

        return Topology._Boolean(
            topologyA,
            topologyB,
            operation="union",
            tranDict=tranDict,
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def Difference(
        topologyA,
        topologyB,
        tranDict: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Subtracts topologyB from topologyA."""
        return Topology._Boolean(
            topologyA,
            topologyB,
            operation="difference",
            tranDict=tranDict,
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def Intersect(
        topologyA,
        topologyB,
        tranDict: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Returns the Boolean intersection of the input topologies."""
        return Topology._Boolean(
            topologyA,
            topologyB,
            operation="intersect",
            tranDict=tranDict,
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def SymmetricDifference(
        topologyA,
        topologyB,
        tranDict: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Returns the symmetric difference of the input topologies."""
        return Topology._Boolean(
            topologyA,
            topologyB,
            operation="symdif",
            tranDict=tranDict,
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def XOR(
        topologyA,
        topologyB,
        tranDict: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Returns the symmetric difference of the input topologies."""
        return Topology.SymmetricDifference(
            topologyA,
            topologyB,
            tranDict=tranDict,
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def Merge(
        topologyA,
        topologyB,
        tranDict: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Merges the input topologies while preserving shared interfaces."""
        return Topology._Boolean(
            topologyA,
            topologyB,
            operation="merge",
            tranDict=tranDict,
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def Slice(
        topologyA,
        topologyB,
        tranDict: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Slices topologyA using topologyB."""
        valid_a = Topology.IsInstance(topologyA, "Topology")
        valid_b = Topology.IsInstance(topologyB, "Topology")

        if not valid_a and not valid_b:
            if not silent:
                print(
                    "Topology.Slice - Error: The input topologyA and topologyB "
                    "parameters are not valid topologies. Returning None."
                )
            return None
        if not valid_a:
            if not silent:
                print(
                    "Topology.Slice - Error: The input topologyA parameter is "
                    "not a valid topology. Returning None."
                )
            return topologyA
        if not valid_b:
            if not silent:
                print(
                    "Topology.Slice - Warning: The input topologyB parameter is "
                    "not a valid topology. Returning topologyA."
                )
            return topologyA

        return Topology._Boolean(
            topologyA,
            topologyB,
            operation="slice",
            tranDict=tranDict,
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def Impose(
        topologyA,
        topologyB,
        tranDict: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Imposes topologyB on topologyA."""
        valid_a = Topology.IsInstance(topologyA, "Topology")
        valid_b = Topology.IsInstance(topologyB, "Topology")

        if not valid_a and not valid_b:
            if not silent:
                print(
                    "Topology.Impose - Error: The input topologyA and topologyB "
                    "parameters are not valid topologies. Returning None."
                )
            return None
        if not valid_a:
            if not silent:
                print(
                    "Topology.Impose - Error: The input topologyA parameter is "
                    "not a valid topology. Returning None."
                )
            return topologyA
        if not valid_b:
            if not silent:
                print(
                    "Topology.Impose - Warning: The input topologyB parameter is "
                    "not a valid topology. Returning topologyA."
                )
            return topologyA

        return Topology._Boolean(
            topologyA,
            topologyB,
            operation="impose",
            tranDict=tranDict,
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def Imprint(
        topologyA,
        topologyB,
        tranDict: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Imprints topologyB on topologyA."""
        valid_a = Topology.IsInstance(topologyA, "Topology")
        valid_b = Topology.IsInstance(topologyB, "Topology")

        if not valid_a and not valid_b:
            if not silent:
                print(
                    "Topology.Imprint - Error: The input topologyA and topologyB "
                    "parameters are not valid topologies. Returning None."
                )
            return None
        if not valid_a:
            if not silent:
                print(
                    "Topology.Imprint - Error: The input topologyA parameter is "
                    "not a valid topology. Returning None."
                )
            return topologyA
        if not valid_b:
            if not silent:
                print(
                    "Topology.Imprint - Warning: The input topologyB parameter is "
                    "not a valid topology. Returning topologyA."
                )
            return topologyA

        return Topology._Boolean(
            topologyA,
            topologyB,
            operation="imprint",
            tranDict=tranDict,
            tolerance=tolerance,
            silent=silent,
        )


    # -------------------------------------------------------------------------
    # Geometric queries
    # -------------------------------------------------------------------------

    @staticmethod
    def CenterOfMass(topology, silent: bool = False):
        """
        Returns the geometric center of mass of the input topology.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologicpy.Vertex or None
            The center of mass of the input topology.
        """
        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.CenterOfMass - Error: The input topology parameter "
                    "is not a valid topology. Returning None."
                )
            return None

        try:
            result = Core.InstanceCall(topology, "CenterOfMass")
        except Exception as error:
            if not silent:
                print(
                    "Topology.CenterOfMass - Error: The backend operation failed. "
                    "Returning None."
                )
                print("Error:", error)
            return None

        if not Topology.IsInstance(result, "Vertex"):
            if not silent:
                print(
                    "Topology.CenterOfMass - Error: The backend did not return a "
                    "valid Vertex. Returning None."
                )
            return None

        return result

    @staticmethod
    def Centroid(topology, silent: bool = False):
        """
        Returns the geometric centroid of the input topology.

        This is an alias for :meth:`Topology.CenterOfMass`.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologicpy.Vertex or None
            The centroid of the input topology.
        """
        return Topology.CenterOfMass(topology, silent=silent)

    @staticmethod
    def IsPlanar(
        topology,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Returns True if all vertices of the input topology are coplanar.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        mantissa : int, optional
            Number of decimal places used by the planar test. Default is 6.
        tolerance : float, optional
            The geometric tolerance. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool or None
            True if planar, False if non-planar, or None if the query fails.
        """
        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.IsPlanar - Error: The input topology parameter is "
                    "not a valid topology. Returning None."
                )
            return None

        try:
            result = Core.InstanceCall(
                topology,
                "IsPlanarNative",
                int(mantissa),
                abs(float(tolerance)),
            )
        except Exception as error:
            if not silent:
                print(
                    "Topology.IsPlanar - Error: The backend operation failed. "
                    "Returning None."
                )
                print("Error:", error)
            return None

        if not isinstance(result, bool):
            if not silent:
                print(
                    "Topology.IsPlanar - Error: The backend returned an invalid "
                    "result. Returning None."
                )
            return None

        return result

    @staticmethod
    def ShortestEdge(
        topologyA,
        topologyB,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Returns the shortest connecting Edge between two topologies.

        If the topologies touch or intersect within tolerance, the shortest
        connecting segment is degenerate and None is returned.

        Parameters
        ----------
        topologyA : topologicpy.Topology
            The first input topology.
        topologyB : topologicpy.Topology
            The second input topology.
        tolerance : float, optional
            The geometric tolerance. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologicpy.Edge or None
            The shortest connecting Edge, or None if it is degenerate or the
            query fails.
        """
        if not Topology.IsInstance(topologyA, "Topology"):
            if not silent:
                print(
                    "Topology.ShortestEdge - Error: The input topologyA parameter "
                    "is not a valid topology. Returning None."
                )
            return None
        if not Topology.IsInstance(topologyB, "Topology"):
            if not silent:
                print(
                    "Topology.ShortestEdge - Error: The input topologyB parameter "
                    "is not a valid topology. Returning None."
                )
            return None

        try:
            tolerance = abs(float(tolerance))
            status, result = Core.InstanceCall(
                topologyA,
                "ShortestEdgeNative",
                topologyB,
                tolerance,
            )
        except Exception as error:
            if not silent:
                print(
                    "Topology.ShortestEdge - Error: The backend operation failed. "
                    "Returning None."
                )
                print("Error:", error)
            return None

        if status is not True:
            if not silent:
                print(
                    "Topology.ShortestEdge - Error: The backend could not compute "
                    "the shortest edge. Returning None."
                )
            return None

        if result is None:
            return None

        if not Topology.IsInstance(result, "Edge"):
            if not silent:
                print(
                    "Topology.ShortestEdge - Error: The backend returned an invalid "
                    "result. Returning None."
                )
            return None

        return result

    @staticmethod
    def ShortestDistance(
        topologyA,
        topologyB,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Returns the shortest Euclidean distance between two topologies.

        Parameters
        ----------
        topologyA : topologicpy.Topology
            The first input topology.
        topologyB : topologicpy.Topology
            The second input topology.
        mantissa : int, optional
            Number of decimal places to which the result is rounded. Default is 6.
        tolerance : float, optional
            Distances less than or equal to this value are returned as zero.
            Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        float or None
            The shortest distance, or None if it cannot be computed.
        """
        import math

        if not Topology.IsInstance(topologyA, "Topology"):
            if not silent:
                print(
                    "Topology.ShortestDistance - Error: The input topologyA "
                    "parameter is not a valid topology. Returning None."
                )
            return None
        if not Topology.IsInstance(topologyB, "Topology"):
            if not silent:
                print(
                    "Topology.ShortestDistance - Error: The input topologyB "
                    "parameter is not a valid topology. Returning None."
                )
            return None

        try:
            mantissa = int(mantissa)
            tolerance = abs(float(tolerance))
        except Exception:
            if not silent:
                print(
                    "Topology.ShortestDistance - Error: Invalid mantissa or "
                    "tolerance parameter. Returning None."
                )
            return None

        try:
            has_native_distance = Core.HasAttribute("TopologyUtility", "Distance")
        except Exception:
            has_native_distance = False

        if has_native_distance:
            try:
                distance = Core.TopologyUtility.Distance(
                    topologyA,
                    topologyB,
                    tolerance,
                )
            except Exception as error:
                if not silent:
                    print(
                        "Topology.ShortestDistance - Error: The backend distance "
                        "operation failed. Returning None."
                    )
                    print("Error:", error)
                return None

            if distance is None:
                return None

            try:
                distance = float(distance)
            except Exception:
                return None

            if not math.isfinite(distance) or distance < -tolerance:
                return None

            if abs(distance) <= tolerance:
                distance = 0.0

            return round(distance, mantissa)

        # Compatibility fallback for a backend that does not expose Distance.
        # The PythonOCC backend normally never reaches this branch.
        from topologicpy.Edge import Edge

        edge = Topology.ShortestEdge(
            topologyA,
            topologyB,
            tolerance=tolerance,
            silent=True,
        )
        if Topology.IsInstance(edge, "Edge"):
            length = Edge.Length(edge, mantissa=mantissa, silent=True)
            if isinstance(length, (int, float)):
                length = float(length)
                return 0.0 if length <= tolerance else round(length, mantissa)

        intersection = Topology.Intersect(
            topologyA,
            topologyB,
            tolerance=tolerance,
            silent=True,
        )
        if Topology.IsInstance(intersection, "Topology"):
            return 0.0

        return None

    @staticmethod
    def InternalVertex(
        topology,
        timeout: int = 30,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Returns a vertex internal to the input topology.

        The PythonOCC implementation is synchronous. The ``timeout`` parameter
        is retained for API compatibility but is not used.

        Parameters
        ----------
        topology : topologicpy.Topology
            The input topology.
        timeout : int, optional
            Retained for API compatibility. Default is 30.
        tolerance : float, optional
            The geometric tolerance. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologicpy.Vertex or None
            An internal vertex, or None if one cannot be computed.
        """
        from topologicpy.Aperture import Aperture
        from topologicpy.Cell import Cell
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Wire import Wire

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            if not silent:
                print(
                    "Topology.InternalVertex - Error: The input tolerance parameter "
                    "is not valid. Returning None."
                )
            return None

        if Topology.IsInstance(topology, "Aperture"):
            try:
                topology = Aperture.Topology(topology)
            except Exception:
                topology = None

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.InternalVertex - Error: The input topology parameter "
                    "is not a valid topology. Returning None."
                )
            return None

        type_name = Topology.TypeAsString(topology, silent=True)
        type_name = type_name.lower() if isinstance(type_name, str) else ""

        if type_name == "vertex":
            return topology

        if type_name == "edge":
            try:
                result = Edge.VertexByParameter(topology, 0.5)
            except Exception:
                result = None
            return result if Topology.IsInstance(result, "Vertex") else None

        if type_name == "wire":
            try:
                closed = Wire.IsClosed(topology)
            except Exception:
                closed = None

            if closed is True:
                try:
                    face = Core.Face.ByExternalInternalBoundaries(topology, [])
                except Exception:
                    face = None
                if not Topology.IsInstance(face, "Face"):
                    return None
                try:
                    result = Face.InternalVertex(
                        face,
                        tolerance=tolerance,
                        silent=True,
                    )
                except Exception:
                    result = None
                return result if Topology.IsInstance(result, "Vertex") else None

            if closed is False:
                edges = Topology.Edges(topology, silent=True)
                if not isinstance(edges, list) or not edges:
                    return None
                try:
                    result = Edge.VertexByParameter(edges[0], 0.5)
                except Exception:
                    result = None
                return result if Topology.IsInstance(result, "Vertex") else None

            return None

        if type_name == "face":
            try:
                result = Face.InternalVertex(
                    topology,
                    tolerance=tolerance,
                    silent=True,
                )
            except Exception:
                result = None
            return result if Topology.IsInstance(result, "Vertex") else None

        if type_name == "shell":
            faces = Topology.Faces(topology, silent=True)
            if not isinstance(faces, list):
                return None
            for face in faces:
                try:
                    result = Face.InternalVertex(
                        face,
                        tolerance=tolerance,
                        silent=True,
                    )
                except Exception:
                    result = None
                if Topology.IsInstance(result, "Vertex"):
                    return result
            return None

        if type_name == "cell":
            try:
                result = Cell.InternalVertex(
                    topology,
                    tolerance=tolerance,
                    silent=True,
                )
            except Exception:
                result = None
            return result if Topology.IsInstance(result, "Vertex") else None

        if type_name == "cellcomplex":
            cells = Topology.Cells(topology, silent=True)
            if not isinstance(cells, list):
                return None
            for cell in cells:
                try:
                    result = Cell.InternalVertex(
                        cell,
                        tolerance=tolerance,
                        silent=True,
                    )
                except Exception:
                    result = None
                if Topology.IsInstance(result, "Vertex"):
                    return result
            return None

        if type_name == "cluster":
            try:
                constituents = Core.InstanceCall(topology, "Topologies")
            except Exception:
                return None

            if not isinstance(constituents, list) or not constituents:
                return None

            constituents = sorted(
                constituents,
                key=lambda item: Topology._TYPE_RANKS.get(
                    str(Topology.TypeAsString(item, silent=True)).lower(),
                    -1,
                ),
                reverse=True,
            )

            for constituent in constituents:
                result = Topology.InternalVertex(
                    constituent,
                    tolerance=tolerance,
                    silent=True,
                )
                if Topology.IsInstance(result, "Vertex"):
                    return result

            return None

        return None

    @staticmethod
    def _InternalVertex(
        topology,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Compatibility alias for :meth:`Topology.InternalVertex`."""
        return Topology.InternalVertex(
            topology,
            tolerance=tolerance,
            silent=silent,
        )


    # -----------------------------------------------------------------------
    # Native persistence / codec routing
    # -----------------------------------------------------------------------

    @staticmethod
    def Save(
        topology,
        path,
        overwrite: bool = False,
        silent: bool = False,
    ) -> bool:
        """Save a topology using the codec selected by the output extension.

        ``.tpy`` is TopologicPy's native lossless persistence format. It stores
        exact OCCT BREP geometry together with TopologicPy dictionaries and
        semantic relationships. Mesh exchange formats are intentionally not
        routed through this native persistence method.
        """
        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print(
                    "Topology.Save - Error: The input topology parameter is not "
                    "a valid topology. Returning False."
                )
            return False

        try:
            from topologicpy.io import codec_for_path
            codec = codec_for_path(path)
        except Exception:
            codec = None

        if codec is None:
            if not silent:
                print(
                    "Topology.Save - Error: No serialization codec is registered "
                    "for the output path. Returning False."
                )
            return False

        return bool(
            codec.save(
                topology,
                path,
                overwrite=overwrite,
                silent=silent,
            )
        )

    @staticmethod
    def Load(path, silent: bool = False):
        """Load a topology using the codec selected by the input extension."""
        try:
            from topologicpy.io import codec_for_path
            codec = codec_for_path(path)
        except Exception:
            codec = None

        if codec is None:
            if not silent:
                print(
                    "Topology.Load - Error: No serialization codec is registered "
                    "for the input path. Returning None."
                )
            return None

        return codec.load(path, silent=silent)

    @staticmethod
    def ExportToSTEP(
        topology,
        path,
        overwrite: bool = False,
        schema: str = "AP242DIS",
        unit: str = "MM",
        assembly="auto",
        tolerance=None,
        silent: bool = False,
    ) -> bool:
        """Export a topology to neutral STEP BREP geometry.

        STEP is a tolerance-based CAD interchange format, not TopologicPy's
        lossless semantic persistence format. Dictionaries, Content, Aperture
        and Context identity are not guaranteed to survive this exchange.
        """
        try:
            from topologicpy.io.step import STEPCodec
        except Exception:
            if not silent:
                print(
                    "Topology.ExportToSTEP - Error: The STEP codec could not be "
                    "loaded. Returning False."
                )
            return False
        return bool(
            STEPCodec.save(
                topology,
                path,
                overwrite=overwrite,
                schema=schema,
                unit=unit,
                assembly=assembly,
                tolerance=tolerance,
                silent=silent,
            )
        )

    @staticmethod
    def BySTEPPath(
        path,
        unit: str = "MM",
        silent: bool = False,
    ):
        """Create a topology from neutral STEP BREP geometry."""
        try:
            from topologicpy.io.step import STEPCodec
        except Exception:
            if not silent:
                print(
                    "Topology.BySTEPPath - Error: The STEP codec could not be "
                    "loaded. Returning None."
                )
            return None
        return STEPCodec.load(path, unit=unit, silent=silent)

    @staticmethod
    def ExportToTPY(
        topology,
        path,
        overwrite: bool = False,
        silent: bool = False,
    ) -> bool:
        """Export a topology to TopologicPy's exact native ``.tpy`` archive."""
        try:
            from topologicpy.io.tpy import TPYCodec
        except Exception:
            if not silent:
                print(
                    "Topology.ExportToTPY - Error: The TPY codec could not be "
                    "loaded. Returning False."
                )
            return False
        return bool(
            TPYCodec.save(
                topology,
                path,
                overwrite=overwrite,
                silent=silent,
            )
        )

    @staticmethod
    def ByTPYPath(path, silent: bool = False):
        """Create a topology from a TopologicPy native ``.tpy`` archive."""
        try:
            from topologicpy.io.tpy import TPYCodec
        except Exception:
            if not silent:
                print(
                    "Topology.ByTPYPath - Error: The TPY codec could not be "
                    "loaded. Returning None."
                )
            return None
        return TPYCodec.load(path, silent=silent)


# ---------------------------------------------------------------------------
# Transitional late-dispatch bridge
# ---------------------------------------------------------------------------
#
# Functions inherited from TopologyLegacy retain the global namespace of the
# module in which they were defined. Rebind that module-global symbol to this
# PythonOCC subclass so calls such as ``Topology.Faces(...)`` inside inherited
# methods resolve to the progressively modernized implementation above.
#
# This bridge is temporary and must be removed together with the legacy
# inheritance when the PythonOCC Topology implementation is complete.
#
_topology_legacy_module.Topology = Topology


__all__ = ["Topology"]
