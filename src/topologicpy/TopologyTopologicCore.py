# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option)
# any later version.

"""TopologicCore public adapter for the backend-independent semantic layer.

All geometry/topology operations remain inherited unchanged from
``TopologyLegacy.Topology``. Only Content/Aperture/Context identity and
relationship methods are implemented at the TopologicPy layer so they have the
same meaning under TopologicCore and PythonOCC.
"""

from __future__ import annotations

import topologicpy.TopologyLegacy as _legacy_module

TopologyLegacy = _legacy_module.Topology


class Topology(TopologyLegacy):

    @staticmethod
    def IsInstance(topology, type: str, silent: bool = True):
        """Returns whether ``topology`` is an instance of the requested type."""
        if not isinstance(type, str):
            if not silent:
                print("Topology.IsInstance - Error: The input type parameter is not a valid string. Returning None.")
            return None

        requested = type.strip().lower()
        if requested in ("content", "aperture", "context"):
            try:
                from topologicpy.Content import Content
                from topologicpy.Aperture import Aperture
                from topologicpy.Context import Context
                if requested == "content":
                    if isinstance(topology, Content):
                        return True
                    # A leaked native Aperture is still semantically Content.
                    try:
                        return bool(TopologyLegacy.IsInstance(topology, "Aperture", silent=True))
                    except Exception:
                        return False
                if requested == "aperture" and isinstance(topology, Aperture):
                    return True
                if requested == "context" and isinstance(topology, Context):
                    return True
            except Exception:
                pass

            native_name = "Aperture" if requested == "aperture" else "Context"
            if requested == "content":
                return False
            try:
                return bool(TopologyLegacy.IsInstance(topology, native_name, silent=True))
            except Exception:
                return False

        return TopologyLegacy.IsInstance(topology, type, silent=silent)

    @staticmethod
    def _Deduplicate(topologies):
        result = []
        for topology in topologies or []:
            duplicate = False
            for existing in result:
                try:
                    if TopologyLegacy.IsSame(topology, existing, silent=True):
                        duplicate = True
                        break
                except Exception:
                    if topology is existing:
                        duplicate = True
                        break
            if not duplicate:
                result.append(topology)
        return result

    @staticmethod
    def _RelationshipCandidates(topology, subTopologyType, allowed, silent=False):
        if subTopologyType is None:
            requested = "self"
        elif isinstance(subTopologyType, str):
            requested = subTopologyType.strip().lower() or "self"
        else:
            if not silent:
                print("Topology - Error: The input subTopologyType parameter is not a valid string. Returning None.")
            return None

        if requested not in allowed:
            if not silent:
                print("Topology - Error: The input subTopologyType parameter is not recognized. Returning None.")
            return None
        if requested == "self":
            return [topology]
        return TopologyLegacy.SubTopologies(topology, subTopologyType=requested, silent=silent)

    @staticmethod
    def _RelationshipHostForContent(content, candidates, tolerance=0.0001):
        from topologicpy.Vertex import Vertex

        try:
            selector = TopologyLegacy.InternalVertex(content, tolerance=tolerance, silent=True)
        except Exception:
            selector = None
        if not Topology.IsInstance(selector, "Vertex"):
            try:
                selector = TopologyLegacy.Centroid(content, silent=True)
            except Exception:
                selector = None
        if not Topology.IsInstance(selector, "Vertex"):
            return None

        for candidate in candidates or []:
            try:
                if Vertex.IsInternal(selector, candidate, tolerance=tolerance, silent=True):
                    return candidate
            except TypeError:
                try:
                    if Vertex.IsInternal(selector, candidate, tolerance=tolerance):
                        return candidate
                except Exception:
                    continue
            except Exception:
                continue
        return None

    @staticmethod
    def AddContent(topology, contents=None, subTopologyType: str = None, tolerance: float = 0.0001, silent: bool = False):
        """Adds backend-independent Content relationships without copying content."""
        from topologicpy.SemanticManager import SemanticManager

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Topology.AddContent - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        if contents is None:
            if not silent:
                print("Topology.AddContent - Warning: The input contents parameter is empty. Returning the input topology unmodified.")
            return topology
        if not isinstance(contents, list):
            contents = [contents]
        contents = [content for content in contents if Topology.IsInstance(content, "Topology")]
        if not contents:
            return topology

        candidates = Topology._RelationshipCandidates(
            topology,
            subTopologyType,
            allowed=("self", "cellcomplex", "cell", "shell", "face", "wire", "edge", "vertex"),
            silent=silent,
        )
        if candidates is None:
            return None

        manager = SemanticManager.GetInstance()
        direct = subTopologyType is None or (
            isinstance(subTopologyType, str)
            and (subTopologyType.strip() == "" or subTopologyType.strip().lower() == "self")
        )
        for content in contents:
            host = candidates[0] if direct else Topology._RelationshipHostForContent(content, candidates, tolerance=tolerance)
            if host is not None:
                manager.register(content, host, parameters=None)
        return topology

    @staticmethod
    def AddApertures(topology, apertures, exclusive=False, subTopologyType=None, tolerance=0.001, silent: bool = False):
        """Adds Aperture Content relationships using SemanticManager."""
        from topologicpy.Dictionary import Dictionary
        from topologicpy.SemanticManager import SemanticManager

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Topology.AddApertures - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        if not apertures:
            return topology
        if not isinstance(apertures, list):
            if not silent:
                print("Topology.AddApertures - Error: The input apertures parameter is not a list. Returning None.")
            return None
        apertures = [aperture for aperture in apertures if Topology.IsInstance(aperture, "Topology")]
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
        direct = subTopologyType is None or (
            isinstance(subTopologyType, str)
            and (subTopologyType.strip() == "" or subTopologyType.strip().lower() == "self")
        )

        for aperture in apertures:
            try:
                dictionary = TopologyLegacy.Dictionary(aperture, silent=True)
                dictionary = Dictionary.SetValueAtKey(dictionary, "type", "Aperture")
                marked = TopologyLegacy.SetDictionary(aperture, dictionary, silent=True)
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
                    if not bool(exclusive) or not manager.aperture_topologies_for_host(candidate)
                ]
                host = Topology._RelationshipHostForContent(aperture, eligible, tolerance=tolerance)

            if host is not None:
                manager.register(aperture, host, aperture=True, parameters=None)
        return topology

    @staticmethod
    def Contents(topology, silent: bool = False):
        """Returns Content topologies hosted by ``topology``."""
        from topologicpy.SemanticManager import SemanticManager

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Topology.Contents - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        manager_result = SemanticManager.GetInstance().content_topologies_for_host(topology)
        if manager_result:
            return Topology._Deduplicate(manager_result)
        # Transitional fallback for relationships created before the semantic layer.
        try:
            return TopologyLegacy.Contents(topology, silent=silent)
        except Exception:
            return []

    @staticmethod
    def Contexts(topology, silent: bool = False):
        """Returns all Context relationships associated with Content ``topology``."""
        from topologicpy.Content import Content
        from topologicpy.SemanticManager import SemanticManager

        if isinstance(topology, Content):
            return SemanticManager.GetInstance().contexts_for_content(topology)
        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Topology.Contexts - Error: The input topology parameter is not a valid topology or Content. Returning None.")
            return None
        manager_result = SemanticManager.GetInstance().contexts_for_content(topology)
        if manager_result:
            return manager_result
        try:
            return TopologyLegacy.Contexts(topology, silent=silent)
        except Exception:
            return []

    @staticmethod
    def Apertures(topology, subTopologyType=None, silent: bool = False):
        """Returns Aperture Content topologies hosted by ``topology``."""
        from topologicpy.SemanticManager import SemanticManager

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Topology.Apertures - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        manager = SemanticManager.GetInstance()

        if subTopologyType is None:
            result = manager.aperture_topologies_for_host(topology)
            if result:
                return Topology._Deduplicate(result)
            try:
                return TopologyLegacy.Apertures(topology, subTopologyType=None, silent=silent)
            except Exception:
                return []

        if not isinstance(subTopologyType, str):
            if not silent:
                print("Topology.Apertures - Error: The input subTopologyType parameter is not a valid string. Returning None.")
            return None
        requested = subTopologyType.strip().lower()
        if requested not in ("vertex", "edge", "face", "cell", "all"):
            if not silent:
                print("Topology.Apertures - Error: The input subTopologyType parameter is not recognized. Returning None.")
            return None

        result = []
        if requested == "all":
            result.extend(manager.aperture_topologies_for_host(topology))
            requested_types = ("vertex", "edge", "face", "cell")
        else:
            requested_types = (requested,)

        for type_name in requested_types:
            subtopologies = TopologyLegacy.SubTopologies(topology, subTopologyType=type_name, silent=True)
            if subtopologies is None:
                return None
            for subtopology in subtopologies:
                result.extend(manager.aperture_topologies_for_host(subtopology))

        if result:
            return Topology._Deduplicate(result)
        try:
            return TopologyLegacy.Apertures(topology, subTopologyType=subTopologyType, silent=silent)
        except Exception:
            return []

    @staticmethod
    def ApertureTopologies(topology, subTopologyType=None, silent: bool = False):
        return Topology.Apertures(topology, subTopologyType=subTopologyType, silent=silent)

    @staticmethod
    def RemoveContent(topology, contents, silent: bool = False):
        """Removes only matching Context relationships from ``topology``."""
        from topologicpy.Content import Content
        from topologicpy.SemanticManager import SemanticManager

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Topology.RemoveContent - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        if contents is None:
            return topology
        if not isinstance(contents, list):
            contents = [contents]
        contents = [item for item in contents if isinstance(item, Content) or Topology.IsInstance(item, "Topology")]
        if not contents:
            return topology

        manager = SemanticManager.GetInstance()
        removed = manager.remove(topology, contents=contents)
        if removed > 0:
            return topology

        # Transitional fallback for pre-existing native relationships only.
        try:
            return TopologyLegacy.RemoveContent(topology, contents, silent=silent)
        except Exception:
            return topology


__all__ = ["Topology"]
