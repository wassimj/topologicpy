# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

"""
TopologicPy Core Facade
=======================

This module provides a thin, centralised facade over ``topologic_core``.

Purpose
-------
TopologicPy should eventually avoid importing ``topologic_core`` directly from
multiple files. Instead, all direct access to the geometry/topology kernel can be
routed through this module:

    from topologicpy.Core import Core

    v = Core.Vertex.ByCoordinates(0, 0, 0)
    e = Core.Edge.ByStartVertexEndVertex(v1, v2, tolerance)
    _ = Core.Topology.Analyze(topology)

The facade is intentionally thin. It mirrors the exposed ``topologic_core``
classes and utility namespaces rather than redefining TopologicPy semantics.

Backend replacement
-------------------
The default backend is ``TopologicCoreBackend``. A future backend can be supplied
with:

    Core.SetBackend(my_backend)

The supplied backend should expose equivalent namespace attributes such as
``Vertex``, ``Edge``, ``Wire``, ``TopologyUtility``, ``FaceUtility``, etc.

Design note
-----------
This file does not wrap every instance method of every returned topology object.
Core objects are still returned as backend-native objects. The goal is to
centralise access to backend namespaces and static/factory/utility methods.
Instance method calls can later be routed through additional helper methods if
needed, but doing so exhaustively would require proxy objects and would be a
larger architectural migration.
"""

from typing import Any, List, Optional


class _MissingNamespace:
    """
    Placeholder namespace used when a backend does not expose a requested
    topologic_core namespace.
    """

    def __init__(self, namespace_name: str):
        self._namespace_name = namespace_name

    def __getattr__(self, attr: str) -> Any:
        raise AttributeError(
            f"The active Core backend does not expose namespace "
            f"'{self._namespace_name}'. Requested attribute: '{attr}'."
        )

    def __repr__(self) -> str:
        return f"<Missing Core namespace: {self._namespace_name}>"


class _NamespaceProxy:
    """
    Dynamic proxy for a backend namespace.

    Example
    -------
    Core.Vertex.ByCoordinates(...) resolves as:

        getattr(Core.Backend(), "Vertex").ByCoordinates(...)
    """

    def __init__(self, namespace_name: str):
        self._namespace_name = namespace_name

    @property
    def NamespaceName(self) -> str:
        return self._namespace_name

    def _namespace(self) -> Any:
        backend = Core.Backend()
        namespace = getattr(backend, self._namespace_name, None)
        if namespace is None:
            return _MissingNamespace(self._namespace_name)
        return namespace

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._namespace(), attr)

    def __call__(self, *args, **kwargs) -> Any:
        namespace = self._namespace()
        if callable(namespace):
            return namespace(*args, **kwargs)
        raise TypeError(f"Core.{self._namespace_name} is not callable.")

    def __repr__(self) -> str:
        return f"<Core namespace proxy: {self._namespace_name}>"


class TopologicCoreBackend:
    """
    Default backend that exposes the ``topologic_core`` namespaces directly.

    This class deliberately mirrors ``topologic_core`` as closely as possible.
    Any missing namespaces are set to ``None`` so that Core can fail gracefully
    with a clear message when accessed.
    """

    def __init__(self):
        try:
            import topologic_core as topologic
        except Exception as e:
            raise ImportError(
                "Could not import topologic_core. Install topologic_core or set "
                "a different Core backend with Core.SetBackend(...)."
            ) from e

        self._topologic = topologic

        # Primary topology namespaces.
        self.Vertex = getattr(topologic, "Vertex", None)
        self.Edge = getattr(topologic, "Edge", None)
        self.Wire = getattr(topologic, "Wire", None)
        self.Face = getattr(topologic, "Face", None)
        self.Shell = getattr(topologic, "Shell", None)
        self.Cell = getattr(topologic, "Cell", None)
        self.CellComplex = getattr(topologic, "CellComplex", None)
        self.Cluster = getattr(topologic, "Cluster", None)
        self.Graph = getattr(topologic, "Graph", None)

        # Attribute/data namespaces.
        self.Dictionary = getattr(topologic, "Dictionary", None)
        self.Aperture = getattr(topologic, "Aperture", None)
        self.Context = getattr(topologic, "Context", None)
        self.Topology = getattr(topologic, "Topology", None)

        # Utility namespaces.
        self.TopologyUtility = getattr(topologic, "TopologyUtility", None)
        self.VertexUtility = getattr(topologic, "VertexUtility", None)
        self.EdgeUtility = getattr(topologic, "EdgeUtility", None)
        self.WireUtility = getattr(topologic, "WireUtility", None)
        self.FaceUtility = getattr(topologic, "FaceUtility", None)
        self.ShellUtility = getattr(topologic, "ShellUtility", None)
        self.CellUtility = getattr(topologic, "CellUtility", None)
        self.CellComplexUtility = getattr(topologic, "CellComplexUtility", None)
        self.ClusterUtility = getattr(topologic, "ClusterUtility", None)
        self.GraphUtility = getattr(topologic, "GraphUtility", None)

        # Preserve all other public symbols exposed by topologic_core. This makes
        # the backend more exhaustive without having to predict every future name.
        for name in dir(topologic):
            if name.startswith("_"):
                continue
            if not hasattr(self, name):
                try:
                    setattr(self, name, getattr(topologic, name))
                except Exception:
                    pass

    def RawModule(self) -> Any:
        """
        Returns the raw imported ``topologic_core`` module.
        """
        return self._topologic

    def Namespaces(self) -> List[str]:
        """
        Returns the public namespace names exposed by this backend.
        """
        return sorted([name for name in dir(self) if not name.startswith("_")])


class Core:
    """
    Thin facade over the active topology kernel backend.

    The default backend is ``TopologicCoreBackend``. Use ``Core.SetBackend`` to
    replace it with another backend object that exposes the same namespace shape.
    """

    _backend: Optional[Any] = None

    # Primary topology namespaces.
    Vertex = _NamespaceProxy("Vertex")
    Edge = _NamespaceProxy("Edge")
    Wire = _NamespaceProxy("Wire")
    Face = _NamespaceProxy("Face")
    Shell = _NamespaceProxy("Shell")
    Cell = _NamespaceProxy("Cell")
    CellComplex = _NamespaceProxy("CellComplex")
    Cluster = _NamespaceProxy("Cluster")
    Graph = _NamespaceProxy("Graph")

    # Attribute/data namespaces.
    Dictionary = _NamespaceProxy("Dictionary")
    Aperture = _NamespaceProxy("Aperture")
    Context = _NamespaceProxy("Context")
    Topology = _NamespaceProxy("Topology")

    # Utility namespaces.
    TopologyUtility = _NamespaceProxy("TopologyUtility")
    VertexUtility = _NamespaceProxy("VertexUtility")
    EdgeUtility = _NamespaceProxy("EdgeUtility")
    WireUtility = _NamespaceProxy("WireUtility")
    FaceUtility = _NamespaceProxy("FaceUtility")
    ShellUtility = _NamespaceProxy("ShellUtility")
    CellUtility = _NamespaceProxy("CellUtility")
    CellComplexUtility = _NamespaceProxy("CellComplexUtility")
    ClusterUtility = _NamespaceProxy("ClusterUtility")
    GraphUtility = _NamespaceProxy("GraphUtility")

    # Attribute Proxies
    IntAttribute = _NamespaceProxy("IntAttribute")
    DoubleAttribute = _NamespaceProxy("DoubleAttribute")
    StringAttribute = _NamespaceProxy("StringAttribute")
    ListAttribute = _NamespaceProxy("ListAttribute")

    @staticmethod
    def Backend() -> Any:
        """
        Returns the active backend. If none has been set, creates the default
        ``TopologicCoreBackend``.
        """
        if Core._backend is None:
            Core._backend = TopologicCoreBackend()
        return Core._backend
    
    @staticmethod
    def InstanceCall(obj, methodName: str, *args, **kwargs):
        """
        Calls an instance method on a backend-native object.
        """
        if obj is None:
            raise ValueError("Core.InstanceCall - Error: obj cannot be None.")
        if not isinstance(methodName, str) or len(methodName) == 0:
            raise ValueError("Core.InstanceCall - Error: methodName must be a non-empty string.")
        method = getattr(obj, methodName)
        if not callable(method):
            raise TypeError(f"Core.InstanceCall - Error: {methodName} is not callable.")
        return method(*args, **kwargs)


    @staticmethod
    def InstanceAttribute(obj, attributeName: str):
        """
        Returns an instance attribute from a backend-native object.
        """
        if obj is None:
            raise ValueError("Core.InstanceAttribute - Error: obj cannot be None.")
        if not isinstance(attributeName, str) or len(attributeName) == 0:
            raise ValueError("Core.InstanceAttribute - Error: attributeName must be a non-empty string.")
        return getattr(obj, attributeName)

    @staticmethod
    def SetBackend(backend: Any) -> Any:
        """
        Sets the active Core backend.

        Parameters
        ----------
        backend : object
            A backend object exposing topologic_core-like namespaces such as
            ``Vertex``, ``Edge``, ``Wire``, ``Face``, ``TopologyUtility``, etc.

        Returns
        -------
        object
            The backend that was set.
        """
        if backend is None:
            raise ValueError("Core.SetBackend - Error: backend cannot be None.")
        Core._backend = backend
        return Core._backend

    @staticmethod
    def ResetBackend() -> Any:
        """
        Resets the active backend to a new ``TopologicCoreBackend`` instance.
        """
        Core._backend = TopologicCoreBackend()
        return Core._backend

    @staticmethod
    def RawModule() -> Any:
        """
        Returns the raw module of the active backend if available.
        """
        backend = Core.Backend()
        if hasattr(backend, "RawModule"):
            return backend.RawModule()
        return backend

    @staticmethod
    def Namespaces() -> List[str]:
        """
        Returns the public namespace names exposed by the active backend.
        """
        backend = Core.Backend()
        if hasattr(backend, "Namespaces"):
            return backend.Namespaces()
        return sorted([name for name in dir(backend) if not name.startswith("_")])

    @staticmethod
    def Namespace(name: str) -> Any:
        """
        Returns a backend namespace by name.
        """
        if not isinstance(name, str) or len(name) == 0:
            raise ValueError("Core.Namespace - Error: name must be a non-empty string.")
        namespace = getattr(Core.Backend(), name, None)
        if namespace is None:
            raise AttributeError(f"The active Core backend does not expose namespace '{name}'.")
        return namespace

    @staticmethod
    def HasNamespace(name: str) -> bool:
        """
        Returns True if the active backend exposes the requested namespace.
        """
        if not isinstance(name, str) or len(name) == 0:
            return False
        return getattr(Core.Backend(), name, None) is not None

    @staticmethod
    def HasAttribute(namespace: str, attribute: str) -> bool:
        """
        Returns True if the active backend namespace exposes the requested
        attribute.
        """
        if not Core.HasNamespace(namespace):
            return False
        ns = Core.Namespace(namespace)
        return hasattr(ns, attribute)

    @staticmethod
    def Call(namespace: str, attribute: str, *args, **kwargs) -> Any:
        """
        Dynamically calls a method or callable attribute on a backend namespace.

        Example
        -------
        Core.Call("Vertex", "ByCoordinates", 0, 0, 0)
        """
        ns = Core.Namespace(namespace)
        fn = getattr(ns, attribute)
        if not callable(fn):
            raise TypeError(f"Core.Call - Error: Core.{namespace}.{attribute} is not callable.")
        return fn(*args, **kwargs)


__all__ = ["Core", "TopologicCoreBackend"]
