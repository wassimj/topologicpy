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

"""Thin namespace facade for the PythonOCC geometry and topology kernel.

Native objects are returned directly. The facade is lazy so importing modules
that do not use geometry does not require loading the compiled kernel.
"""

from typing import Any, List, Optional


def _CreateDefaultBackend() -> Any:
    """Load the sole supported geometry kernel; never fall back."""
    try:
        from OCC.Core import TopoDS  # noqa: F401
        from topologicpy.pythonocc_backend import PythonOCCBackend
    except (ImportError, OSError) as exc:
        raise ImportError(
            "TopologicPy requires PythonOCC (pythonocc-core). Install it in "
            "the active Python environment using the installation guide in README.md."
        ) from exc
    return PythonOCCBackend()


class _MissingNamespace:
    """
    Placeholder namespace used when a backend does not expose a requested
    kernel namespace.
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


class Core:
    """
    Thin facade over the PythonOCC topology kernel.
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
    Grid = _NamespaceProxy("Grid")

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
        Returns the active backend.

        PythonOCC is loaded on first use.
        """
        if Core._backend is None:
            Core._backend = _CreateDefaultBackend()
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
        Injects a namespace provider for testing; PythonOCC is the only supported runtime.

        Parameters
        ----------
        backend : object
            A test double exposing the kernel namespaces such as
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
        Resets the facade to PythonOCC.
        """
        Core._backend = None
        return Core.Backend()

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


__all__ = ["Core"]
