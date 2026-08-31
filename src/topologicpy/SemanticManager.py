# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option)
# any later version.

from __future__ import annotations

import copy
from typing import Optional


class SemanticManager:
    """Authoritative store for TopologicPy Content/Context relationships.

    Geometry remains owned by topology/backend objects. This manager stores only
    semantic Content identity and Context relationships and resolves topology
    identity through the underlying kernel shape rather than transient Python
    wrapper identity.
    """

    _instance = None

    def __init__(self):
        self._contents = []
        self._contexts = []

    @classmethod
    def GetInstance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def Reset(cls):
        cls._instance = cls()
        return cls._instance

    @staticmethod
    def _represented_topology(value):
        try:
            from topologicpy.Content import Content
            if isinstance(value, Content):
                return value.topology
        except Exception:
            pass
        return value

    @staticmethod
    def _shape(value):
        value = SemanticManager._represented_topology(value)
        if value is None:
            return None

        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                if not shape.IsNull():
                    return shape
            except Exception:
                return shape

        method = getattr(value, "GetOcctShape", None)
        if callable(method):
            try:
                shape = method()
                if shape is not None:
                    return shape
            except Exception:
                pass
        return None

    @staticmethod
    def same_topology(a, b) -> bool:
        """Returns True when two values represent the same kernel topology."""
        a = SemanticManager._represented_topology(a)
        b = SemanticManager._represented_topology(b)
        if a is b:
            return a is not None
        if a is None or b is None:
            return False

        shape_a = SemanticManager._shape(a)
        shape_b = SemanticManager._shape(b)
        if shape_a is not None and shape_b is not None:
            try:
                return bool(shape_a.IsSame(shape_b))
            except Exception:
                pass

        # TopologicCore objects do not expose an OCCT ``shape`` attribute, but
        # the Core facade provides the kernel identity operation. Keep this
        # below the direct OCCT path so PythonOCC stays fast.
        try:
            from topologicpy.Core import Core
            return bool(Core.Topology.IsSame(a, b))
        except Exception:
            pass

        uuid_a = getattr(a, "_uuid", None)
        uuid_b = getattr(b, "_uuid", None)
        if isinstance(uuid_a, str) and uuid_a and isinstance(uuid_b, str):
            return uuid_a == uuid_b
        return False

    def _promote_to_aperture(self, content):
        from topologicpy.Aperture import Aperture
        from topologicpy.Content import Content

        if isinstance(content, Aperture):
            return content
        if not isinstance(content, Content):
            return None

        promoted = Aperture(
            content.topology,
            dictionary=content.dictionary,
            uuid_value=content._uuid,
        )
        try:
            index = self._contents.index(content)
            self._contents[index] = promoted
        except ValueError:
            self._contents.append(promoted)

        for context in self._contexts:
            if getattr(context, "content", None) is content:
                context.content = promoted
        return promoted

    def content_for_topology(
        self,
        topology,
        aperture: Optional[bool] = None,
        create: bool = False,
        dictionary=None,
    ):
        """Returns the unique Content represented by ``topology``."""
        from topologicpy.Aperture import Aperture
        from topologicpy.Content import Content

        if isinstance(topology, Content):
            content = topology
            if content not in self._contents:
                existing = self.content_for_topology(
                    content.topology, aperture=None, create=False
                )
                if existing is None:
                    self._contents.append(content)
                else:
                    content = existing
        else:
            content = None
            for existing in self._contents:
                if self.same_topology(existing.topology, topology):
                    content = existing
                    break

        if content is None and create:
            cls = Aperture if aperture is True else Content
            content = cls(topology, dictionary=dictionary)
            self._contents.append(content)
        elif content is not None and aperture is True and not isinstance(content, Aperture):
            content = self._promote_to_aperture(content)

        if content is not None and isinstance(dictionary, dict):
            content.dictionary = dict(dictionary)

        return content

    def register(
        self,
        content,
        host,
        aperture: Optional[bool] = None,
        parameters=None,
        content_dictionary=None,
        context_dictionary=None,
        context=None,
    ):
        """Registers one Content-to-host Context and returns both objects."""
        from topologicpy.Content import Content
        from topologicpy.Context import Context

        content_object = self.content_for_topology(
            content,
            aperture=aperture,
            create=True,
            dictionary=content_dictionary,
        )
        if content_object is None or host is None:
            return None, None

        for existing in self._contexts:
            if (
                getattr(existing, "content", None) is content_object
                and self.same_topology(getattr(existing, "host", None), host)
            ):
                if parameters is not None:
                    existing.parameters = copy.deepcopy(parameters)
                if isinstance(context_dictionary, dict):
                    existing.dictionary = dict(context_dictionary)
                return content_object, existing

        if isinstance(context, Context):
            relation = context
            relation.content = content_object
            relation.host = host
            if parameters is not None:
                relation.parameters = copy.deepcopy(parameters)
            if isinstance(context_dictionary, dict):
                relation.dictionary = dict(context_dictionary)
        else:
            relation = Context(
                content=content_object,
                host=host,
                parameters=parameters,
                dictionary=context_dictionary,
            )

        self._contexts.append(relation)
        return content_object, relation

    def contexts_for_content(self, content_or_topology):
        from topologicpy.Content import Content

        if isinstance(content_or_topology, Content):
            content = self.content_for_topology(content_or_topology, create=False)
        else:
            content = self.content_for_topology(content_or_topology, create=False)
        if content is None:
            return []
        return [
            context
            for context in self._contexts
            if getattr(context, "content", None) is content
        ]

    def contents_for_host(self, host, apertures_only: bool = False):
        from topologicpy.Aperture import Aperture

        result = []
        for context in self._contexts:
            if not self.same_topology(getattr(context, "host", None), host):
                continue
            content = getattr(context, "content", None)
            if content is None:
                continue
            if apertures_only and not isinstance(content, Aperture):
                continue
            if content not in result:
                result.append(content)
        return result

    def content_topologies_for_host(self, host):
        return [content.topology for content in self.contents_for_host(host)]

    def aperture_topologies_for_host(self, host):
        return [
            content.topology
            for content in self.contents_for_host(host, apertures_only=True)
        ]

    def remove(self, host, contents=None, apertures_only: bool = False):
        """Removes matching Contexts from ``host`` and returns their count."""
        from topologicpy.Aperture import Aperture
        from topologicpy.Content import Content

        if contents is None:
            targets = None
        elif isinstance(contents, (list, tuple)):
            targets = list(contents)
        else:
            targets = [contents]

        def matches_target(content_object):
            if targets is None:
                return True
            for target in targets:
                if isinstance(target, Content):
                    if content_object is target:
                        return True
                    target_topology = target.topology
                else:
                    target_topology = target
                if self.same_topology(content_object.topology, target_topology):
                    return True
            return False

        kept = []
        removed = 0
        for context in self._contexts:
            content_object = getattr(context, "content", None)
            same_host = self.same_topology(getattr(context, "host", None), host)
            subtype_ok = (
                not apertures_only
                or isinstance(content_object, Aperture)
            )
            if same_host and subtype_ok and content_object is not None and matches_target(content_object):
                removed += 1
                continue
            kept.append(context)
        self._contexts = kept
        return removed

    def transfer_topology(self, source, result):
        """Transfers semantic participation from ``source`` to ``result``.

        Host relationships are copied so the transformed/copied result hosts the
        same Content objects. If ``source`` itself represents Content, ``result``
        receives a new Content of the same semantic subtype and the same Contexts.
        """
        from topologicpy.Aperture import Aperture

        if source is None or result is None or self.same_topology(source, result):
            return result

        # Source acting as a host.
        host_contexts = [
            context for context in list(self._contexts)
            if self.same_topology(getattr(context, "host", None), source)
        ]
        for context in host_contexts:
            self.register(
                context.content,
                result,
                aperture=isinstance(context.content, Aperture),
                parameters=copy.deepcopy(context.parameters),
                context_dictionary=copy.deepcopy(context.dictionary),
            )

        # Source acting as represented Content.
        source_content = self.content_for_topology(source, create=False)
        if source_content is not None:
            result_content = self.content_for_topology(
                result,
                aperture=isinstance(source_content, Aperture),
                create=True,
                dictionary=copy.deepcopy(source_content.dictionary),
            )
            for context in self.contexts_for_content(source_content):
                self.register(
                    result_content,
                    context.host,
                    aperture=isinstance(source_content, Aperture),
                    parameters=copy.deepcopy(context.parameters),
                    context_dictionary=copy.deepcopy(context.dictionary),
                )

        return result

    def all_contents(self):
        return list(self._contents)

    def all_contexts(self):
        return list(self._contexts)


__all__ = ["SemanticManager"]
