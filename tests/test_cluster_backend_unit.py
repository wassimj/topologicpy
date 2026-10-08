"""Kernel-free checks for aggregate integrity and native-edit dispatch.

Run directly with Python when no geometry kernel is available. A private
package alias loads the real backend modules without importing OCC-dependent
public namespaces; only the compound builder is replaced by a test double.
"""
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


package_name = "_topologicpy_cluster_unit_backend"
package = types.ModuleType(package_name)
package.__path__ = [str(Path(__file__).resolve().parents[1] / "src" / "topologicpy" / "pythonocc_backend")]
sys.modules[package_name] = package
backend = importlib.import_module(package_name + ".topology")
Cluster = importlib.import_module(package_name + ".cluster").Cluster
Topology = backend.Topology


class Shape:
    def __init__(self):
        self.members = []

    def IsNull(self):
        return False

    def ShapeType(self):
        return "compound"


class Builder:
    def MakeCompound(self, shape):
        shape.members = []

    def Add(self, shape, member):
        shape.members.append(member)


class Editable(Topology):
    def edit(self, outcome):
        return outcome(self)


class ClusterIntegrityTests(unittest.TestCase):
    def setUp(self):
        topods = types.ModuleType("OCC.Core.TopoDS")
        topods.TopoDS_Compound = Shape
        brep = types.ModuleType("OCC.Core.BRep")
        brep.BRep_Builder = Builder
        modules = {"OCC": types.ModuleType("OCC"), "OCC.Core": types.ModuleType("OCC.Core"),
                   "OCC.Core.TopoDS": topods, "OCC.Core.BRep": brep}
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_nested_compound_preserves_all_members_without_mutation(self):
        a, b = Topology(shape=Shape()), Topology(shape=Shape())
        inner = Cluster.ByTopologies([a])
        outer = Cluster.ByTopologies([inner, b])
        shape = outer.GetOcctShape()
        self.assertEqual(shape.members[0].members, [a.shape])
        self.assertIs(shape.members[1], b.shape)
        self.assertIsNone(inner.shape)
        self.assertIsNone(outer.shape)
        self.assertEqual(outer.Topologies(), [inner, b])

    def test_missing_member_shape_fails_without_partial_geometry(self):
        cluster = Cluster.ByTopologies([Topology(shape=Shape()), Topology()])
        self.assertIsNone(cluster.GetOcctShape())

    def test_cyclic_cluster_fails_without_recursion_error(self):
        cluster = Cluster.ByTopologies([Topology(shape=Shape())])
        cluster.topologies.append(cluster)
        self.assertIsNone(cluster.GetOcctShape())

    def test_repeated_member_is_not_mistaken_for_a_cycle(self):
        member = Topology(shape=Shape())
        inner = Cluster.ByTopologies([member])
        cluster = Cluster.ByTopologies([inner, inner])
        self.assertEqual(len(cluster.GetOcctShape().members), 2)

    def test_native_shape_is_returned_without_rebuilding(self):
        topology = Topology(shape=Shape())
        self.assertIs(topology.GetOcctShape(), topology.shape)

    def test_member_edit_preserves_survivors_hierarchy_and_metadata(self):
        removed, retained = Editable(), Editable()
        inner = Cluster.ByTopologies([removed, retained])
        outer = Cluster.ByTopologies([inner, retained])
        outer.SetDictionary({"marker": "root"})
        outer.contents = [retained]
        outer.contexts = ["context"]
        outer.apertures = ["aperture"]
        def outcome(member):
            return True, None if member is removed else member
        # Give nested clusters the same dispatch contract as native editors.
        with patch.object(Cluster, "edit", lambda self, fn: self._EditClusterMembers("edit", fn), create=True):
            status, result = outer._EditClusterMembers("edit", outcome)
        self.assertTrue(status)
        self.assertIsInstance(result.Topologies()[0], Cluster)
        self.assertEqual(result.Topologies()[0].Topologies(), [retained])
        self.assertIs(result.Topologies()[1], retained)
        self.assertEqual(Topology.GetDictionary(result), {"marker": "root"})
        for name in ("contents", "contexts", "apertures"):
            self.assertEqual(getattr(result, name), getattr(outer, name))
            self.assertIsNot(getattr(result, name), getattr(outer, name))
        self.assertEqual(inner.Topologies(), [removed, retained])

    def test_failed_member_aborts_instead_of_dropping_it(self):
        a, b = Editable(), Editable()
        cluster = Cluster.ByTopologies([a, b])
        self.assertEqual(cluster._EditClusterMembers("edit", lambda t: (False, None) if t is b else (True, t)),
                         (False, None))
        self.assertEqual(cluster.Topologies(), [a, b])

    def test_complete_deletion_returns_empty(self):
        cluster = Cluster.ByTopologies([Editable()])
        self.assertEqual(cluster._EditClusterMembers("edit", lambda t: (True, None)), (True, None))

    def test_no_op_edit_preserves_identity(self):
        cluster = Cluster.ByTopologies([Editable()])
        self.assertEqual(cluster._EditClusterMembers("edit", lambda t: (True, t)), (True, cluster))

    def test_edit_recovers_uncached_native_compound_members(self):
        a, b = Editable(shape=Shape()), Editable(shape=Shape())
        shape = Shape()
        shape.members = [a.shape, b.shape]
        cluster = Cluster(shape=shape, topologies=[])
        class Iterator:
            def __init__(self, shape):
                self.members = iter(shape.members)
                self.current = next(self.members, None)

            def More(self):
                return self.current is not None

            def Value(self):
                return self.current

            def Next(self):
                self.current = next(self.members, None)
        with patch.object(sys.modules["OCC.Core.TopoDS"], "TopoDS_Iterator", Iterator, create=True), \
             patch.object(Topology, "ByOcctShape", side_effect=lambda s: a if s is a.shape else b):
            status, result = cluster._EditClusterMembers("edit", lambda t: (True, None if t is a else t))
        self.assertTrue(status)
        self.assertEqual(result.Topologies(), [b])
        self.assertEqual(cluster.Topologies(), [])

    def test_cleanup_reaches_nested_members_and_retains_metadata(self):
        class Member(Topology):
            def Cleanup(self):
                return Editable(dictionary={"cleaned": True})
        inner = Cluster.ByTopologies([Member()])
        inner.SetDictionary({"inner": True})
        outer = Cluster.ByTopologies([inner])
        outer.SetDictionary({"outer": True})
        result = outer.Cleanup()
        self.assertEqual(Topology.GetDictionary(result), {"outer": True})
        cleaned_inner = result.Topologies()[0]
        self.assertEqual(Topology.GetDictionary(cleaned_inner), {"inner": True})
        self.assertEqual(Topology.GetDictionary(cleaned_inner.Topologies()[0]), {"cleaned": True})

    def test_failed_native_wrapping_is_not_reported_as_complete_deletion(self):
        class ReShape:
            def Remove(self, shape):
                pass

            def Apply(self, shape):
                return shape
        shape_build = types.ModuleType("OCC.Core.ShapeBuild")
        shape_build.ShapeBuild_ReShape = ReShape
        member = Topology(shape=Shape())
        with patch.dict(sys.modules, {"OCC.Core.ShapeBuild": shape_build}), \
             patch.object(Topology, "_NativeMatchingShapes", return_value=[member.shape]), \
             patch.object(Topology, "_FinalizeNativeEdit", return_value=None), \
             patch.object(backend, "TopExp_Explorer", return_value=types.SimpleNamespace(More=lambda: True)):
            self.assertEqual(member.RemoveFacesNative([member]), (False, None))

    def test_certified_empty_native_edit_is_reported_as_complete_deletion(self):
        class ReShape:
            def Remove(self, shape):
                pass

            def Apply(self, shape):
                return None
        shape_build = types.ModuleType("OCC.Core.ShapeBuild")
        shape_build.ShapeBuild_ReShape = ReShape
        member = Topology(shape=Shape())
        with patch.dict(sys.modules, {"OCC.Core.ShapeBuild": shape_build}), \
             patch.object(Topology, "_NativeMatchingShapes", return_value=[member.shape]):
            self.assertEqual(member.RemoveFacesNative([member]), (True, None))

    def test_non_null_empty_compound_is_complete_deletion(self):
        member = Topology(shape=Shape())
        with patch.object(Topology, "_FinalizeNativeEdit", return_value=None), \
             patch.object(backend, "TopExp_Explorer", return_value=types.SimpleNamespace(More=lambda: False)):
            self.assertEqual(member._FinalizeNativeRemoval(Shape()), (True, None))

    def test_failed_empty_result_traversal_is_not_complete_deletion(self):
        member = Topology(shape=Shape())
        with patch.object(Topology, "_FinalizeNativeEdit", return_value=None), \
             patch.object(backend, "TopExp_Explorer", side_effect=RuntimeError("traversal failed")):
            self.assertEqual(member._FinalizeNativeRemoval(Shape()), (False, None))


if __name__ == "__main__":
    unittest.main()
