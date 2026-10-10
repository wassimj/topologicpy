"""Bounded, dependency-checked Python calculation reuse for TGraph.

No native compiler or optional numerical package is required. Contexts are opt-in;
existing TGraph methods keep their signatures and uncached fallback paths.
"""
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
import copy
import hashlib
import inspect
import math
import pickle
import sys
import threading


def _freeze(value):
    """Deterministic content key; opaque objects require an explicit token."""
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return (type(value).__name__, value)
    if isinstance(value, float):
        return ('float', value.hex())
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_freeze(v) for v in value))
    if isinstance(value, dict):
        return ('dict', tuple(sorted(((_freeze(k), _freeze(v)) for k, v in value.items()), key=repr)))
    if isinstance(value, (set, frozenset)):
        return ('set', tuple(sorted((_freeze(v) for v in value), key=repr)))
    raise TypeError('Opaque or callable input requires uncached execution or a dependency token')


def _validate_observable(value):
    """Check fingerprint inputs without allocating a second nested structure."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return
    if isinstance(value, dict):
        for k, v in value.items():
            _validate_observable(k)
            _validate_observable(v)
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for v in value:
            _validate_observable(v)
        return
    raise TypeError('Opaque or callable input requires uncached execution or a dependency token')


def _digest(value):
    return hashlib.sha256(repr(_freeze(value)).encode('utf-8')).digest()


def _copy_result(value):
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    from topologicpy.TGraph import TGraph
    if isinstance(value, TGraph):
        g = TGraph(directed=value._directed, allowSelfLoops=value._allow_self_loops,
                   allowParallelEdges=value._allow_parallel_edges)
        # Native originators/representations have the same ownership as normal
        # TGraph copies. Python graph records and dictionaries are independent.
        for name in ('_vertices', '_edges'):
            records = []
            for record in getattr(value, name):
                records.append({k: v if k in ('representation', 'originator') else _copy_result(v)
                                for k, v in record.items()})
            setattr(g, name, records)
        for name in ('_out_edges', '_in_edges', '_incident_edges', '_edge_lookup', '_dictionary'):
            setattr(g, name, _copy_result(getattr(value, name)))
        g._version = value._version
        return g
    if isinstance(value, dict):
        record = 'index' in value and 'dictionary' in value
        return {_copy_result(k): v if record and k in ('representation','originator') else _copy_result(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_result(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_result(v) for v in value)
    if isinstance(value, set):
        return {_copy_result(v) for v in value}
    return copy.deepcopy(value)


def _size(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_size(k, seen) + _size(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(_size(v, seen) for v in value)
    elif hasattr(value, '_vertices') and hasattr(value, '_edges'):
        size += sum(_size(getattr(value, name), seen) for name in
                    ('_vertices', '_edges', '_out_edges', '_in_edges', '_incident_edges', '_edge_lookup', '_dictionary'))
    else:
        # ndarray.nbytes and other buffers are not assumed to be Python objects.
        size = max(size, getattr(value, 'nbytes', 0))
        if hasattr(value,'__dict__'):
            size += _size(vars(value),seen)
    return size


def _live_route_records(value, graph):
    """Routing APIs return live vertex/edge records: preserve that contract."""
    if isinstance(value, dict):
        if 'index' in value and 'dictionary' in value:
            records = graph._edges if 'src' in value and 'dst' in value else graph._vertices
            return records[value['index']]
        return {k:_live_route_records(v,graph) for k,v in value.items()}
    if isinstance(value,list):
        return [_live_route_records(v,graph) for v in value]
    if isinstance(value,tuple):
        return tuple(_live_route_records(v,graph) for v in value)
    return value


class TGraphAnalysis:
    """One graph's explicit, bounded reusable analysis context.

    Create through TGraph.AnalysisContext(graph). Existing compatible TGraph
    calls then reuse this context, or call context.Compute('ClosenessCentrality').
    Relevant live dictionary edits are detected by content, not only _version.
    External mutable geometry/callback calculations need caller-owned tokens.
    """
    def __init__(self, graph, maxEntries=128, maxBytes=64*1024*1024,
                 maxSourceTrees=16, enabled=True):
        from topologicpy.TGraph import TGraph
        if not isinstance(graph, TGraph):
            raise TypeError('graph must be a TGraph')
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0
               for v in (maxEntries, maxBytes, maxSourceTrees)):
            raise ValueError('Cache limits must be nonnegative integers')
        self.graph = graph
        self.maxEntries, self.maxBytes, self.maxSourceTrees = maxEntries, maxBytes, maxSourceTrees
        self.enabled = bool(enabled)
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()
        self._local = threading.local()
        self._counts = {'hits': 0, 'misses': 0, 'evictions': 0, 'bypasses': 0}
        self._namespaces = {}

    @contextmanager
    def _scope(self):
        outer = not getattr(self._local, 'depth', 0)
        if outer:
            self._local.stamps = {}
        self._local.depth = getattr(self._local, 'depth', 0) + 1
        try:
            yield
        finally:
            self._local.depth -= 1
            if outer:
                self._local.stamps = {}

    def Clear(self, resetStatistics=False):
        with self._lock:
            self._entries.clear()
            self._bytes = 0
            if resetStatistics:
                self._counts = dict.fromkeys(self._counts, 0)
                self._namespaces.clear()
        return self

    def Detach(self):
        if self.graph._analysis_context is self:
            self.graph._analysis_context = None
        return self

    def Info(self):
        with self._lock:
            return dict(self._counts, enabled=self.enabled,
                        attached=self.graph._analysis_context is self,
                        entries=len(self._entries), estimatedBytes=self._bytes,
                        maxEntries=self.maxEntries, maxBytes=self.maxBytes,
                        maxSourceTrees=self.maxSourceTrees,
                        namespaces=copy.deepcopy(self._namespaces))

    def _stamp(self, geometry=False, attributes=(), allAttributes=False, graphAttributes=()):
        from topologicpy.TGraph import TGraph
        attrs = tuple(sorted(set(k for k in attributes if isinstance(k, str))))
        cache_key = (bool(geometry), attrs, bool(allAttributes), tuple(graphAttributes))
        stamps = getattr(self._local, 'stamps', {})
        if cache_key in stamps:
            return stamps[cache_key]
        g = self.graph
        topology = (g._directed, g._allow_parallel_edges, g._allow_self_loops,
                    [(v.get('index'), v.get('active', True)) for v in g._vertices],
                    [(e.get('index'), e.get('src'), e.get('dst'), e.get('directed', g._directed),
                      e.get('active', True)) for e in g._edges])
        def values(record):
            d = record.get('dictionary', {})
            return d if allAttributes else {k: d[k] for k in attrs if k in d}
        properties = ([values(v) for v in g._vertices] if attrs or allAttributes else [],
                      [values(e) for e in g._edges] if attrs or allAttributes else [],
                      g._dictionary if allAttributes else {k: g._dictionary[k] for k in graphAttributes if k in g._dictionary})
        coordinates = []
        if geometry:
            from topologicpy.Vertex import Vertex
            from topologicpy.Edge import Edge
            from topologicpy.Topology import Topology
            for v in g._vertices:
                if v.get('active', True):
                    coordinates.append(TGraph.Coordinates(g, v['index']))
            # Segment graphs can put native edges in vertex originators, rather
            # than ordinary graph edges. Validate actual endpoint coordinates.
            for record in list(g._vertices) + list(g._edges):
                obj = record.get('originator')
                if obj is None:
                    obj = record.get('representation')
                if obj is None:
                    coordinates.append(None)
                    continue
                if isinstance(obj,(dict,list,tuple)):
                    coordinates.append(obj)
                    continue
                try:
                    if Topology.IsInstance(obj,'Edge'):
                        coordinates.append((Vertex.Coordinates(Edge.StartVertex(obj)),
                                            Vertex.Coordinates(Edge.EndVertex(obj))))
                    elif Topology.IsInstance(obj,'Vertex'):
                        coordinates.append(Vertex.Coordinates(obj))
                    else:
                        raise TypeError('Unobservable external geometry')
                except Exception:
                    raise TypeError('Unobservable external geometry')
        # Validate observable attribute/geometry types before using pickle only
        # as a fast local encoder. No pickle input is ever loaded/executed.
        _validate_observable(properties)
        _validate_observable(coordinates)
        result = hashlib.sha256(pickle.dumps((topology,properties,coordinates),protocol=5)).digest()
        if getattr(self._local, 'depth', 0):
            stamps[cache_key] = result
        return result

    def _key(self, namespace, settings, **dependencies):
        return (namespace, self._stamp(**dependencies), _digest(settings))

    def _lookup(self, key):
        namespace = key[0]
        counts = self._namespaces.setdefault(namespace, {'hits': 0, 'misses': 0})
        if key not in self._entries:
            self._counts['misses'] += 1
            counts['misses'] += 1
            return False, None
        self._counts['hits'] += 1
        counts['hits'] += 1
        item, size = self._entries.pop(key)
        self._entries[key] = (item, size)
        return True, _copy_result(item)

    def _store(self, key, value):
        if value is None or not self.maxEntries or not self.maxBytes:
            return
        try:
            saved = _copy_result(value)
            size = _size(saved) + _size(key)
        except Exception:
            self._counts['bypasses'] += 1
            return
        if size > self.maxBytes:
            self._counts['bypasses'] += 1
            return
        if key in self._entries:
            self._bytes -= self._entries.pop(key)[1]
        self._entries[key] = (saved, size)
        self._bytes += size
        if key[0] in ('source_tree', 'ShortestPathTree'):
            matching = [k for k in self._entries if k[0] in ('source_tree', 'ShortestPathTree')]
            for victim in matching[:max(0, len(matching)-self.maxSourceTrees)]:
                self._bytes -= self._entries.pop(victim)[1]
                self._counts['evictions'] += 1
        while len(self._entries) > self.maxEntries or self._bytes > self.maxBytes:
            _, (_, removed_size) = self._entries.popitem(last=False)
            self._bytes -= removed_size
            self._counts['evictions'] += 1

    def Memo(self, namespace, settings, calculate, **dependencies):
        """Internal numerical preparation/result reuse; returned values are copies."""
        if not self.enabled:
            return calculate()
        with self._lock, self._scope():
            try:
                key = self._key(namespace, settings, **dependencies)
            except (TypeError, ValueError, RecursionError):
                self._counts['bypasses'] += 1
                return calculate()
            found, value = self._lookup(key)
            if found:
                return value
            value = calculate()
            self._store(key, value)
            return value

    def Publish(self, namespace, settings, value, **dependencies):
        """Make already-calculated raw data available to compatible measures."""
        if self.enabled:
            with self._lock, self._scope():
                try:
                    self._store(self._key(namespace, settings, **dependencies), value)
                except (TypeError, ValueError, RecursionError):
                    self._counts['bypasses'] += 1

    def Compute(self, method, *args, **kwargs):
        from topologicpy.TGraph import TGraph
        if not isinstance(method, str) or method not in CACHE_POLICIES:
            raise ValueError('Method is not a supported reusable TGraph analysis')
        return getattr(TGraph, method)(self.graph, *args, **kwargs)

    def ComputeGeometry(self, method, *args, dependencyToken=None, **kwargs):
        """Reuse external geometry construction with an explicit revision token.

        Increment/change the token whenever external geometry changes. Tokens
        are mandatory because these inputs have no TGraph mutation lifecycle.
        """
        from topologicpy.TGraph import TGraph
        if method not in ('VisibilityGraph', 'NavigationGraph', 'AxialGraph', 'SegmentGraph', 'BySpatialRelationships'):
            raise ValueError('Unsupported external geometry calculation')
        if dependencyToken is None:
            raise ValueError('External geometry reuse requires dependencyToken')
        def external_key(value):
            try:
                _freeze(value)
                return value
            except TypeError:
                return ('external_identity',id(value))
        settings = (dependencyToken, {k:external_key(v) for k,v in kwargs.items()})
        # Token owns the opaque external args; their identities prevent distinct
        # input collections sharing a result accidentally.
        settings = (settings, tuple(id(arg) for arg in args))
        return self.Memo('geometry:'+method, settings,
                         lambda: getattr(TGraph, method)(*args, **kwargs))

    def Centralities(self, weightKey=None, radius=None, mode='out',
                     includeBetweenness=True, mantissa=6):
        """Ordinary vertex measures with compatible distance-work reuse.

        Betweenness follows the existing outgoing directed semantics; other
        modes deliberately use their own closeness searches.
        """
        from topologicpy.TGraph import TGraph
        result = {}
        if includeBetweenness:
            result['betweenness'] = TGraph.BetweennessCentrality(
                self.graph, weightKey=weightKey, radius=radius, normalize=False,
                nxCompatible=False, key=None, colorKey=None, mantissa=mantissa)
        result['closeness'] = TGraph.ClosenessCentrality(
            self.graph, weightKey=weightKey, radius=radius, mode=mode,
            normalize=False, key=None, colorKey=None, mantissa=mantissa)
        result['degree'] = TGraph.DegreeCentrality(
            self.graph, weightKey=weightKey, key=None, colorKey=None, mantissa=mantissa)
        return result

    def _call(self, name, original, signature, args, kwargs, policy):
        if not self.enabled:
            return original(*args, **kwargs)
        with self._lock, self._scope():
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            options = dict(bound.arguments)
            options.pop(next(iter(signature.parameters)), None)
            from topologicpy.TGraph import TGraph
            for field, value in list(options.items()):
                if isinstance(value, TGraph):
                    other = value._analysis_context or TGraphAnalysis(value)
                    try:
                        options[field] = ('other_graph', other._stamp(geometry=True, allAttributes=True))
                    except (TypeError, ValueError, RecursionError):
                        self._counts['bypasses'] += 1
                        return original(*args, **kwargs)
            # Callables can depend on state outside the graph; mutable filters
            # and stochastic computations deliberately retain fallback paths.
            if any(callable(v) for v in options.values()):
                self._counts['bypasses'] += 1
                return original(*args, **kwargs)
            effective = dict(options)
            for parameter in signature.parameters.values():
                if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                    effective.update(options.get(parameter.name, {}))
            if options.get('copy') is False or name in ('InferOntology', 'Reason') and (effective.get('applyToGraph') or effective.get('inplace')):
                self._counts['bypasses'] += 1
                return original(*args, **kwargs)
            if name in ('CommunityPartition', 'Community', 'Partition') and options.get('seed') is None:
                self._counts['bypasses'] += 1
                return original(*args, **kwargs)
            geometry, all_attributes, input_params, outputs = policy
            input_attrs = [options[p] for p in input_params if isinstance(options.get(p), str) and options[p]]
            routes = ('ShortestPath','ShortestPathTree','ShortestPaths','ShortestPathsFromSource',
                      'ShortestPathViaVertices','Path','TopologicalDistance','Depth')
            if name in routes:
                edge_key = str(options.get('edgeKey','')).lower()
                if edge_key in ('hop','hops','unweighted','unit','length','distance','metric'):
                    input_attrs = [key for key in input_attrs if key != options.get('edgeKey')]
                def indexed(value):
                    return value is None or isinstance(value,int) or isinstance(value,dict) and isinstance(value.get('index'),int)
                endpoints_ok = all(indexed(options[p]) for p in ('source','target','vertexA','vertexB','vertex') if p in options)
                endpoints_ok = endpoints_ok and all(indexed(v) for v in (options.get('targets') or []))
                if name in ('ShortestPath','ShortestPathTree','ShortestPathsFromSource') and edge_key in ('hop','hops','unweighted','unit') and endpoints_ok and not options.get('turnWeight') and not options.get('turnKey'):
                    geometry = False
            if name.startswith('Angular') or name.startswith('_Angular'):
                input_attrs += ['depthmap_ref', self.graph._dictionary.get('angular_weight_key', 'angular_weight')]
            fields = [options[p] for p in outputs if isinstance(options.get(p), str)]
            if name in ('AngularChoice', 'AngularIntegration'):
                fields += ['bc_color' if name=='AngularChoice' else 'cc_color']
            if name == 'Choice':
                fields += ['bc_color']
            if name == 'AngularTulipAnalysis' and options.get('writeValues', True):
                fields += ['choice', 'integration', 'nain', 'nach']
                if options.get('colorize', True):
                    fields += ['bc_color', 'cc_color']
            graph_keys = ('graph_type', 'depthmap_segment_data', 'angular_weight_key') if 'Angular' in name else ()
            try:
                key = self._key(name, options, geometry=geometry, attributes=input_attrs,
                                allAttributes=all_attributes, graphAttributes=graph_keys)
            except (TypeError, ValueError, RecursionError):
                self._counts['bypasses'] += 1
                return original(*args, **kwargs)
            found, item = self._lookup(key)
            if found:
                value, patch = item
                for element, index, field, v in patch:
                    records = self.graph._vertices if element=='v' else self.graph._edges
                    records[index].setdefault('dictionary', {})[field] = v
                if name in ('ShortestPath','ShortestPaths','ShortestPathsFromSource','ShortestPathViaVertices','Path'):
                    value = _live_route_records(value,self.graph)
                return value
            value = original(*args, **kwargs)
            if value is None:
                return value
            # Replay only declared presentation outputs, never source geometry
            # or arbitrary graph mutation. Independent return copies avoid cache
            # poisoning through callers editing lists, trees or derived graphs.
            patch = []
            vertex_outputs = name not in ('AngularChoice','AngularIntegration','AngularConnectivity','AngularTulipAnalysis')
            if name in ('BetweennessCentrality','ClosenessCentrality','DegreeCentrality'):
                vertex_outputs = not options.get('useEdges',False)
            elif name.startswith('Angular'):
                vertex_outputs = self.graph._dictionary.get('graph_type') == 'segment'
            selected = None
            if name == 'LocalClusteringCoefficient' and options.get('vertices') is not None:
                selected = {TGraph._as_index(v) for v in options['vertices']}
            element, records = ('v',self.graph._vertices) if vertex_outputs else ('e',self.graph._edges)
            for element, records in ((element,records),):
                for index, record in enumerate(records):
                    if not record.get('active', True):
                        continue
                    if selected is not None and record.get('index') not in selected:
                        continue
                    dictionary = record.get('dictionary', {})
                    for field in set(fields):
                        if field in dictionary:
                            patch.append((element, index, field, dictionary[field]))
            self._store(key, (value, patch))
            return value


# Policy: observes geometry, all metadata, named input-attribute parameters,
# declared output-attribute parameters. Algorithm-specific raw reuse is added
# inside TGraph; these wrappers also reuse unchanged complete calculations.
CACHE_POLICIES = {}
def _policy(names, geometry=False, allAttributes=False, inputs=(), outputs=()):
    for name in names.split():
        CACHE_POLICIES[name] = (geometry, allAttributes, inputs, outputs)

_policy('BetweennessCentrality ClosenessCentrality DegreeCentrality', True,
        inputs=('weightKey','edgeKey','angularWeightKey'), outputs=('key','colorKey'))
_policy('Choice Integration', outputs=('key','colorKey'))
_policy('AngularChoice AngularIntegration AngularTulipAnalysis AngularConnectivity', True, outputs=('key','colorKey'))
_policy('Connectivity', outputs=('key','colorKey'))
_policy('Degree DegreeSequence DegreeMatrix MaximumDelta MinimumDelta Diameter DepthMap BreadthFirstSearch DepthFirstSearch ConnectedComponents IsConnected IsTree IsBipartite IsComplete')
_policy('LocalClusteringCoefficient AverageClusteringCoefficient GlobalClusteringCoefficient', outputs=('key',))
_policy('BiconnectedComponents Bridges CutVertices Leaves IsolatedVertices', allAttributes=True)
_policy('AdjacencyMatrix Laplacian', True, inputs=('vertexKey','edgeKeyFwd','edgeKeyBwd','bidirKey'))
_policy('PageRank EigenvectorCentrality EigenVectorCentrality FiedlerVector FiedlerVectorPartition AccessibilityCentrality', outputs=('key','colorKey'))
_policy('ShortestPath ShortestPathTree ShortestPaths ShortestPathsFromSource ShortestPathViaVertices Path TopologicalDistance Depth', True,
        inputs=('vertexKey','edgeKey','turnKey','viaKey'))
_policy('LineGraph Quotient MinimumSpanningTree Subgraph InducedSubgraph KHopsSubgraph Neigborhood Neighborhood Tree LongestPath Complement Complete', True, True)
_policy('WLFeatures WLKernel HopperKernel WeightedJaccardSimilarity Kernel Compare Match SubGraphMatches IsIsomorphic', True, True)
_policy('MaximumFlow MinimumCut DisjointPaths EdgeConnectivity VertexConnectivity', True,
        inputs=('capacityKey','vertexKey','edgeKey'))
_policy('CommunityPartition Community BetweennessPartition Partition Color ChromaticNumber', True,
        inputs=('weightKey','oldKey'), outputs=('key','colorKey'))
_policy('AABB MetricDistance Distance NearestVertex ContainsVertex AdjacentVerticesByVector AdjacentVerticesByCompassDirection PathLength MeshData', True, True)
_policy('KnowledgeGraph ToKnowledgeGraph RDFGraph SemanticGraph SemanticFingerprint SemanticSummary SemanticDiff OntologyTriples Triples InferOntology Reason ExplainInference ProofGraph ProofGraphData', True, True)
_policy('_SimpleUndirectedNeighborSets _UndirectedAdjacency _NativeEdgeBetweenness')
_policy('_P8GraphArrays _P81HopFeatures _P81GraphEdgeWeights', True, True)
_policy('_FlowNetwork', inputs=('capacityKey',))
_policy('_FlowCosts', True, inputs=('edgeKey','vertexKey'))


def install(TGraph):
    """Install transparent opt-in access without changing public signatures."""
    for name, policy in CACHE_POLICIES.items():
        original = getattr(TGraph, name, None)
        if original is None or getattr(original, '_analysis_wrapped', False):
            continue
        signature = inspect.signature(original)
        if not signature.parameters or next(iter(signature.parameters)) not in ('graph', 'graphA', 'topology'):
            continue
        def make(name, original, signature, policy):
            @wraps(original)
            def wrapped(*args, **kwargs):
                graph = args[0] if args else kwargs.get(next(iter(signature.parameters)))
                context = getattr(graph, '_analysis_context', None)
                if context is None:
                    return original(*args, **kwargs)
                return context._call(name, original, signature, args, kwargs, policy)
            wrapped._analysis_wrapped = True
            return wrapped
        setattr(TGraph, name, staticmethod(make(name, original, signature, policy)))
