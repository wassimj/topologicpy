# Copyright (C) 2026
# TopologicPy
#
# This file is part of TopologicPy.
#
# TopologicPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3.0 of the License, or
# at your option any later version.
#
# TopologicPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

class Ontology:
    """
    A lightweight ontology helper class for TopologicPy.

    The Ontology class provides a dictionary-based semantic layer for TopologicPy
    topologies and graphs. It does not replace TopologicPy dictionaries. Instead,
    it standardises semantic keys such as ontology class, category, label, URI,
    IFC class, IFC GUID, source, and provenance.

    The class intentionally avoids mandatory RDF/OWL dependencies. It can export
    simple RDF Turtle strings/files directly from TopologicPy dictionaries and
    graph connectivity.

    Recommended canonical dictionary keys
    -------------------------------------
    ontology_class : str
        The TopologicPy ontology class, for example "top:Room", "top:Wall",
        "top:AdjacencyGraph". The classes "top:Graph" and "top:TGraph" are treated as aliases; exported instances use "top:Graph" unless a more specific graph subclass is assigned.
    ontology_uri : str
        The full URI for the ontology class or entity.
    label : str
        A human-readable label.
    category : str
        A broad category such as "space", "element", "graph", "surface".
    ifc_class : str
        IFC entity type, for example "IfcSpace".
    ifc_guid : str
        IFC GlobalId.
    source : str
        Source file, database, method, or process.
    derived_from : str
        Identifier or URI of the source object.
    generated_by : str
        Name of the method or process that generated the object.
    """

    # -------------------------------------------------------------------------
    # Canonical dictionary keys
    # -------------------------------------------------------------------------

    ONTOLOGY_CLASS_KEY = "ontology_class"
    ONTOLOGY_URI_KEY = "ontology_uri"
    LABEL_KEY = "label"
    CATEGORY_KEY = "category"
    IFC_CLASS_KEY = "ifc_class"
    IFC_GUID_KEY = "ifc_guid"
    SOURCE_KEY = "source"
    DERIVED_FROM_KEY = "derived_from"
    GENERATED_BY_KEY = "generated_by"
    URI_KEY = "uri"

    # -------------------------------------------------------------------------
    # Namespaces
    # -------------------------------------------------------------------------

    NAMESPACES = {'bot': 'https://w3id.org/bot#',
     'brick': 'https://brickschema.org/schema/Brick#',
     'geo': 'http://www.opengis.net/ont/geosparql#',
     'ifc': 'https://standards.buildingsmart.org/IFC/DEV/IFC4/ADD2_TC1/OWL#',
     'prov': 'http://www.w3.org/ns/prov#',
     'dcterms': 'http://purl.org/dc/terms/',
     'vann': 'http://purl.org/vocab/vann/',
     'skos': 'http://www.w3.org/2004/02/skos/core#',
     'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
     'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
     'xsd': 'http://www.w3.org/2001/XMLSchema#',
     'owl': 'http://www.w3.org/2002/07/owl#',
     'top': 'http://w3id.org/topologicpy#'}


    # -------------------------------------------------------------------------
    # Core ontology hierarchy
    # -------------------------------------------------------------------------

    TOP_SUPERCLASSES = {'top:AccessGraph': ['top:SpatialGraph'],
     'top:AdjacencyGraph': ['top:SpatialGraph'],
     'top:AnalysisGraph': ['top:Graph'],
     'top:AnalysisMetric': [],
     'top:Aperture': ['top:Face', 'top:Element'],
     'top:Attribute': [],
     'top:Beam': ['top:Element'],
     'top:Boundary': ['top:Topology'],
     'top:Building': ['top:Zone'],
     'top:Cell': ['top:Topology'],
     'top:CellComplex': ['top:Topology'],
     'top:CirculationGraph': ['top:SpatialGraph'],
     'top:CirculationZone': ['top:Zone'],
     'top:ClassificationReference': [],
     'top:Cluster': ['top:Topology'],
     'top:Column': ['top:Element'],
     'top:ConnectivityGraph': ['top:SpatialGraph'],
     'top:Context': [],
     'top:CurtainWall': ['top:Wall'],
     'top:Dictionary': [],
     'top:DirectedRelationship': ['top:Relationship'],
     'top:Door': ['top:Element'],
     'top:DualGraph': ['top:SpatialGraph'],
     'top:Edge': ['top:Topology'],
     'top:EdgeFeature': ['top:Attribute'],
     'top:Element': ['top:Topology'],
     'top:Equipment': ['top:Element'],
     'top:ExternalBoundary': ['top:Boundary'],
     'top:Face': ['top:Topology'],
     'top:FunctionalZone': ['top:Zone'],
     'top:Furniture': ['top:Element'],
     'top:Graph': [],
     'top:GraphDataset': [],
     'top:GraphFeature': ['top:Attribute'],
     'top:Grid': [],
     'top:HasseDiagramGraph': ['top:Graph'],
     'top:Interface': ['top:Face'],
     'top:InternalBoundary': ['top:Boundary'],
     'top:Isovist': ['top:AnalysisMetric'],
     'top:IsovistGraph': ['top:SpatialGraph'],
     'top:KnowledgeGraph': ['top:Graph'],
     'top:LineGraph': ['top:Graph'],
     'top:Material': [],
     'top:MaterialSet': [],
     'top:Matrix': [],
     'top:Member': ['top:Element'],
     'top:NavigationGraph': ['top:SpatialGraph'],
     'top:Node': ['top:Vertex'],
     'top:NodeFeature': ['top:Attribute'],
     'top:Opening': ['top:Element'],
     'top:Path': ['top:Graph'],
     'top:Point': ['top:Vertex'],
     'top:Port': [],
     'top:PrimalGraph': ['top:SpatialGraph'],
     'top:Project': [],
     'top:PropertySet': [],
     'top:QualityIssue': [],
     'top:Quantity': [],
     'top:QuotientGraph': ['top:Graph'],
     'top:Railing': ['top:Element'],
     'top:Relationship': ['top:Edge'],
     'top:Roof': ['top:Element'],
     'top:Room': ['top:Space'],
     'top:SemanticGraph': ['top:Graph'],
     'top:Sensor': ['top:Element'],
     'top:Shell': ['top:Topology'],
     'top:Site': ['top:Zone'],
     'top:Slab': ['top:Element'],
     'top:Space': ['top:Zone'],
     'top:SpaceSyntaxMetric': ['top:AnalysisMetric'],
     'top:SpatialGraph': ['top:Graph'],
     'top:Stair': ['top:Element'],
     'top:Storey': ['top:Zone'],
     'top:Surface': ['top:Face'],
     'top:System': [],
     'top:TGraph': ['top:Graph'],
     'top:ThermalZone': ['top:Space'],
     'top:Topology': [],
     'top:TreeGraph': ['top:Graph'],
     'top:UndirectedRelationship': ['top:Relationship'],
     'top:ValidationRule': [],
     'top:Vector': [],
     'top:Vertex': ['top:Topology'],
     'top:VisibilityGraph': ['top:SpatialGraph'],
     'top:Wall': ['top:Element'],
     'top:Window': ['top:Element'],
     'top:Wire': ['top:Topology'],
     'top:Zone': ['top:Cell']}


    # -------------------------------------------------------------------------
    # IFC to TopologicPy ontology mappings
    # -------------------------------------------------------------------------

    IFC_TO_TOP = {'IfcProject': 'top:Project',
     'IfcSite': 'top:Site',
     'IfcBuilding': 'top:Building',
     'IfcBuildingStorey': 'top:Storey',
     'IfcSpace': 'top:Space',
     'IfcZone': 'top:Zone',
     'IfcWall': 'top:Wall',
     'IfcWallStandardCase': 'top:Wall',
     'IfcCurtainWall': 'top:CurtainWall',
     'IfcDoor': 'top:Door',
     'IfcWindow': 'top:Window',
     'IfcSlab': 'top:Slab',
     'IfcRoof': 'top:Roof',
     'IfcColumn': 'top:Column',
     'IfcBeam': 'top:Beam',
     'IfcMember': 'top:Member',
     'IfcStair': 'top:Stair',
     'IfcStairFlight': 'top:Stair',
     'IfcRailing': 'top:Railing',
     'IfcOpeningElement': 'top:Opening',
     'IfcVirtualElement': 'top:Element',
     'IfcFurnishingElement': 'top:Furniture',
     'IfcFurniture': 'top:Furniture',
     'IfcFlowTerminal': 'top:Equipment',
     'IfcDistributionElement': 'top:Equipment',
     'IfcDistributionFlowElement': 'top:Equipment',
     'IfcEnergyConversionDevice': 'top:Equipment',
     'IfcFlowController': 'top:Equipment',
     'IfcFlowFitting': 'top:Equipment',
     'IfcFlowMovingDevice': 'top:Equipment',
     'IfcFlowSegment': 'top:Equipment',
     'IfcFlowStorageDevice': 'top:Equipment',
     'IfcFlowTreatmentDevice': 'top:Equipment',
     'IfcSensor': 'top:Sensor',
     'IfcBuildingElementProxy': 'top:Element',
     'IfcRelSpaceBoundary': 'top:Interface',
     'IfcMaterial': 'top:Material',
     'IfcMaterialLayerSet': 'top:MaterialSet',
     'IfcMaterialProfileSet': 'top:MaterialSet',
     'IfcPropertySet': 'top:PropertySet',
     'IfcElementQuantity': 'top:Quantity',
     'IfcClassificationReference': 'top:ClassificationReference'}


    TOP_TO_BOT = {'top:Building': 'bot:Building',
     'top:Element': 'bot:Element',
     'top:Equipment': 'brick:Equipment',
     'top:Interface': 'bot:Interface',
     'top:Project': 'prov:Entity',
     'top:Sensor': 'brick:Point',
     'top:Site': 'bot:Site',
     'top:Space': 'bot:Space',
     'top:Storey': 'bot:Storey',
     'top:Zone': 'bot:Zone',
     'top:Room': 'bot:Space',
     'top:ThermalZone': 'bot:Zone',
     'top:FunctionalZone': 'bot:Zone',
     'top:CirculationZone': 'bot:Zone',
     'top:Wall': 'bot:Element',
     'top:CurtainWall': 'bot:Element',
     'top:Door': 'bot:Element',
     'top:Window': 'bot:Element',
     'top:Slab': 'bot:Element',
     'top:Roof': 'bot:Element',
     'top:Column': 'bot:Element',
     'top:Beam': 'bot:Element',
     'top:Member': 'bot:Element',
     'top:Stair': 'bot:Element',
     'top:Railing': 'bot:Element',
     'top:Opening': 'bot:Element',
     'top:Aperture': 'bot:Element',
     'top:Furniture': 'bot:Element'}


    TOP_CATEGORIES = {'top:AccessGraph': 'graph',
     'top:AdjacencyGraph': 'graph',
     'top:AnalysisGraph': 'graph',
     'top:AnalysisMetric': 'analysis',
     'top:Aperture': 'topology',
     'top:Attribute': 'metadata',
     'top:Beam': 'element',
     'top:Boundary': 'topology',
     'top:Building': 'building',
     'top:Cell': 'topology',
     'top:CellComplex': 'topology',
     'top:CirculationGraph': 'graph',
     'top:CirculationZone': 'space',
     'top:ClassificationReference': 'metadata',
     'top:Cluster': 'topology',
     'top:Column': 'element',
     'top:ConnectivityGraph': 'graph',
     'top:Context': 'context',
     'top:CurtainWall': 'element',
     'top:Dictionary': 'metadata',
     'top:DirectedRelationship': 'graph',
     'top:Door': 'element',
     'top:DualGraph': 'graph',
     'top:Edge': 'topology',
     'top:EdgeFeature': 'graph',
     'top:Element': 'element',
     'top:Equipment': 'element',
     'top:ExternalBoundary': 'topology',
     'top:Face': 'topology',
     'top:FunctionalZone': 'space',
     'top:Furniture': 'element',
     'top:Graph': 'graph',
     'top:GraphDataset': 'graph',
     'top:GraphFeature': 'graph',
     'top:Grid': 'utility',
     'top:HasseDiagramGraph': 'graph',
     'top:Interface': 'interface',
     'top:InternalBoundary': 'topology',
     'top:Isovist': 'analysis',
     'top:IsovistGraph': 'graph',
     'top:KnowledgeGraph': 'graph',
     'top:LineGraph': 'graph',
     'top:Material': 'metadata',
     'top:MaterialSet': 'metadata',
     'top:Matrix': 'mathematics',
     'top:Member': 'element',
     'top:NavigationGraph': 'graph',
     'top:Node': 'graph',
     'top:NodeFeature': 'graph',
     'top:Opening': 'element',
     'top:Path': 'graph',
     'top:Point': 'topology',
     'top:Port': 'element',
     'top:PrimalGraph': 'graph',
     'top:Project': 'project',
     'top:PropertySet': 'metadata',
     'top:QualityIssue': 'analysis',
     'top:Quantity': 'metadata',
     'top:QuotientGraph': 'graph',
     'top:Railing': 'element',
     'top:Relationship': 'graph',
     'top:Roof': 'element',
     'top:Room': 'space',
     'top:SemanticGraph': 'graph',
     'top:Sensor': 'element',
     'top:Shell': 'topology',
     'top:Site': 'site',
     'top:Slab': 'element',
     'top:Space': 'space',
     'top:SpaceSyntaxMetric': 'analysis',
     'top:SpatialGraph': 'graph',
     'top:Stair': 'element',
     'top:Storey': 'storey',
     'top:Surface': 'topology',
     'top:System': 'element',
     'top:TGraph': 'graph',
     'top:ThermalZone': 'space',
     'top:Topology': 'topology',
     'top:TreeGraph': 'graph',
     'top:UndirectedRelationship': 'graph',
     'top:ValidationRule': 'analysis',
     'top:Vector': 'mathematics',
     'top:Vertex': 'topology',
     'top:VisibilityGraph': 'graph',
     'top:Wall': 'element',
     'top:Window': 'element',
     'top:Wire': 'topology',
     'top:Zone': 'space'}



    # -------------------------------------------------------------------------
    # Ontology specification fragments
    # -------------------------------------------------------------------------

    CLASS_COMMENTS = {'top:AccessGraph': 'A directed or undirected graph used to represent permitted access between spaces, rooms, zones, '
                        'portals, doors, apertures, circulation elements, or other traversable entities. Edges may encode '
                        'passability, access control, traversal direction, door type, cost, distance, or movement '
                        'constraints.',
     'top:AdjacencyGraph': 'An undirected graph used to represent adjacency between topological, geometric, spatial, or '
                           'semantic entities. Edges typically indicate that two entities touch, share a boundary, are '
                           'incident, are proximal within a specified tolerance, or satisfy a declared adjacency '
                           'predicate.',
     'top:AnalysisGraph': 'A directed or undirected graph used to represent entities and relationships prepared for '
                          'computational analysis. Vertices, edges, and graph-level dictionaries may encode analytical '
                          'attributes, weights, labels, features, metrics, simulation values, or machine-learning inputs '
                          'and outputs.',
     'top:AnalysisMetric': 'A computed metric associated with geometry, topology, graphs, buildings, systems, simulation, '
                           'performance analysis, or machine-learning workflows.',
     'top:Aperture': 'A topological or building-domain opening associated with a host topology or element. In TopologicPy '
                     'it is treated as an aperture/opening construct and may be represented geometrically by a face-like '
                     'boundary.',
     'top:Attribute': 'A semantic or analytical key-value attribute associated with a topology, graph, node, relationship, '
                      'dataset, metric, or external entity.',
     'top:Beam': 'A beam, girder, or linear structural member.',
     'top:Boundary': 'A topology used to represent the boundary of another topology, spatial region, element, or '
                     'analytical domain.',
     'top:Building': 'A building-level spatial container used to represent a whole building, facility, or IFC building.',
     'top:Cell': 'A three-dimensional TopologicPy topology bounded by faces or shells and representing a volumetric '
                 'region.',
     'top:CellComplex': 'A TopologicPy topology composed of cells, typically with non-manifold adjacency through shared '
                        'faces.',
     'top:CirculationGraph': 'A directed or undirected graph used to represent movement, circulation, or route-choice '
                             'relationships among spaces, portals, corridors, stairs, entrances, exits, or waypoints. '
                             'Edges may encode route length, cost, capacity, direction, or accessibility.',
     'top:CirculationZone': 'A zone primarily used to represent movement, access, circulation, transition, or evacuation '
                            'functions.',
     'top:ClassificationReference': 'A classification reference used to represent external classification systems, codes, '
                                    'taxonomies, or IFC classification references.',
     'top:Cluster': 'A heterogeneous TopologicPy topology used to group vertices, edges, wires, faces, shells, cells, cell '
                    'complexes, or other clusters without requiring manifold connectivity.',
     'top:Column': 'A column or vertical structural member.',
     'top:ConnectivityGraph': 'A directed or undirected graph used to represent generic connectivity or reachability among '
                              'topological, spatial, building, system, or graph entities. Edges indicate that two entities '
                              'are connected according to an explicit computational or semantic rule.',
     'top:Context': 'A contextual reference used by TopologicPy to associate content with a host topology or subtopology.',
     'top:CurtainWall': 'A curtain-wall element represented as a specialised wall or façade system.',
     'top:Dictionary': 'A key-value metadata container used by TopologicPy to attach semantic, analytical, geometric, '
                       'provenance, and interoperability data to topologies, graphs, vertices, edges, and records.',
     'top:DirectedRelationship': 'A directed graph relationship used to represent an ordered source-to-target relation '
                                 'between two nodes.',
     'top:Door': 'A door, access, or operable passage element, commonly associated with openings, access graphs, and '
                 'navigation relationships.',
     'top:DualGraph': 'An undirected graph used to represent adjacency between higher-dimensional entities such as faces, '
                      'cells, rooms, spaces, zones, or regions. Each entity is represented as a vertex, and an edge '
                      'represents a shared boundary, interface, aperture, adjacency relation, or other declared connection '
                      'between two entities.',
     'top:Edge': 'A one-dimensional TopologicPy topology connecting a start vertex to an end vertex.',
     'top:EdgeFeature': 'A numerical, categorical, vector, or encoded feature associated with a graph edge or relationship '
                        'record.',
     'top:Element': 'A physical, conceptual, or analytical building/infrastructure element that may be represented by '
                    'topology and aligned with BIM entities.',
     'top:Equipment': 'A building, MEP, operational, or service equipment element, optionally aligned with Brick equipment '
                      'classes when system semantics are available.',
     'top:ExternalBoundary': 'A boundary topology used to represent the outer boundary of another topology, region, '
                             'element, or analytical domain.',
     'top:Face': 'A two-dimensional TopologicPy topology bounded by an external wire and, optionally, one or more internal '
                 'boundary wires.',
     'top:FunctionalZone': 'A zone grouped by use, function, programme, activity, department, or other non-geometric '
                           'classification.',
     'top:Furniture': 'A furniture or furnishing element.',
     'top:Graph': 'A directed, undirected, mixed, weighted, unweighted, simple, or multigraph representation used to '
                  'represent entities and relationships emitted, consumed, analysed, or transformed by TopologicPy. More '
                  'specific graph semantics are defined by subclasses.',
     'top:GraphDataset': 'A dataset used to represent graphs, nodes, edges, labels, masks, features, metadata, and '
                         'ontology annotations for graph analysis or graph machine learning.',
     'top:GraphFeature': 'A numerical, categorical, vector, or encoded feature associated with a graph-level record.',
     'top:Grid': 'A spatial subdivision, sampling, or reference structure used to generate or organise vertices, edges, '
                 'cells, or analysis samples.',
     'top:HasseDiagramGraph': 'A directed acyclic graph used to represent a partially ordered set, typically by showing '
                              'only cover relations after transitive reduction. In TopologicPy, it may represent '
                              'hierarchical containment, decomposition, incidence, or boundary relationships among '
                              'topological entities.',
     'top:Interface': 'A shared boundary or interface between spaces, zones, elements, or topologies, including IFC '
                      'space-boundary-style relationships.',
     'top:InternalBoundary': 'A boundary topology used to represent a hole, void, inner loop, or internal boundary of '
                             'another topology, region, element, or analytical domain.',
     'top:Isovist': 'A visibility field, polygon, volume, or related visual metric computed from an observation point or '
                    'region.',
     'top:IsovistGraph': 'A directed or undirected spatial graph used to represent isovists, visibility fields, or '
                         'relationships between observation points and visible spatial extents. Vertices may represent '
                         'viewpoints, isovist polygons, spatial samples, or visible objects, while edges may encode '
                         'visibility, overlap, containment, or intervisibility relationships.',
     'top:KnowledgeGraph': 'A directed semantic graph used to represent entities, attributes, relationships, and inferred '
                           'knowledge. In TopologicPy, it may encode topological objects, BIM entities, ontology classes, '
                           'RDF triples, rules, reasoning outputs, or links between geometry, topology, semantics, '
                           'analysis, and external knowledge sources.',
     'top:LineGraph': 'An undirected or directed graph used to represent edge-to-edge relationships derived from another '
                      'graph. Each vertex in the line graph represents an edge of the source graph, and each edge in the '
                      'line graph represents adjacency, incidence, continuity, succession, or compatibility between source '
                      'edges.',
     'top:Material': 'A material record used to represent material identity, composition, assignment, or IFC material '
                     'information.',
     'top:MaterialSet': 'A material-set record used to represent material lists, material layer sets, material constituent '
                        'sets, or other grouped material definitions.',
     'top:Matrix': 'A numerical matrix used to represent transformations, coordinate operations, or other matrix-valued '
                   'quantities.',
     'top:Member': 'A generic structural, framing, or linear member when a more specific class such as Beam or Column is '
                   'not asserted.',
     'top:NavigationGraph': 'A directed or undirected weighted graph used to represent navigable movement through spaces, '
                            'paths, portals, waypoints, circulation systems, or walkable regions. Edges typically encode '
                            'traversability and may store distance, travel time, slope, accessibility, obstruction, cost, '
                            'or agent-specific movement constraints.',
     'top:Node': 'A graph node represented by, or projected from, a TopologicPy vertex or TGraph vertex record. Nodes may '
                 'carry dictionary metadata, ontology class information, labels, features, coordinates, and external '
                 'identifiers.',
     'top:NodeFeature': 'A numerical, categorical, vector, or encoded feature associated with a graph node or vertex '
                        'record.',
     'top:Opening': 'An opening element, void, recess, penetration, or host-element subtraction associated with doors, '
                    'windows, services, or coordination checks.',
     'top:Path': 'An ordered graph structure used to represent a sequence of nodes and relationships, typically returned '
                 'by routing, shortest-path, circulation, or navigation methods.',
     'top:Point': 'A point-like spatial entity represented as a specialised vertex.',
     'top:Port': 'A distribution or connection port used to represent connection points on systems, equipment, MEP '
                 'components, or IFC distribution ports.',
     'top:PrimalGraph': 'A directed or undirected graph used to represent direct physical, spatial, topological, or '
                        'semantic connectivity among entities in their original form. In spatial and topological '
                        'modelling, vertices and edges typically correspond directly to objects, points, spaces, elements, '
                        'or their explicit connections, before any dual transformation is applied.',
     'top:Project': 'A project-level container used to represent a model, dataset, building project, computational study, '
                    'or IFC project.',
     'top:PropertySet': 'A property set used to represent grouped name-value properties, commonly mapped from IFC property '
                        'sets or related property-definition entities.',
     'top:QualityIssue': 'A validation, model-checking, coordination, geometry, topology, semantic, or data-quality issue '
                         'detected during analysis.',
     'top:Quantity': 'A quantity record used to represent measured or computed quantities, commonly mapped from IFC '
                     'element quantities.',
     'top:QuotientGraph': 'A directed or undirected graph used to represent a simplified graph obtained by grouping '
                          'vertices, edges, spaces, zones, components, or other entities according to an equivalence '
                          'relation or partition. Vertices represent groups or classes, and edges represent relationships '
                          'between those groups inherited from the source graph.',
     'top:Railing': 'A railing, guard, handrail, or balustrade element.',
     'top:Relationship': 'A graph relationship represented by, or projected from, a TopologicPy edge or TGraph edge '
                         'record. Relationships may carry source and target references, directionality, weights, '
                         'predicates, labels, features, and dictionary metadata.',
     'top:Roof': 'A roof or roof-like envelope element.',
     'top:Room': 'A room-like occupiable space represented as a specialised building space.',
     'top:SemanticGraph': 'A directed graph used to represent typed entities and typed relationships with explicit '
                          'semantic meaning. In TopologicPy, it may encode classes, instances, properties, labels, '
                          'attributes, ontology terms, BIM semantics, or RDF-style subject-predicate-object statements.',
     'top:Sensor': 'A sensor, observation point, or measurement device associated with a space, element, system, or '
                   'analytical process.',
     'top:Shell': 'A two-dimensional TopologicPy topology composed of connected faces, typically forming an open or closed '
                  'segmented surface.',
     'top:Site': 'A site-level spatial container used to represent land, context, parcels, infrastructure extents, or an '
                 'IFC site.',
     'top:Slab': 'A slab, floor, ceiling, plate, or horizontal building element.',
     'top:Space': 'A bounded or occupiable spatial region used to represent rooms, spaces, compartments, or IFC spaces.',
     'top:SpaceSyntaxMetric': 'A spatial network metric used in space syntax, visibility analysis, access analysis, or '
                              'graph-based spatial analysis.',
     'top:SpatialGraph': 'A directed or undirected graph used to represent spatial entities and spatial relationships. '
                         'Relationships may include metric, topological, directional, containment, proximity, overlap, '
                         'visibility, accessibility, or connectivity predicates, including relations aligned with OGC '
                         'Simple Features, ISO 19107 spatial schema concepts, DE-9IM predicates, or RCC-8 qualitative '
                         'spatial relations.',
     'top:Stair': 'A stair, stair flight, or vertical circulation element.',
     'top:Storey': 'A storey-level spatial container used to represent a building level, floor, or IFC building storey.',
     'top:Surface': 'A surface-like spatial, analytical, or building-domain entity represented as a specialised face.',
     'top:System': 'A building, distribution, MEP, operational, or analytical system that groups elements, equipment, '
                   'ports, spaces, or functions.',
     'top:TGraph': 'A TopologicPy-native graph representation used to represent indexed vertices, indexed edges, graph '
                   'dictionaries, optional topology representations, and ontology-aware metadata. This class distinguishes '
                   'the Python TGraph implementation from the more general top:Graph concept.',
     'top:ThermalZone': 'A thermal analysis zone represented as a specialised space used in environmental, energy, HVAC, '
                        'or comfort analysis.',
     'top:Topology': 'Superclass for TopologicPy topological entities, including vertices, edges, wires, faces, shells, '
                     'cells, cell complexes, clusters, apertures, and related topological specialisations.',
     'top:TreeGraph': 'A connected acyclic graph used to represent hierarchical, branching, containment, decomposition, '
                      'classification, or dependency structures. In TopologicPy and BIM contexts, it may represent project '
                      'breakdowns, spatial decomposition, assembly hierarchies, taxonomy structures, or rooted '
                      'parent-child relationships.',
     'top:UndirectedRelationship': 'An undirected graph relationship used to represent a symmetric or '
                                   'direction-independent relation between two nodes.',
     'top:ValidationRule': 'A rule used to validate geometry, topology, semantics, IFC/BIM content, graph structure, '
                           'provenance, or analytical metadata.',
     'top:Vector': 'A mathematical vector used to represent direction, displacement, orientation, or other vector-valued '
                   'quantities.',
     'top:Vertex': 'A zero-dimensional TopologicPy topology representing a point-like entity, commonly defined by X, Y, '
                   'and Z coordinates.',
     'top:VisibilityGraph': 'An undirected or directed spatial graph used to represent line-of-sight visibility or '
                            'intervisibility between points, locations, spaces, surfaces, objects, or sampled positions. '
                            'Edges indicate that two entities are mutually visible or directionally visible, and may store '
                            'visibility distance, angular range, obstruction status, or visual weight.',
     'top:Wall': 'A wall or wall-like building element used to bound, separate, support, enclose, or subdivide spaces.',
     'top:Window': 'A window or transparent/translucent opening element, commonly associated with daylight, view, '
                   'envelope, and façade relationships.',
     'top:Wire': 'A one-dimensional TopologicPy topology composed of connected edges, typically forming an open path or a '
                 'closed boundary.',
     'top:Zone': 'A spatial region, bounded domain, or analytical zone used to organise spaces, buildings, storeys, sites, '
                 'thermal zones, or functional regions.'}


    OBJECT_PROPERTIES = {'top:adjacentTo': ('top:Topology',
                        'top:Topology',
                        'Associates two topologies, spaces, regions, or elements that are adjacent according to a declared '
                        'spatial, topological, or tolerance-based rule.'),
     'top:aggregates': ('top:Topology',
                        'top:Topology',
                        'Represents a whole-part, decomposition, or aggregation relationship, commonly mapped from IFC '
                        'aggregation or decomposition relations.'),
     'top:connects': ('top:Topology',
                      'top:Topology',
                      'Generic semantic connection used when a relationship is known but more specific semantics are '
                      'unavailable.'),
     'top:connectsPort': ('top:Port',
                          'top:Port',
                          'Connects two ports without asserting flow direction. This is the preferred direct mapping for '
                          'port-to-port connectivity such as IfcRelConnectsPorts.'),
     'top:connectsTo': ('top:Topology',
                        'top:Topology',
                        'Generic undirected topological or graph connectivity between two topologies, vertices, nodes, '
                        'elements, spaces, or other entities.'),
     'top:containsElement': ('top:Topology',
                             'top:Topology',
                             'Associates a spatial, topological, or semantic container with a contained topology, element, '
                             'space, or entity.'),
     'top:derivedFrom': ('owl:Thing',
                         'owl:Thing',
                         'Associates an entity, topology, graph, or record with the source entity, model, file, process, '
                         'or data object from which it was derived.'),
     'top:endsAt': (['top:Edge', 'top:Relationship'],
                    'top:Vertex',
                    'Alias property for associating an edge or relationship with its end vertex or target node.'),
     'top:fillsOpening': ('top:Element',
                          'top:Opening',
                          'Associates an element such as a door, window, or service component with the opening it fills.'),
     'top:generatedBy': ('owl:Thing',
                         'owl:Thing',
                         'Associates an entity, topology, graph, or record with the method, script, notebook, process, or '
                         'software operation that generated it.'),
     'top:hasApproval': ('owl:Thing',
                         'owl:Thing',
                         'Associates an entity with an approval, review, authorisation, or sign-off record.'),
     'top:hasCell': ('top:Topology', 'top:Cell', 'Associates a topology with a constituent cell.'),
     'top:hasCellComplex': ('top:Cluster',
                            'top:CellComplex',
                            'Associates a cluster or model container with a constituent cell complex.'),
     'top:hasClassification': ('top:Topology',
                               'top:ClassificationReference',
                               'Associates a topology, element, system, or mapped BIM entity with a classification '
                               'reference.'),
     'top:hasConnectedPort': ('top:Element',
                              'top:Port',
                              'Associates an element, system component, or equipment item with a connected distribution or '
                              'connection port.'),
     'top:hasConstraint': ('owl:Thing',
                           'owl:Thing',
                           'Associates an entity with a rule, constraint, requirement, limit, or validation condition.'),
     'top:hasCoordinationIssue': ('top:Topology',
                                  'top:Relationship',
                                  'Associates an entity with a detected coordination, clash, validation, or quality '
                                  'issue.'),
     'top:hasDictionary': (['top:Topology', 'top:Graph'],
                           'top:Dictionary',
                           'Associates a topology, graph, node, relationship, or record with a TopologicPy dictionary '
                           'containing metadata, attributes, semantics, analysis values, or provenance.'),
     'top:hasDocument': ('owl:Thing',
                         'owl:Thing',
                         'Associates an entity with a document reference, external file, specification, drawing, approval '
                         'package, or supporting document.'),
     'top:hasEdge': (['top:Topology', 'top:Graph'],
                     'top:Edge',
                     'Associates a topology or graph with an edge that belongs to it.'),
     'top:hasEndVertex': (['top:Edge', 'top:Relationship'],
                          'top:Vertex',
                          'Associates an edge or relationship with its end vertex or target node.'),
     'top:hasExternalBoundary': ('top:Topology',
                                 'top:Boundary',
                                 'Associates a topology, region, element, or analytical domain with its external '
                                 'boundary.'),
     'top:hasFace': ('top:Topology', 'top:Face', 'Associates a topology with a constituent face.'),
     'top:hasIFCType': ('top:Topology',
                        'owl:Thing',
                        'Associates an IFC occurrence or mapped topology with its IFC type object.'),
     'top:hasInternalBoundary': ('top:Topology',
                                 'top:Boundary',
                                 'Associates a topology, region, element, or analytical domain with an internal boundary, '
                                 'hole, or void boundary.'),
     'top:hasMaterial': ('top:Topology',
                         ['top:Material', 'top:MaterialSet'],
                         'Associates a topology, element, or mapped BIM entity with a material or material set.'),
     'top:hasMissingOpening': ('top:Topology',
                               'top:Element',
                               'Associates an element or topology with a coordination issue in which an expected opening '
                               'is absent.'),
     'top:hasNode': ('top:Graph', 'top:Node', 'Associates a graph with a node that belongs to it.'),
     'top:hasOpening': ('top:Element',
                        'top:Opening',
                        'Associates an element with an opening, void, penetration, or recess.'),
     'top:hasPredicate': ('top:Relationship',
                          'rdf:Property',
                          'Associates a TopologicPy relationship record with the RDF predicate that gives the relationship '
                          'its semantic meaning.'),
     'top:hasPropertySet': (['top:Topology', 'top:System', 'top:Relationship'],
                            'top:PropertySet',
                            'Associates a topology, element, type, system, graph entity, or relationship with a property '
                            'set.'),
     'top:hasRelationship': ('top:Graph',
                             'top:Relationship',
                             'Associates a graph with a relationship or edge that belongs to it.'),
     'top:hasShell': ('top:Topology', 'top:Shell', 'Associates a topology with a constituent shell.'),
     'top:hasStartVertex': (['top:Edge', 'top:Relationship'],
                            'top:Vertex',
                            'Associates an edge or relationship with its start vertex or source node.'),
     'top:hasSubTopology': ('top:Topology',
                            'top:Topology',
                            'Associates a topology with a contained or constituent subtopology.'),
     'top:hasTopology': ('owl:Thing',
                         'top:Topology',
                         'Associates an entity with a topology that geometrically or topologically represents it.'),
     'top:hasVertex': (['top:Topology', 'top:Graph'],
                       'top:Vertex',
                       'Associates a topology or graph with a vertex that belongs to it.'),
     'top:hasWire': ('top:Topology', 'top:Wire', 'Associates a topology with a constituent wire.'),
     'top:interfaceOf': ('top:Interface',
                         'top:Topology',
                         'Associates an interface with the topology, element, space, or zone that it bounds, separates, or '
                         'connects.'),
     'top:intersects': ('top:Topology',
                        'top:Topology',
                        'Associates two topologies, elements, spaces, or regions that geometrically or topologically '
                        'intersect according to a declared tolerance or spatial predicate.'),
     'top:isAggregatedBy': ('top:Topology',
                            'top:Topology',
                            'Inverse relation of top:aggregates, associating a part with its aggregate or whole.'),
     'top:isApprovalOf': ('owl:Thing', 'owl:Thing', 'Inverse relation of top:hasApproval.'),
     'top:isCellComplexOf': ('top:CellComplex',
                             'top:Cluster',
                             'Associates a cell complex with a containing cluster or model container.'),
     'top:isCellOf': ('top:Cell', 'top:Topology', 'Associates a cell with its parent topology.'),
     'top:isClassificationOf': ('top:ClassificationReference',
                                'top:Topology',
                                'Inverse relation of top:hasClassification.'),
     'top:isConnectedPortOf': ('top:Port',
                               'top:Port',
                               'Inverse or companion relation for top:connectsPort where a directional statement is '
                               'required by an export process.'),
     'top:isConnectedTo': ('top:Topology',
                           'top:Topology',
                           'Alias property for generic semantic or topological connection.'),
     'top:isConstraintOf': ('owl:Thing', 'owl:Thing', 'Inverse relation of top:hasConstraint.'),
     'top:isDocumentOf': ('owl:Thing', 'owl:Thing', 'Inverse relation of top:hasDocument.'),
     'top:isEdgeOf': ('top:Edge',
                      ['top:Topology', 'top:Graph'],
                      'Associates an edge with the topology or graph to which it belongs.'),
     'top:isFaceOf': ('top:Face', 'top:Topology', 'Associates a face with its parent topology.'),
     'top:isFilledBy': ('top:Opening',
                        'top:Element',
                        'Inverse relation of top:fillsOpening, associating an opening with the element that fills it.'),
     'top:isIFCTypeOf': ('owl:Thing', 'top:Topology', 'Inverse relation of top:hasIFCType.'),
     'top:isMaterialOf': (['top:Material', 'top:MaterialSet'], 'top:Topology', 'Inverse relation of top:hasMaterial.'),
     'top:isOpeningIn': ('top:Opening',
                         'top:Element',
                         'Inverse relation of top:hasOpening, associating an opening with its host element.'),
     'top:isPartOf': ('top:Topology',
                      'top:Topology',
                      'Associates a topology, element, space, or entity with a containing or aggregating whole.'),
     'top:isPropertySetOf': ('top:PropertySet',
                             ['top:Topology', 'top:System', 'top:Relationship'],
                             'Inverse relation of top:hasPropertySet.'),
     'top:isServedBy': ('top:Topology',
                        ['top:System', 'top:Equipment'],
                        'Associates a spatial structure with the system or equipment item that serves it.'),
     'top:isShellOf': ('top:Shell', 'top:Topology', 'Associates a shell with its parent topology.'),
     'top:isSubTopologyOf': ('top:Topology', 'top:Topology', 'Associates a topology with a containing or parent topology.'),
     'top:isTopologyOf': ('top:Topology',
                          'owl:Thing',
                          'Inverse relation of top:hasTopology, associating a topology with the entity it represents.'),
     'top:isVertexOf': ('top:Vertex',
                        ['top:Topology', 'top:Graph'],
                        'Associates a vertex with the topology or graph to which it belongs.'),
     'top:isWireOf': ('top:Wire', 'top:Topology', 'Associates a wire with its parent topology.'),
     'top:locatedIn': ('top:Topology',
                       'top:Topology',
                       'Associates a topology, element, node, or entity with the containing, nearest, or inferred spatial '
                       'structure derived by geometric or semantic analysis.'),
     'top:passesThrough': ('top:Topology',
                           'top:Topology',
                           'Indicates that one topology, element, or system component passes through another topology, '
                           'element, space, or region.'),
     'top:requiresOpening': ('top:Topology',
                             'top:Element',
                             'Indicates that an element, system component, route, or topology requires an opening through '
                             'another element.'),
     'top:servesBuilding': (['top:System', 'top:Equipment'],
                            'top:Building',
                            'Associates a system or equipment item with a building it serves.'),
     'top:servesSpatialStructure': (['top:System', 'top:Equipment'],
                                    'top:Topology',
                                    'Associates a system or equipment item with the spatial structure it serves, such as a '
                                    'site, building, storey, space, or zone.'),
     'top:startsAt': (['top:Edge', 'top:Relationship'],
                      'top:Vertex',
                      'Alias property for associating an edge or relationship with its start vertex or source node.'),
     'top:violatesCoordinationRule': ('top:Topology',
                                      'top:Relationship',
                                      'Associates an entity with a violated coordination rule, model-checking rule, or '
                                      'relationship record.')}


    DATA_PROPERTIES = {'top:area': ('top:Topology',
                  'xsd:double',
                  'Alias data property for area when TopologicPy dictionary export emits the raw key area.'),
     'top:category': ('owl:Thing',
                      'xsd:string',
                      'A broad category value emitted from TopologicPy dictionaries, such as topology, graph, space, '
                      'element, equipment, interface, project, metadata, mathematics, or analysis.'),
     'top:createdAt': ('owl:Thing', 'xsd:dateTime', 'The creation timestamp of an entity, topology, graph, or record.'),
     'top:description': ('owl:Thing', 'xsd:string', 'A human-readable description emitted from a TopologicPy dictionary.'),
     'top:hasArea': ('top:Topology',
                     'xsd:double',
                     'The area of a face, shell, cell, cell complex, surface, spatial region, or other area-bearing '
                     'topology or analytical record.'),
     'top:hasLength': ('top:Topology',
                       'xsd:double',
                       'The length of an edge, wire, path, graph edge, or other length-bearing topology or analytical '
                       'record.'),
     'top:hasMantissa': ('owl:Thing',
                         'xsd:integer',
                         'The number of decimal places used to round, serialize, compare, or report numeric values.'),
     'top:hasUnit': ('owl:Thing',
                     'xsd:string',
                     'The unit of measurement associated with a value, topology, graph, metric, or record.'),
     'top:hasVolume': ('top:Topology',
                       'xsd:double',
                       'The volume of a cell, cell complex, zone, space, or other volume-bearing topology or analytical '
                       'record.'),
     'top:hasX': ('top:Vertex', 'xsd:double', 'The X coordinate of a vertex, point, node, or graph vertex record.'),
     'top:hasY': ('top:Vertex', 'xsd:double', 'The Y coordinate of a vertex, point, node, or graph vertex record.'),
     'top:hasZ': ('top:Vertex', 'xsd:double', 'The Z coordinate of a vertex, point, node, or graph vertex record.'),
     'top:ifcClass': ('owl:Thing',
                      'xsd:string',
                      'The IFC entity class name associated with a topology, graph entity, or record, commonly stored '
                      'under the dictionary key ifc_class.'),
     'top:ifcGUID': ('owl:Thing',
                     'xsd:string',
                     'The IFC GlobalId associated with a topology, graph entity, or record, commonly stored under the '
                     'dictionary key ifc_guid.'),
     'top:label': ('owl:Thing',
                   'xsd:string',
                   'A human-readable label emitted from a TopologicPy dictionary when represented as data rather than '
                   'rdfs:label.'),
     'top:length': ('top:Topology',
                    'xsd:double',
                    'Alias data property for length when TopologicPy dictionary export emits the raw key length.'),
     'top:mantissa': ('owl:Thing',
                      'xsd:integer',
                      'Alias data property for mantissa when TopologicPy dictionary export emits the raw key mantissa.'),
     'top:modifiedAt': ('owl:Thing',
                        'xsd:dateTime',
                        'The last modification timestamp of an entity, topology, graph, or record.'),
     'top:name': ('owl:Thing', 'xsd:string', 'A human-readable name emitted from a TopologicPy dictionary.'),
     'top:ontologyClass': ('owl:Thing',
                           'xsd:string',
                           'The ontology class QName or URI recorded in a TopologicPy dictionary, commonly stored under '
                           'the dictionary key ontology_class.'),
     'top:ontologyURI': ('owl:Thing',
                         'xsd:anyURI',
                         'The expanded ontology URI recorded in a TopologicPy dictionary, commonly stored under the '
                         'dictionary key ontology_uri.'),
     'top:relationship': ('owl:Thing',
                          'xsd:string',
                          'A general-purpose relationship label emitted from TopologicPy dictionaries when a more specific '
                          'ontology predicate is not available.'),
     'top:source': ('owl:Thing',
                    'xsd:string',
                    'The source file, model, database, method, or process associated with an entity, topology, graph, or '
                    'record.'),
     'top:unit': ('owl:Thing',
                  'xsd:string',
                  'Alias data property for unit when TopologicPy dictionary export emits the raw key unit.'),
     'top:volume': ('top:Topology',
                    'xsd:double',
                    'Alias data property for volume when TopologicPy dictionary export emits the raw key volume.'),
     'top:x': ('top:Vertex',
               'xsd:double',
               'Alias data property for the X coordinate when TopologicPy dictionary export emits the raw key x.'),
     'top:y': ('top:Vertex',
               'xsd:double',
               'Alias data property for the Y coordinate when TopologicPy dictionary export emits the raw key y.'),
     'top:z': ('top:Vertex',
               'xsd:double',
               'Alias data property for the Z coordinate when TopologicPy dictionary export emits the raw key z.')}


    PROPERTY_ALIASES = {'hasStartVertex': 'startsAt',
     'hasEndVertex': 'endsAt',
     'startVertex': 'startsAt',
     'endVertex': 'endsAt',
     'hasVertices': 'hasVertex',
     'hasEdges': 'hasEdge',
     'hasWires': 'hasWire',
     'hasFaces': 'hasFace',
     'hasShells': 'hasShell',
     'hasCells': 'hasCell',
     'hasCellComplexes': 'hasCellComplex',
     'connectedTo': 'connectsTo',
     'adjacent': 'adjacentTo',
     'contains': 'containsElement',
     'containedIn': 'isPartOf',
     'ontology_class': 'ontologyClass',
     'ontology_uri': 'ontologyURI',
     'ifc_class': 'ifcClass',
     'ifc_guid': 'ifcGUID',
     'derived_from': 'derivedFrom',
     'generated_by': 'generatedBy',
     'created_at': 'createdAt',
     'modified_at': 'modifiedAt',
     'source': 'source',
     'label': 'label',
     'category': 'category',
     'relationship': 'relationship',
     'x': 'hasX',
     'y': 'hasY',
     'z': 'hasZ',
     'length': 'hasLength',
     'area': 'hasArea',
     'volume': 'hasVolume',
     'mantissa': 'hasMantissa',
     'unit': 'hasUnit',
     'weight': 'hasWeight',
     'feature': 'hasFeature',
     'feature_vector': 'hasFeatureVector'}


    CLASS_ALIASES = {'top:TGraph': 'top:Graph', 'TGraph': 'top:Graph', 'Graph': 'top:Graph'}

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _dictionary(topology):
        """Returns a TopologicPy or Python dictionary from the input topology or graph."""
        if topology is None:
            return None

        record_dict = Ontology._record_dictionary(topology)
        if record_dict is not None:
            return record_dict

        if Ontology._is_tgraph(topology):
            try:
                from topologicpy.TGraph import TGraph
                return TGraph.Dictionary(topology)
            except Exception:
                return None

        try:
            from topologicpy.Topology import Topology
            return Topology.Dictionary(topology)
        except Exception:
            return None
    @staticmethod
    def _is_tgraph(obj):
        """Returns True if the input object is a TGraph."""
        try:
            from topologicpy.TGraph import TGraph
            return isinstance(obj, TGraph)
        except Exception:
            return False

    @staticmethod
    def _is_graph_like(obj):
        """Returns True if the input object is a legacy Graph or TGraph."""
        if obj is None:
            return False
        if Ontology._is_tgraph(obj):
            return True
        try:
            from topologicpy.Topology import Topology
            return bool(Topology.IsInstance(obj, "graph"))
        except Exception:
            return False

    @staticmethod
    def _graph_vertices(graph, asTopologic: bool = False):
        """Returns graph vertices from either a legacy Graph or TGraph."""
        if graph is None:
            return []
        if Ontology._is_tgraph(graph):
            try:
                from topologicpy.TGraph import TGraph
                return TGraph.Vertices(graph, asTopologic=asTopologic, active=True) or []
            except Exception:
                return []
        try:
            from topologicpy.Graph import Graph
            return Graph.Vertices(graph) or []
        except Exception:
            return []

    @staticmethod
    def _graph_edges(graph, asTopologic: bool = False):
        """Returns graph edges from either a legacy Graph or TGraph."""
        if graph is None:
            return []
        if Ontology._is_tgraph(graph):
            try:
                from topologicpy.TGraph import TGraph
                return TGraph.Edges(graph, asTopologic=asTopologic, active=True) or []
            except Exception:
                return []
        try:
            from topologicpy.Graph import Graph
            return Graph.Edges(graph) or []
        except Exception:
            return []

    @staticmethod
    def _graph_edge_endpoints(graph, edge, vertex_records=None):
        """Returns the start and end vertices/records of a graph edge."""
        if graph is None or edge is None:
            return None, None
        if Ontology._is_tgraph(graph):
            try:
                src = edge.get("src", None) if isinstance(edge, dict) else None
                dst = edge.get("dst", None) if isinstance(edge, dict) else None
                if isinstance(src, int) and isinstance(dst, int):
                    if vertex_records is None:
                        vertex_records = Ontology._graph_vertices(graph, asTopologic=False)
                    by_index = {}
                    for i, v in enumerate(vertex_records or []):
                        if isinstance(v, dict):
                            by_index[v.get("index", i)] = v
                    return by_index.get(src), by_index.get(dst)
            except Exception:
                return None, None
            return None, None
        try:
            from topologicpy.Edge import Edge
            return Edge.StartVertex(edge), Edge.EndVertex(edge)
        except Exception:
            pass
        try:
            from topologicpy.Topology import Topology
            return Topology.StartVertex(edge), Topology.EndVertex(edge)
        except Exception:
            pass
        try:
            from topologicpy.Graph import Graph
            return Graph.StartVertex(graph, edge), Graph.EndVertex(graph, edge)
        except Exception:
            return None, None

    @staticmethod
    def _record_dictionary(obj):
        """Returns a Python dictionary from a TGraph record-like object."""
        if isinstance(obj, dict):
            d = obj.get("dictionary", obj)
            if isinstance(d, dict):
                return d
        return None

    @staticmethod
    def _record_coordinates(obj, default=None):
        """Returns XYZ coordinates from a TGraph vertex record-like object."""
        default = default if default is not None else [0.0, 0.0, 0.0]
        if isinstance(obj, dict):
            d = obj.get("dictionary", obj)
            if isinstance(d, dict):
                x = d.get("x", obj.get("x", None))
                y = d.get("y", obj.get("y", None))
                z = d.get("z", obj.get("z", None))
                try:
                    return [float(x), float(y), float(z)]
                except Exception:
                    pass
            try:
                coords = obj.get("coordinates", None)
                if isinstance(coords, (list, tuple)) and len(coords) >= 3:
                    return [float(coords[0]), float(coords[1]), float(coords[2])]
            except Exception:
                pass
        return list(default)

    @staticmethod
    def _set_value(topology, key, value, silent=False):
        """Sets a dictionary key/value pair on the input topology, graph, or TGraph record."""
        try:
            from topologicpy.Dictionary import Dictionary
        except Exception:
            Dictionary = None

        if topology is None:
            if not silent:
                print("Ontology._set_value - Error: The input topology is None. Returning None.")
            return None

        if key is None:
            if not silent:
                print("Ontology._set_value - Error: The input key is None. Returning None.")
            return None

        if isinstance(topology, dict):
            d = topology.get("dictionary", topology)
            if not isinstance(d, dict):
                d = {}
            d[key] = value
            if "dictionary" in topology:
                topology["dictionary"] = d
            return topology

        if Ontology._is_tgraph(topology):
            try:
                from topologicpy.TGraph import TGraph
                d = TGraph.Dictionary(topology)
                d[key] = value
                topology.SetDictionary(d)
                return topology
            except Exception as e:
                if not silent:
                    print("Ontology._set_value - Error: Could not set the TGraph dictionary value. Returning None.")
                    print("Error:", e)
                return None

        try:
            from topologicpy.Topology import Topology
            d = Topology.Dictionary(topology)
            d = Dictionary.SetValueAtKey(d, key, value)
            Topology.SetDictionary(topology, d)
            return topology
        except Exception as e:
            if not silent:
                print("Ontology._set_value - Error: Could not set the dictionary value. Returning None.")
                print("Error:", e)
            return None

    @staticmethod
    def _value(topology, key, defaultValue=None):
        """Returns a dictionary value from the input topology, graph, or TGraph record."""
        d = Ontology._dictionary(topology)
        if isinstance(d, dict):
            return d.get(key, defaultValue)
        try:
            from topologicpy.Dictionary import Dictionary
            return Dictionary.ValueAtKey(d, key, defaultValue)
        except Exception:
            return defaultValue

    @staticmethod
    def _as_list(item):
        """Returns item as a list."""
        if item is None:
            return []
        if isinstance(item, list):
            return item
        if isinstance(item, tuple):
            return list(item)
        return [item]

    @staticmethod
    def _safe_string(value):
        """Returns a safe string representation for RDF literals and identifiers."""
        if value is None:
            return ""
        return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")

    @staticmethod
    def _safe_local_name(value):
        """Returns a conservative RDF local name."""
        import re

        if value is None:
            return "unnamed"
        s = str(value).strip()
        if s == "":
            return "unnamed"
        s = re.sub(r"[^A-Za-z0-9_\\-]", "_", s)
        if len(s) == 0:
            s = "unnamed"
        if s[0].isdigit():
            s = "id_" + s
        return s

    @staticmethod
    def _is_number(value):
        """Returns True if value is an int or float but not a bool."""
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _rdf_literal(value):
        """Returns an RDF literal string with a basic datatype."""
        if isinstance(value, bool):
            return '"' + str(value).lower() + '"^^xsd:boolean'
        if isinstance(value, int) and not isinstance(value, bool):
            return '"' + str(value) + '"^^xsd:integer'
        if isinstance(value, float):
            return '"' + str(value) + '"^^xsd:double'
        return '"' + Ontology._safe_string(value) + '"'

    @staticmethod
    def _uri_for_topology(topology, prefix="inst"):
        """Returns a stable URI-like QName for a topology, graph, or TGraph entity."""
        guid = Ontology._value(topology, Ontology.IFC_GUID_KEY, None)
        uri = Ontology._value(topology, Ontology.URI_KEY, None)
        label = Ontology._value(topology, Ontology.LABEL_KEY, None)
        local_id = Ontology._value(topology, "id", None)

        if uri:
            if ":" in str(uri):
                return str(uri)
            return prefix + ":" + Ontology._safe_local_name(uri)

        if guid:
            return prefix + ":" + Ontology._safe_local_name(guid)

        if local_id:
            return prefix + ":" + Ontology._safe_local_name(local_id)

        if label:
            return prefix + ":" + Ontology._safe_local_name(label)

        try:
            from topologicpy.Topology import Topology
            uid = Topology.UUID(topology, silent=True)
            return prefix + ":" + Ontology._safe_local_name(uid)
        except Exception:
            return prefix + ":" + Ontology._safe_local_name(id(topology))

    # -------------------------------------------------------------------------
    # Namespace methods
    # -------------------------------------------------------------------------

    @staticmethod
    def Namespaces():
        """
        Returns the default ontology namespace dictionary.

        Returns
        -------
        dict
            The namespace dictionary.
        """

        return dict(Ontology.NAMESPACES)

    @staticmethod
    def Namespace(prefix, defaultValue=None):
        """
        Returns the namespace URI associated with the input prefix.

        Parameters
        ----------
        prefix : str
            The namespace prefix.
        defaultValue : any , optional
            The value to return if the prefix is not found. Default is None.

        Returns
        -------
        str
            The namespace URI.
        """

        if prefix is None:
            return defaultValue
        return Ontology.NAMESPACES.get(prefix, defaultValue)

    @staticmethod
    def ExpandQName(qname, defaultValue=None):
        """
        Expands a QName into a full URI.

        Parameters
        ----------
        qname : str
            The QName, for example "top:Room".
        defaultValue : any , optional
            The value to return if the QName cannot be expanded. Default is None.

        Returns
        -------
        str
            The expanded URI.
        """

        if not isinstance(qname, str):
            return defaultValue
        if ":" not in qname:
            return defaultValue
        prefix, local = qname.split(":", 1)
        ns = Ontology.Namespace(prefix)
        if ns is None:
            return defaultValue
        return ns + local


    @staticmethod
    def IsQName(value):
        """
        Returns True if the input value is a QName using a known namespace prefix.
        """
        if not isinstance(value, str) or ":" not in value:
            return False
        prefix, local = value.split(":", 1)
        return bool(prefix) and bool(local) and prefix in Ontology.NAMESPACES

    @staticmethod
    def CanonicalClass(ontologyClass, defaultValue=None):
        """
        Returns the canonical TopologicPy ontology class for the input class name.

        Notes
        -----
        ``top:Graph`` and ``top:TGraph`` are treated as aliases. The canonical
        exported class is ``top:Graph`` so that legacy Graph and TGraph instances
        serialise to the same ontology term unless a caller explicitly stores a
        more specific graph subclass such as ``top:NavigationGraph``.
        """
        if ontologyClass is None:
            return defaultValue
        cls = str(ontologyClass).strip()
        if cls == "":
            return defaultValue
        return Ontology.CLASS_ALIASES.get(cls, cls)

    @staticmethod
    def IsResourceString(value):
        """Returns True if the input string appears to be a QName or absolute URI."""
        if not isinstance(value, str):
            return False
        s = value.strip()
        if s.startswith("http://") or s.startswith("https://"):
            return True
        return Ontology.IsQName(s)

    # -------------------------------------------------------------------------
    # Class and category methods
    # -------------------------------------------------------------------------

    @staticmethod
    def Class(topology, defaultValue=None):
        """
        Returns the ontology class assigned to the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        defaultValue : any , optional
            The value to return if no ontology class is found. Default is None.

        Returns
        -------
        str
            The ontology class.
        """

        return Ontology._value(topology, Ontology.ONTOLOGY_CLASS_KEY, defaultValue)

    @staticmethod
    def SetClass(topology, ontologyClass, setCategory=True, setURI=True, silent=False):
        """
        Sets the ontology class of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        ontologyClass : str
            The ontology class, for example "top:Room".
        setCategory : bool , optional
            If True, the category is also set if it can be inferred. Default is True.
        setURI : bool , optional
            If True, ontology_uri is also set if the class can be expanded. Default is True.
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        """

        if topology is None:
            if not silent:
                print("Ontology.SetClass - Error: The input topology is None. Returning None.")
            return None
        if not isinstance(ontologyClass, str) or ontologyClass.strip() == "":
            if not silent:
                print("Ontology.SetClass - Error: The input ontologyClass is not a valid string. Returning None.")
            return None

        ontologyClass = Ontology.CanonicalClass(ontologyClass.strip(), defaultValue=ontologyClass.strip())
        topology = Ontology._set_value(topology, Ontology.ONTOLOGY_CLASS_KEY, ontologyClass, silent=silent)

        if topology is not None and setCategory:
            category = Ontology.CategoryByClass(ontologyClass)
            if category is not None:
                topology = Ontology.SetCategory(topology, category, silent=silent)

        if topology is not None and setURI:
            uri = Ontology.ExpandQName(ontologyClass)
            if uri is not None:
                topology = Ontology._set_value(topology, Ontology.ONTOLOGY_URI_KEY, uri, silent=silent)

        return topology

    @staticmethod
    def Category(topology, defaultValue=None):
        """
        Returns the category assigned to the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        defaultValue : any , optional
            The value to return if no category is found. Default is None.

        Returns
        -------
        str
            The category.
        """

        return Ontology._value(topology, Ontology.CATEGORY_KEY, defaultValue)

    @staticmethod
    def SetCategory(topology, category, silent=False):
        """
        Sets the category of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        category : str
            The category.
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        """

        if not isinstance(category, str) or category.strip() == "":
            if not silent:
                print("Ontology.SetCategory - Error: The input category is not a valid string. Returning None.")
            return None
        return Ontology._set_value(topology, Ontology.CATEGORY_KEY, category.strip(), silent=silent)

    @staticmethod
    def Label(topology, defaultValue=None):
        """
        Returns the label assigned to the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        defaultValue : any , optional
            The value to return if no label is found. Default is None.

        Returns
        -------
        str
            The label.
        """

        return Ontology._value(topology, Ontology.LABEL_KEY, defaultValue)

    @staticmethod
    def SetLabel(topology, label, silent=False):
        """
        Sets the label of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        label : str
            The label.
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        """

        return Ontology._set_value(topology, Ontology.LABEL_KEY, label, silent=silent)

    @staticmethod
    def URI(topology, defaultValue=None):
        """
        Returns the URI assigned to the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        defaultValue : any , optional
            The value to return if no URI is found. Default is None.

        Returns
        -------
        str
            The URI.
        """

        return Ontology._value(topology, Ontology.URI_KEY, defaultValue)

    @staticmethod
    def SetURI(topology, uri, silent=False):
        """
        Sets the URI of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        uri : str
            The URI.
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        """

        return Ontology._set_value(topology, Ontology.URI_KEY, uri, silent=silent)

    @staticmethod
    def CategoryByClass(ontologyClass, defaultValue=None):
        """
        Returns a category for the input ontology class.

        Parameters
        ----------
        ontologyClass : str
            The ontology class.
        defaultValue : any , optional
            The value to return if no category is found. Default is None.

        Returns
        -------
        str
            The category.
        """

        if ontologyClass is None:
            return defaultValue
        ontologyClass = Ontology.CanonicalClass(str(ontologyClass).strip(), defaultValue=str(ontologyClass).strip())
        if ontologyClass in Ontology.TOP_CATEGORIES:
            return Ontology.TOP_CATEGORIES[ontologyClass]
        return defaultValue

    @staticmethod
    def Superclasses(ontologyClass, transitive=True):
        """
        Returns the superclasses of the input ontology class.

        Parameters
        ----------
        ontologyClass : str
            The ontology class.
        transitive : bool , optional
            If True, all transitive superclasses are returned. If False, only direct
            superclasses are returned. Default is True.

        Returns
        -------
        list
            The list of superclasses.
        """

        if ontologyClass is None:
            return []

        ontologyClass = Ontology.CanonicalClass(str(ontologyClass).strip(), defaultValue=str(ontologyClass).strip())
        direct = Ontology.TOP_SUPERCLASSES.get(ontologyClass, [])
        if not transitive:
            return list(direct)

        result = []
        stack = list(direct)
        while len(stack) > 0:
            cls = stack.pop(0)
            if cls not in result:
                result.append(cls)
                stack.extend(Ontology.TOP_SUPERCLASSES.get(cls, []))
        return result

    @staticmethod
    def IsA(topology, ontologyClass, transitive=True):
        """
        Returns True if the input topology is an instance of the input ontology class.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        ontologyClass : str
            The ontology class to test against.
        transitive : bool , optional
            If True, superclass inheritance is considered. Default is True.

        Returns
        -------
        bool
            True if the topology is an instance of ontologyClass. Otherwise False.
        """

        assignedClass = Ontology.Class(topology)
        if assignedClass is None or ontologyClass is None:
            return False

        assignedClass = Ontology.CanonicalClass(str(assignedClass).strip(), defaultValue=str(assignedClass).strip())
        ontologyClass = Ontology.CanonicalClass(str(ontologyClass).strip(), defaultValue=str(ontologyClass).strip())

        if assignedClass == ontologyClass:
            return True

        if transitive:
            return ontologyClass in Ontology.Superclasses(assignedClass, transitive=True)

        return False

    # -------------------------------------------------------------------------
    # IFC mappings
    # -------------------------------------------------------------------------

    @staticmethod
    def ClassByIFCClass(ifcClass, defaultValue="top:Element"):
        """
        Returns a TopologicPy ontology class from an IFC class.

        Parameters
        ----------
        ifcClass : str
            The IFC class, for example "IfcSpace".
        defaultValue : any , optional
            The value to return if the IFC class is not mapped. Default is "top:Element".

        Returns
        -------
        str
            The TopologicPy ontology class.
        """

        if ifcClass is None:
            return defaultValue
        ifcClass = str(ifcClass).strip()
        return Ontology.IFC_TO_TOP.get(ifcClass, defaultValue)

    @staticmethod
    def BOTClassByClass(ontologyClass, defaultValue=None):
        """
        Returns a BOT class from a TopologicPy ontology class.

        Parameters
        ----------
        ontologyClass : str
            The TopologicPy ontology class.
        defaultValue : any , optional
            The value to return if the class is not mapped. Default is None.

        Returns
        -------
        str
            The BOT class.
        """

        if ontologyClass is None:
            return defaultValue

        ontologyClass = str(ontologyClass).strip()
        if ontologyClass in Ontology.TOP_TO_BOT:
            return Ontology.TOP_TO_BOT[ontologyClass]

        for superclass in Ontology.Superclasses(ontologyClass, transitive=True):
            if superclass in Ontology.TOP_TO_BOT:
                return Ontology.TOP_TO_BOT[superclass]

        return defaultValue

    @staticmethod
    def AnnotateIFC(topology,
                    ifcClass=None,
                    ifcGUID=None,
                    ifcName=None,
                    source=None,
                    silent=False):
        """
        Annotates a topology using IFC metadata.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        ifcClass : str , optional
            The IFC class. Default is None.
        ifcGUID : str , optional
            The IFC GlobalId. Default is None.
        ifcName : str , optional
            The IFC name. If specified, it is stored as the label. Default is None.
        source : str , optional
            The source file or source identifier. Default is None.
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The annotated topology or graph.
        """

        if topology is None:
            if not silent:
                print("Ontology.AnnotateIFC - Error: The input topology is None. Returning None.")
            return None

        if ifcClass is not None:
            topology = Ontology._set_value(topology, Ontology.IFC_CLASS_KEY, ifcClass, silent=silent)
            ontologyClass = Ontology.ClassByIFCClass(ifcClass, defaultValue=None)
            if ontologyClass is not None:
                topology = Ontology.SetClass(topology, ontologyClass, silent=silent)

        if ifcGUID is not None:
            topology = Ontology._set_value(topology, Ontology.IFC_GUID_KEY, ifcGUID, silent=silent)

        if ifcName is not None:
            topology = Ontology.SetLabel(topology, ifcName, silent=silent)

        if source is not None:
            topology = Ontology._set_value(topology, Ontology.SOURCE_KEY, source, silent=silent)

        return topology

    # -------------------------------------------------------------------------
    # Topologic class inference
    # -------------------------------------------------------------------------

    @staticmethod
    def ClassByTopology(topology, defaultValue="top:Topology"):
        """
        Returns a TopologicPy ontology class inferred from the topology type.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        defaultValue : any , optional
            The value to return if the topology type cannot be inferred. Default is "top:Topology".

        Returns
        -------
        str
            The inferred ontology class.
        """

        if topology is None:
            return defaultValue

        if Ontology._is_tgraph(topology):
            return "top:Graph"

        try:
            from topologicpy.Topology import Topology

            type_name = None
            try:
                type_name = Topology.TypeAsString(topology)
            except Exception:
                pass

            if isinstance(type_name, str):
                type_name_lower = type_name.lower()
                mapping = {
                    "vertex": "top:Vertex",
                    "edge": "top:Edge",
                    "wire": "top:Wire",
                    "face": "top:Face",
                    "shell": "top:Shell",
                    "cell": "top:Cell",
                    "cellcomplex": "top:CellComplex",
                    "cluster": "top:Cluster",
                    "graph": "top:Graph",
                    "tgraph": "top:Graph",
                }
                if type_name_lower in mapping:
                    return mapping[type_name_lower]

            if Topology.IsInstance(topology, "vertex"):
                return "top:Vertex"
            if Topology.IsInstance(topology, "edge"):
                return "top:Edge"
            if Topology.IsInstance(topology, "wire"):
                return "top:Wire"
            if Topology.IsInstance(topology, "face"):
                return "top:Face"
            if Topology.IsInstance(topology, "shell"):
                return "top:Shell"
            if Topology.IsInstance(topology, "cell"):
                return "top:Cell"
            if Topology.IsInstance(topology, "cellcomplex"):
                return "top:CellComplex"
            if Topology.IsInstance(topology, "cluster"):
                return "top:Cluster"
            if Topology.IsInstance(topology, "graph"):
                return "top:Graph"
        except Exception:
            pass

        return defaultValue

    @staticmethod
    def Annotate(topology,
                 ontologyClass=None,
                 category=None,
                 label=None,
                 uri=None,
                 source=None,
                 derivedFrom=None,
                 generatedBy=None,
                 inferClass=False,
                 silent=False):
        """
        Annotates a topology using canonical ontology dictionary keys.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        ontologyClass : str , optional
            The ontology class. Default is None.
        category : str , optional
            The category. Default is None.
        label : str , optional
            The label. Default is None.
        uri : str , optional
            The URI. Default is None.
        source : str , optional
            The source identifier. Default is None.
        derivedFrom : str , optional
            The source entity from which this topology was derived. Default is None.
        generatedBy : str , optional
            The process or method that generated this topology. Default is None.
        inferClass : bool , optional
            If True and ontologyClass is None, the ontology class is inferred from
            the Topologic type. Default is False.
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The annotated topology or graph.
        """

        if topology is None:
            if not silent:
                print("Ontology.Annotate - Error: The input topology is None. Returning None.")
            return None

        if ontologyClass is None and inferClass:
            ontologyClass = Ontology.ClassByTopology(topology, defaultValue=None)

        if ontologyClass is not None:
            topology = Ontology.SetClass(topology, ontologyClass, silent=silent)

        if category is not None:
            topology = Ontology.SetCategory(topology, category, silent=silent)

        if label is not None:
            topology = Ontology.SetLabel(topology, label, silent=silent)

        if uri is not None:
            topology = Ontology.SetURI(topology, uri, silent=silent)

        if source is not None:
            topology = Ontology._set_value(topology, Ontology.SOURCE_KEY, source, silent=silent)

        if derivedFrom is not None:
            topology = Ontology._set_value(topology, Ontology.DERIVED_FROM_KEY, derivedFrom, silent=silent)

        if generatedBy is not None:
            topology = Ontology._set_value(topology, Ontology.GENERATED_BY_KEY, generatedBy, silent=silent)

        return topology

    @staticmethod
    def AnnotateSubtopologies(topology,
                              vertices=True,
                              edges=True,
                              wires=True,
                              faces=True,
                              shells=True,
                              cells=True,
                              cellComplexes=True,
                              inferClass=True,
                              silent=False):
        """
        Annotates the requested subtopologies with inferred ontology classes.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        vertices : bool , optional
            If True, annotate vertices. Default is True.
        edges : bool , optional
            If True, annotate edges. Default is True.
        wires : bool , optional
            If True, annotate wires. Default is True.
        faces : bool , optional
            If True, annotate faces. Default is True.
        shells : bool , optional
            If True, annotate shells. Default is True.
        cells : bool , optional
            If True, annotate cells. Default is True.
        cellComplexes : bool , optional
            If True, annotate cell complexes. Default is True.
        inferClass : bool , optional
            If True, infer ontology classes from Topologic types. Default is True.
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Topology
            The input topology.
        """

        if topology is None:
            if not silent:
                print("Ontology.AnnotateSubtopologies - Error: The input topology is None. Returning None.")
            return None

        try:
            from topologicpy.Topology import Topology

            selectors = []
            if vertices:
                selectors.append("vertex")
            if edges:
                selectors.append("edge")
            if wires:
                selectors.append("wire")
            if faces:
                selectors.append("face")
            if shells:
                selectors.append("shell")
            if cells:
                selectors.append("cell")
            if cellComplexes:
                selectors.append("cellcomplex")

            for selector in selectors:
                try:
                    subtopologies = Topology.SubTopologies(topology, subTopologyType=selector)
                except Exception:
                    subtopologies = []
                for subtopology in subtopologies:
                    Ontology.Annotate(subtopology, inferClass=inferClass, silent=True)
            return topology
        except Exception as e:
            if not silent:
                print("Ontology.AnnotateSubtopologies - Error: Could not annotate subtopologies. Returning input topology.")
                print("Error:", e)
            return topology

    # -------------------------------------------------------------------------
    # Dictionary normalisation
    # -------------------------------------------------------------------------

    @staticmethod
    def NormalizeDictionary(topology,
                            labelKeys=None,
                            categoryKeys=None,
                            ifcClassKeys=None,
                            ifcGUIDKeys=None,
                            silent=False):
        """
        Normalises common dictionary keys into canonical ontology keys.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        labelKeys : list , optional
            Candidate keys for the label. Default is ["name", "Name", "LongName", "ifc_name", "label"].
        categoryKeys : list , optional
            Candidate keys for the category. Default is ["category", "type", "ObjectType"].
        ifcClassKeys : list , optional
            Candidate keys for the IFC class. Default is ["ifc_class", "IfcClass", "class", "type"].
        ifcGUIDKeys : list , optional
            Candidate keys for the IFC GUID. Default is ["ifc_guid", "GlobalId", "global_id", "guid"].
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        """

        try:
            from topologicpy.Dictionary import Dictionary
        except Exception:
            Dictionary = None

        if topology is None:
            if not silent:
                print("Ontology.NormalizeDictionary - Error: The input topology is None. Returning None.")
            return None

        labelKeys = labelKeys or ["name", "Name", "LongName", "ifc_name", "label"]
        categoryKeys = categoryKeys or ["category", "type", "ObjectType"]
        ifcClassKeys = ifcClassKeys or ["ifc_class", "IfcClass", "class", "type"]
        ifcGUIDKeys = ifcGUIDKeys or ["ifc_guid", "GlobalId", "global_id", "guid"]

        d = Ontology._dictionary(topology)
        if d is None:
            return topology

        def first_value(keys):
            for key in keys:
                value = Dictionary.ValueAtKey(d, key, None)
                if value is not None and value != "":
                    return value
            return None

        label = first_value(labelKeys)
        category = first_value(categoryKeys)
        ifcClass = first_value(ifcClassKeys)
        ifcGUID = first_value(ifcGUIDKeys)

        if label is not None:
            topology = Ontology.SetLabel(topology, label, silent=silent)
        if category is not None:
            topology = Ontology.SetCategory(topology, str(category).lower(), silent=silent)
        if ifcClass is not None:
            topology = Ontology.AnnotateIFC(topology, ifcClass=ifcClass, silent=silent)
        if ifcGUID is not None:
            topology = Ontology._set_value(topology, Ontology.IFC_GUID_KEY, ifcGUID, silent=silent)

        return topology


    @staticmethod
    def _triple_object(predicate, value):
        """
        Returns a Turtle-compatible object term for a predicate/value pair.
        Object-property values that look like QNames or absolute URIs are emitted
        as resources; everything else is emitted as a typed literal.
        """
        if predicate in Ontology.OBJECT_PROPERTIES and Ontology.IsResourceString(value):
            return str(value).strip()
        return Ontology._rdf_literal(value)

    # -------------------------------------------------------------------------
    # Triple and Turtle export
    # -------------------------------------------------------------------------

    @staticmethod
    def Triples(topology,
                subject=None,
                includeDictionaries=True,
                includeBOT=True,
                includeType=True,
                includeLabel=True,
                includeCategory=True,
                namespacePrefix="inst",
                silent=False):
        """
        Returns RDF-like triples from a topology or graph dictionary.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        subject : str , optional
            The subject QName. If None, one is inferred. Default is None.
        includeDictionaries : bool , optional
            If True, dictionary entries are exported as top properties. Default is True.
        includeBOT : bool , optional
            If True, a BOT rdf:type triple is added when a mapping exists. Default is True.
        includeType : bool , optional
            If True, rdf:type triples are exported. Default is True.
        includeLabel : bool , optional
            If True, rdfs:label is exported when a label exists. Default is True.
        includeCategory : bool , optional
            If True, top:category is exported when a category exists. Default is True.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            A list of triples. Each triple is a tuple of the form (subject, predicate, object).
        """

        try:
            from topologicpy.Dictionary import Dictionary
        except Exception:
            Dictionary = None

        if topology is None:
            if not silent:
                print("Ontology.Triples - Error: The input topology is None. Returning an empty list.")
            return []

        triples = []
        if subject is None:
            subject = Ontology._uri_for_topology(topology, prefix=namespacePrefix)

        ontologyClass = Ontology.CanonicalClass(Ontology.Class(topology), defaultValue=Ontology.Class(topology))
        label = Ontology.Label(topology)
        category = Ontology.Category(topology)

        if includeType and ontologyClass is not None:
            triples.append((subject, "rdf:type", ontologyClass))
            if includeBOT:
                botClass = Ontology.BOTClassByClass(ontologyClass)
                if botClass is not None:
                    triples.append((subject, "rdf:type", botClass))

        if includeLabel and label is not None:
            triples.append((subject, "rdfs:label", Ontology._rdf_literal(label)))

        if includeCategory and category is not None:
            triples.append((subject, "top:category", Ontology._rdf_literal(category)))

        if includeDictionaries:
            d = Ontology._dictionary(topology)
            if d is not None:
                if isinstance(d, dict):
                    keys = list(d.keys())
                    def _dict_value(_d, _key):
                        return _d.get(_key, None)
                else:
                    try:
                        keys = Dictionary.Keys(d)
                    except Exception:
                        keys = []
                    def _dict_value(_d, _key):
                        try:
                            return Dictionary.ValueAtKey(_d, _key, None)
                        except Exception:
                            return None
                skip_keys = {
                    Ontology.ONTOLOGY_CLASS_KEY,
                    Ontology.ONTOLOGY_URI_KEY,
                    Ontology.LABEL_KEY,
                    Ontology.CATEGORY_KEY,
                    Ontology.URI_KEY,
                }
                for key in keys:
                    if key in skip_keys:
                        continue
                    value = _dict_value(d, key)
                    if value is None:
                        continue
                    predicate = Ontology.PropertyQName(key)
                    if isinstance(value, (list, tuple)):
                        for v in value:
                            triples.append((subject, predicate, Ontology._triple_object(predicate, v)))
                    else:
                        triples.append((subject, predicate, Ontology._triple_object(predicate, value)))

        return triples

    @staticmethod
    def GraphTriples(graph,
                     includeVertices=True,
                     includeEdges=True,
                     includeDictionaries=True,
                     includeBOT=True,
                     namespacePrefix="inst",
                     silent=False):
        """
        Returns RDF-like triples from a TopologicPy graph.

        Parameters
        ----------
        graph : topologic_core.Graph or topologicpy.TGraph
            The input graph.
        includeVertices : bool , optional
            If True, vertex triples are exported. Default is True.
        includeEdges : bool , optional
            If True, edge triples are exported. Default is True.
        includeDictionaries : bool , optional
            If True, dictionary entries are exported. Default is True.
        includeBOT : bool , optional
            If True, BOT rdf:type triples are added when mappings exist. Default is True.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            A list of triples.
        """

        triples = []

        if graph is None:
            if not silent:
                print("Ontology.GraphTriples - Error: The input graph is None. Returning an empty list.")
            return triples

        try:
            graph_subject = Ontology._uri_for_topology(graph, prefix=namespacePrefix)
            triples.extend(Ontology.Triples(graph,
                                            subject=graph_subject,
                                            includeDictionaries=includeDictionaries,
                                            includeBOT=includeBOT,
                                            namespacePrefix=namespacePrefix,
                                            silent=True))

            vertex_records = []
            if includeVertices:
                vertex_records = Ontology._graph_vertices(graph, asTopologic=False)
                for i, vertex in enumerate(vertex_records):
                    if Ontology.Class(vertex) is None:
                        Ontology.Annotate(vertex, ontologyClass="top:Node", silent=True)
                    v_subject = Ontology._uri_for_topology(vertex, prefix=namespacePrefix)
                    triples.append((graph_subject, "top:hasNode", v_subject))
                    triples.extend(Ontology.Triples(vertex,
                                                    subject=v_subject,
                                                    includeDictionaries=includeDictionaries,
                                                    includeBOT=includeBOT,
                                                    namespacePrefix=namespacePrefix,
                                                    silent=True))
                    if isinstance(vertex, dict):
                        x, y, z = Ontology._record_coordinates(vertex)
                        triples.append((v_subject, "top:hasX", Ontology._rdf_literal(x)))
                        triples.append((v_subject, "top:hasY", Ontology._rdf_literal(y)))
                        triples.append((v_subject, "top:hasZ", Ontology._rdf_literal(z)))

            if includeEdges:
                if not vertex_records:
                    vertex_records = Ontology._graph_vertices(graph, asTopologic=False)
                edges = Ontology._graph_edges(graph, asTopologic=False)
                for i, edge in enumerate(edges):
                    if Ontology.Class(edge) is None:
                        Ontology.Annotate(edge, ontologyClass="top:Relationship", silent=True)
                    e_subject = Ontology._uri_for_topology(edge, prefix=namespacePrefix)
                    # TGraph edge records commonly lack stable labels/URIs, so provide
                    # a deterministic fallback edge subject.
                    if isinstance(edge, dict):
                        d = Ontology._record_dictionary(edge) or {}
                        if not any(d.get(k) for k in [Ontology.URI_KEY, Ontology.IFC_GUID_KEY, Ontology.LABEL_KEY, "id"]):
                            src = edge.get("src", "u")
                            dst = edge.get("dst", "v")
                            e_subject = f"{namespacePrefix}:edge_{Ontology._safe_local_name(str(src))}_{Ontology._safe_local_name(str(dst))}_{i}"
                    triples.append((graph_subject, "top:hasRelationship", e_subject))
                    triples.extend(Ontology.Triples(edge,
                                                    subject=e_subject,
                                                    includeDictionaries=includeDictionaries,
                                                    includeBOT=includeBOT,
                                                    namespacePrefix=namespacePrefix,
                                                    silent=True))

                    sv, ev = Ontology._graph_edge_endpoints(graph, edge, vertex_records=vertex_records)

                    if sv is not None:
                        triples.append((e_subject, "top:startsAt", Ontology._uri_for_topology(sv, prefix=namespacePrefix)))
                    if ev is not None:
                        triples.append((e_subject, "top:endsAt", Ontology._uri_for_topology(ev, prefix=namespacePrefix)))
                    if sv is not None and ev is not None:
                        triples.append((Ontology._uri_for_topology(sv, prefix=namespacePrefix), "top:connectsTo", Ontology._uri_for_topology(ev, prefix=namespacePrefix)))
            return triples
        except Exception as e:
            if not silent:
                print("Ontology.GraphTriples - Error: Could not create graph triples. Returning partial list.")
                print("Error:", e)
            return triples

    @staticmethod
    def TurtleFromTriples(triples,
                          namespaces=None,
                          instanceNamespace="http://w3id.org/topologicpy/instance#",
                          includeHeader=True):
        """
        Returns a conservative Turtle string from the input triples.

        This lightweight serializer accepts triples whose terms are already
        QNames, literals returned by ``_rdf_literal``, absolute URIs, or blank-node
        identifiers. For richer serialisation, prefer ``RDFString``/``ExportRDF``
        when RDFLib is available.
        """

        namespaces = namespaces or Ontology.NAMESPACES
        namespaces = dict(namespaces)
        if "inst" not in namespaces:
            namespaces["inst"] = instanceNamespace

        def is_literal(term):
            return isinstance(term, str) and term.startswith('"')

        def is_qname(term):
            if not isinstance(term, str) or ":" not in term:
                return False
            prefix, local = term.split(":", 1)
            return bool(prefix) and bool(local) and prefix in namespaces

        def format_term(term, predicate=False):
            if term is None:
                return None
            if is_literal(term):
                return term
            if not isinstance(term, str):
                return Ontology._rdf_literal(term)
            s = term.strip()
            if s == "":
                return None
            if is_qname(s):
                return s
            if s.startswith("_:"):
                return s
            if s.startswith("http://") or s.startswith("https://"):
                return "<" + s + ">"
            if predicate:
                return "top:" + Ontology._safe_local_name(s)
            return Ontology._rdf_literal(s)

        lines = []
        if includeHeader:
            for prefix, uri in namespaces.items():
                lines.append("@prefix " + prefix + ": <" + uri + "> .")
            lines.append("")

        seen = set()
        for triple in triples:
            if triple is None or len(triple) != 3:
                continue
            s, p, o = triple
            fs = format_term(s)
            fp = format_term(p, predicate=True)
            fo = format_term(o)
            if fs is None or fp is None or fo is None:
                continue
            line = fs + " " + fp + " " + fo + " ."
            if line not in seen:
                lines.append(line)
                seen.add(line)

        return "\n".join(lines) + "\n"

    @staticmethod
    def PropertyQName(key, defaultPrefix="top"):
        """
        Returns a canonical ontology property QName from a dictionary key.
        """

        if key is None:
            return None
        key = str(key).strip()
        if key == "":
            return None
        if ":" in key:
            prefix, local = key.split(":", 1)
            if prefix == defaultPrefix:
                local = Ontology.PROPERTY_ALIASES.get(local, local)
                return prefix + ":" + Ontology._safe_local_name(local)
            return key
        key = Ontology.PROPERTY_ALIASES.get(key, key)
        return str(defaultPrefix) + ":" + Ontology._safe_local_name(key)

    @staticmethod
    def OntologyTriples(includeBOT=True):
        """
        Returns triples describing the TopologicPy ontology itself.
        """

        def _label(qname):
            local = str(qname).split(":", 1)[1] if ":" in str(qname) else str(qname)
            # Keep class labels compact; property labels are converted from camelCase.
            return local

        def _as_iter(value):
            if isinstance(value, (list, tuple, set)):
                return list(value)
            return [value]

        triples = []
        seen = set()

        def add(s, p, o):
            if s is None or p is None or o is None:
                return
            t = (s, p, o)
            if t not in seen:
                triples.append(t)
                seen.add(t)

        add("top:TopologicPyOntology", "rdf:type", "owl:Ontology")
        add("top:TopologicPyOntology", "rdfs:label", Ontology._rdf_literal("TopologicPy Ontology"))
        add("top:TopologicPyOntology", "rdfs:comment", Ontology._rdf_literal("A compact OWL/RDFS vocabulary for TopologicPy topology, graph, BIM/IFC/BOT/Brick alignment, dictionary metadata, graph metadata, provenance, and analysis results."))

        all_classes = sorted(set(Ontology.TOP_SUPERCLASSES.keys()) | set(Ontology.CLASS_COMMENTS.keys()))
        for cls in all_classes:
            add(cls, "rdf:type", "owl:Class")
            add(cls, "rdfs:label", Ontology._rdf_literal(_label(cls)))
            comment = Ontology.CLASS_COMMENTS.get(cls, None)
            if comment:
                add(cls, "rdfs:comment", Ontology._rdf_literal(comment))
            for superclass in Ontology.TOP_SUPERCLASSES.get(cls, []):
                add(cls, "rdfs:subClassOf", superclass)
            if includeBOT:
                botClass = Ontology.TOP_TO_BOT.get(cls, None)
                if botClass is None:
                    botClass = Ontology.BOTClassByClass(cls)
                if botClass is not None:
                    add(cls, "rdfs:subClassOf", botClass)

        # Graph and TGraph are aliases in this ontology helper.
        add("top:TGraph", "owl:equivalentClass", "top:Graph")
        add("top:Graph", "owl:equivalentClass", "top:TGraph")

        for prop, data in sorted(Ontology.OBJECT_PROPERTIES.items()):
            domain, range_, comment = data
            add(prop, "rdf:type", "owl:ObjectProperty")
            add(prop, "rdfs:label", Ontology._rdf_literal(_label(prop)))
            for d in _as_iter(domain):
                add(prop, "rdfs:domain", d)
            for r in _as_iter(range_):
                add(prop, "rdfs:range", r)
            if comment:
                add(prop, "rdfs:comment", Ontology._rdf_literal(comment))

        for prop, data in sorted(Ontology.DATA_PROPERTIES.items()):
            domain, range_, comment = data
            add(prop, "rdf:type", "owl:DatatypeProperty")
            add(prop, "rdfs:label", Ontology._rdf_literal(_label(prop)))
            for d in _as_iter(domain):
                add(prop, "rdfs:domain", d)
            for r in _as_iter(range_):
                add(prop, "rdfs:range", r)
            if comment:
                add(prop, "rdfs:comment", Ontology._rdf_literal(comment))

        # Legacy/canonical equivalence links.
        for a, b in [
            ("top:startsAt", "top:hasStartVertex"),
            ("top:endsAt", "top:hasEndVertex"),
            ("top:x", "top:hasX"),
            ("top:y", "top:hasY"),
            ("top:z", "top:hasZ"),
            ("top:length", "top:hasLength"),
            ("top:area", "top:hasArea"),
            ("top:volume", "top:hasVolume"),
            ("top:mantissa", "top:hasMantissa"),
            ("top:unit", "top:hasUnit"),
        ]:
            if a in Ontology.OBJECT_PROPERTIES or b in Ontology.OBJECT_PROPERTIES:
                add(a, "owl:equivalentProperty", b)
            else:
                add(a, "owl:equivalentProperty", b)

        if includeBOT:
            # Restrict BOT subproperty alignment to semantically compatible building/spatial properties.
            bot_subproperties = {
                "top:connectsTo": "bot:connectsTo",
                "top:adjacentTo": "bot:adjacentZone",
                "top:interfaceOf": "bot:interfaceOf",
                "top:containsElement": "bot:containsElement",
            }
            for prop, bot_prop in bot_subproperties.items():
                add(prop, "rdfs:subPropertyOf", bot_prop)

        return triples

    @staticmethod
    def OntologyTTLString(includeBOT=True,
                          instanceNamespace="http://w3id.org/topologicpy/instance#"):
        """
        Returns the TopologicPy ontology specification as a Turtle string.

        Parameters
        ----------
        includeBOT : bool , optional
            If True, BOT alignment triples are included. Default is True.
        instanceNamespace : str , optional
            The instance namespace URI. Default is "http://w3id.org/topologicpy/instance#".

        Returns
        -------
        str
            The ontology Turtle string.
        """

        triples = Ontology.OntologyTriples(includeBOT=includeBOT)
        return Ontology.TurtleFromTriples(triples,
                                          namespaces=Ontology.NAMESPACES,
                                          instanceNamespace=instanceNamespace,
                                          includeHeader=True)

    @staticmethod
    def ExportOntologyTTL(path,
                          includeBOT=True,
                          instanceNamespace="http://w3id.org/topologicpy/instance#",
                          silent=False):
        """
        Exports the TopologicPy ontology specification as a Turtle file.

        Parameters
        ----------
        path : str
            The output Turtle file path.
        includeBOT : bool , optional
            If True, BOT alignment triples are included. Default is True.
        instanceNamespace : str , optional
            The instance namespace URI. Default is "http://w3id.org/topologicpy/instance#".
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            The input path if successful. Otherwise None.
        """

        if path is None:
            if not silent:
                print("Ontology.ExportOntologyTTL - Error: The input path is None. Returning None.")
            return None
        try:
            ttl = Ontology.OntologyTTLString(includeBOT=includeBOT,
                                             instanceNamespace=instanceNamespace)
            with open(path, "w", encoding="utf-8") as f:
                f.write(ttl)
            return path
        except Exception as e:
            if not silent:
                print("Ontology.ExportOntologyTTL - Error: Could not export ontology Turtle file. Returning None.")
                print("Error:", e)
            return None

    @staticmethod
    def ExportTTL(topology,
                  path,
                  includeGraph=True,
                  includeDictionaries=True,
                  includeBOT=True,
                  namespacePrefix="inst",
                  instanceNamespace="http://w3id.org/topologicpy/instance#",
                  silent=False):
        """
        Exports a topology or graph as a Turtle file.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        path : str
            The output Turtle file path.
        includeGraph : bool , optional
            If True and the input is a graph, graph vertices and edges are exported.
            Default is True.
        includeDictionaries : bool , optional
            If True, dictionary entries are exported as top properties. Default is True.
        includeBOT : bool , optional
            If True, BOT rdf:type triples are added when mappings exist. Default is True.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        instanceNamespace : str , optional
            The instance namespace URI. Default is "http://w3id.org/topologicpy/instance#".
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            The input path if successful. Otherwise None.
        """

        if topology is None:
            if not silent:
                print("Ontology.ExportTTL - Error: The input topology is None. Returning None.")
            return None

        if path is None:
            if not silent:
                print("Ontology.ExportTTL - Error: The input path is None. Returning None.")
            return None

        try:
            if includeGraph and Ontology._is_graph_like(topology):
                triples = Ontology.GraphTriples(topology,
                                                includeDictionaries=includeDictionaries,
                                                includeBOT=includeBOT,
                                                namespacePrefix=namespacePrefix,
                                                silent=silent)
            else:
                triples = Ontology.Triples(topology,
                                           includeDictionaries=includeDictionaries,
                                           includeBOT=includeBOT,
                                           namespacePrefix=namespacePrefix,
                                           silent=silent)

            ttl = Ontology.TurtleFromTriples(triples, instanceNamespace=instanceNamespace)
            with open(path, "w", encoding="utf-8") as f:
                f.write(ttl)
            return path
        except Exception as e:
            if not silent:
                print("Ontology.ExportTTL - Error: Could not export Turtle file. Returning None.")
                print("Error:", e)
            return None

    @staticmethod
    def TTLString(topology,
                  includeGraph=True,
                  includeDictionaries=True,
                  includeBOT=True,
                  namespacePrefix="inst",
                  instanceNamespace="http://w3id.org/topologicpy/instance#",
                  silent=False):
        """
        Returns a Turtle string from a topology or graph.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        includeGraph : bool , optional
            If True and the input is a graph, graph vertices and edges are exported.
            Default is True.
        includeDictionaries : bool , optional
            If True, dictionary entries are exported as top properties. Default is True.
        includeBOT : bool , optional
            If True, BOT rdf:type triples are added when mappings exist. Default is True.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        instanceNamespace : str , optional
            The instance namespace URI. Default is "http://w3id.org/topologicpy/instance#".
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            The Turtle string.
        """

        if topology is None:
            if not silent:
                print("Ontology.TTLString - Error: The input topology is None. Returning None.")
            return None

        try:
            if includeGraph and Ontology._is_graph_like(topology):
                triples = Ontology.GraphTriples(topology,
                                                includeDictionaries=includeDictionaries,
                                                includeBOT=includeBOT,
                                                namespacePrefix=namespacePrefix,
                                                silent=silent)
            else:
                triples = Ontology.Triples(topology,
                                           includeDictionaries=includeDictionaries,
                                           includeBOT=includeBOT,
                                           namespacePrefix=namespacePrefix,
                                           silent=silent)

            return Ontology.TurtleFromTriples(triples, instanceNamespace=instanceNamespace)
        except Exception as e:
            if not silent:
                print("Ontology.TTLString - Error: Could not create Turtle string. Returning None.")
                print("Error:", e)
            return None

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    @staticmethod
    def Validate(topology,
                 requireClass=True,
                 requireCategory=False,
                 requireLabel=False,
                 requireURI=False,
                 checkClassKnown=True,
                 checkCategory=True,
                 checkQName=True,
                 silent=False):
        """
        Validates the ontology annotation of a topology or graph object.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        requireClass : bool , optional
            If True, ``ontology_class`` must be present. Default is True.
        requireCategory : bool , optional
            If True, ``category`` must be present. Default is False.
        requireLabel : bool , optional
            If True, ``label`` must be present. Default is False.
        requireURI : bool , optional
            If True, ``uri`` or ``ontology_uri`` must be present. Default is False.
        checkClassKnown : bool , optional
            If True, ``ontology_class`` must be in the built-in ontology class map or be expandable as a QName. Default is True.
        checkCategory : bool , optional
            If True, validates ``category`` against the inferred category where possible. Default is True.
        checkQName : bool , optional
            If True, validates QName prefixes against ``Ontology.NAMESPACES``. Default is True.
        silent : bool , optional
            If True, warnings are suppressed. Default is False.

        Returns
        -------
        dict
            A validation report with keys ``ok``, ``errors``, ``warnings``, and ``dictionary``.
        """

        report = {
            "ok": False,
            "errors": [],
            "warnings": [],
            "dictionary": {},
        }

        if topology is None:
            report["errors"].append("The input topology is None.")
            return report

        d = Ontology._dictionary(topology)
        if isinstance(d, dict):
            report["dictionary"] = dict(d)
        else:
            try:
                from topologicpy.Dictionary import Dictionary
                if d is not None:
                    report["dictionary"] = dict(Dictionary.PythonDictionary(d) or {})
            except Exception:
                report["dictionary"] = {}

        ontologyClass = report["dictionary"].get(Ontology.ONTOLOGY_CLASS_KEY, None)
        category = report["dictionary"].get(Ontology.CATEGORY_KEY, None)
        label = report["dictionary"].get(Ontology.LABEL_KEY, None)
        uri = report["dictionary"].get(Ontology.URI_KEY, None)
        ontologyURI = report["dictionary"].get(Ontology.ONTOLOGY_URI_KEY, None)

        if requireClass and (ontologyClass is None or str(ontologyClass).strip() == ""):
            report["errors"].append("Missing ontology_class.")

        if requireCategory and (category is None or str(category).strip() == ""):
            report["errors"].append("Missing category.")

        if requireLabel and (label is None or str(label).strip() == ""):
            report["errors"].append("Missing label.")

        if requireURI and (uri is None or str(uri).strip() == "") and (ontologyURI is None or str(ontologyURI).strip() == ""):
            report["errors"].append("Missing uri or ontology_uri.")

        if ontologyClass is not None and str(ontologyClass).strip() != "":
            ontologyClass = str(ontologyClass).strip()

            if checkQName and ":" in ontologyClass:
                prefix = ontologyClass.split(":", 1)[0]
                if prefix not in Ontology.NAMESPACES:
                    report["errors"].append(f"Unknown ontology_class prefix: {prefix}.")

            if checkClassKnown:
                ontologyClassCanonical = Ontology.CanonicalClass(ontologyClass, defaultValue=ontologyClass)
                known = ontologyClassCanonical in Ontology.TOP_SUPERCLASSES
                expandable = Ontology.ExpandQName(ontologyClassCanonical, defaultValue=None) is not None
                if ontologyClassCanonical.startswith("top:") and not known:
                    report["warnings"].append(f"ontology_class uses the top: namespace but is not in the built-in class map: {ontologyClass}.")
                elif not known and not expandable:
                    report["warnings"].append(f"ontology_class is not known and cannot be expanded: {ontologyClass}.")

            if checkCategory and category is not None and str(category).strip() != "":
                expected = Ontology.CategoryByClass(ontologyClass, defaultValue=None)
                if expected is not None and str(category).strip().lower() != str(expected).strip().lower():
                    report["warnings"].append(
                        f"Category '{category}' does not match inferred category '{expected}' for {ontologyClass}."
                    )

            if ontologyURI is not None and str(ontologyURI).strip() != "":
                expanded = Ontology.ExpandQName(ontologyClass, defaultValue=None)
                if expanded is not None and str(ontologyURI).strip() != str(expanded).strip():
                    report["warnings"].append("ontology_uri does not match expanded ontology_class.")

        report["ok"] = len(report["errors"]) == 0

        if not silent and (report["errors"] or report["warnings"]):
            for error in report["errors"]:
                print("Ontology.Validate - Error:", error)
            for warning in report["warnings"]:
                print("Ontology.Validate - Warning:", warning)

        return report

    @staticmethod
    def ValidateGraph(graph,
                      requireClass=True,
                      requireVertexClasses=True,
                      requireEdgeClasses=True,
                      requireLabels=False,
                      checkConnectivity=True,
                      silent=False):
        """
        Validates ontology annotations on a graph, its vertices, and its edges.

        Parameters
        ----------
        graph : topologic_core.Graph or topologicpy.TGraph
            The input graph.
        requireClass : bool , optional
            If True, the graph itself must have ``ontology_class``. Default is True.
        requireVertexClasses : bool , optional
            If True, every vertex must have ``ontology_class``. Default is True.
        requireEdgeClasses : bool , optional
            If True, every edge must have ``ontology_class``. Default is True.
        requireLabels : bool , optional
            If True, graph elements must have labels. Default is False.
        checkConnectivity : bool , optional
            If True, validates that edge endpoints can be resolved. Default is True.
        silent : bool , optional
            If True, warnings are suppressed. Default is False.

        Returns
        -------
        dict
            A validation report with aggregate status and per-element reports.
        """

        report = {
            "ok": False,
            "errors": [],
            "warnings": [],
            "graph": None,
            "vertices": [],
            "edges": [],
        }

        if graph is None:
            report["errors"].append("The input graph is None.")
            return report

        report["graph"] = Ontology.Validate(graph,
                                             requireClass=requireClass,
                                             requireLabel=requireLabels,
                                             silent=True)
        if not report["graph"].get("ok", False):
            for error in report["graph"].get("errors", []):
                report["errors"].append("Graph: " + error)
        for warning in report["graph"].get("warnings", []):
            report["warnings"].append("Graph: " + warning)

        vertices = Ontology._graph_vertices(graph, asTopologic=False)
        edges = Ontology._graph_edges(graph, asTopologic=False)

        for i, vertex in enumerate(vertices):
            r = Ontology.Validate(vertex,
                                  requireClass=requireVertexClasses,
                                  requireLabel=requireLabels,
                                  silent=True)
            r["index"] = i
            report["vertices"].append(r)
            if not r.get("ok", False):
                for error in r.get("errors", []):
                    report["errors"].append(f"Vertex {i}: {error}")
            for warning in r.get("warnings", []):
                report["warnings"].append(f"Vertex {i}: {warning}")

        for i, edge in enumerate(edges):
            r = Ontology.Validate(edge,
                                  requireClass=requireEdgeClasses,
                                  requireLabel=False,
                                  silent=True)
            r["index"] = i

            if checkConnectivity:
                sv, ev = Ontology._graph_edge_endpoints(graph, edge, vertex_records=vertices)
                if sv is None or ev is None:
                    r.setdefault("errors", []).append("Could not resolve edge start/end vertices.")
                    r["ok"] = False

            report["edges"].append(r)
            if not r.get("ok", False):
                for error in r.get("errors", []):
                    report["errors"].append(f"Edge {i}: {error}")
            for warning in r.get("warnings", []):
                report["warnings"].append(f"Edge {i}: {warning}")

        report["ok"] = len(report["errors"]) == 0

        if not silent and (report["errors"] or report["warnings"]):
            for error in report["errors"]:
                print("Ontology.ValidateGraph - Error:", error)
            for warning in report["warnings"]:
                print("Ontology.ValidateGraph - Warning:", warning)

        return report

    @staticmethod
    def ValidateTTLString(ttlString, silent=False):
        """
        Validates a Turtle string using RDFLib when available.

        Parameters
        ----------
        ttlString : str
            The Turtle string.
        silent : bool , optional
            If True, errors are suppressed. Default is False.

        Returns
        -------
        dict
            A validation report with keys ``ok``, ``errors``, ``warnings``, and ``triple_count``.
        """

        report = {"ok": False, "errors": [], "warnings": [], "triple_count": 0}
        if not isinstance(ttlString, str) or ttlString.strip() == "":
            report["errors"].append("The input ttlString is not a valid string.")
            return report
        try:
            import rdflib
        except Exception as e:
            report["warnings"].append("RDFLib is not installed, so syntax validation could not be performed.")
            report["ok"] = True
            return report
        try:
            g = rdflib.Graph()
            g.parse(data=ttlString, format="turtle")
            report["triple_count"] = len(g)
            report["ok"] = True
            return report
        except Exception as e:
            report["errors"].append(str(e))
            if not silent:
                print("Ontology.ValidateTTLString - Error:", e)
            return report

    @staticmethod
    def ValidateTTLFile(path, silent=False):
        """
        Validates a Turtle file using RDFLib when available.

        Parameters
        ----------
        path : str
            The Turtle file path.
        silent : bool , optional
            If True, errors are suppressed. Default is False.

        Returns
        -------
        dict
            A validation report.
        """

        if path is None:
            return {"ok": False, "errors": ["The input path is None."], "warnings": [], "triple_count": 0}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return Ontology.ValidateTTLString(f.read(), silent=silent)
        except Exception as e:
            if not silent:
                print("Ontology.ValidateTTLFile - Error:", e)
            return {"ok": False, "errors": [str(e)], "warnings": [], "triple_count": 0}

    # -------------------------------------------------------------------------
    # RDFLib export/import
    # -------------------------------------------------------------------------

    @staticmethod
    def RDFGraph(topology,
                 includeGraph=True,
                 includeDictionaries=True,
                 includeBOT=True,
                 namespacePrefix="inst",
                 instanceNamespace="http://w3id.org/topologicpy/instance#",
                 silent=False):
        """
        Returns an RDFLib graph from a topology or graph.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        includeGraph : bool , optional
            If True and the input is a graph, vertices and edges are exported. Default is True.
        includeDictionaries : bool , optional
            If True, dictionary keys are exported as data properties. Default is True.
        includeBOT : bool , optional
            If True, BOT type triples are included where mapped. Default is True.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        instanceNamespace : str , optional
            The instance namespace URI. Default is "http://w3id.org/topologicpy/instance#".
        silent : bool , optional
            If True, warnings are suppressed. Default is False.

        Returns
        -------
        rdflib.Graph
            The RDFLib graph, or None if RDFLib is unavailable.
        """

        try:
            import rdflib
            from rdflib import Graph as RDFLibGraph
            from rdflib import Namespace, URIRef, Literal
            from rdflib.namespace import RDF, RDFS, XSD, OWL
        except Exception as e:
            if not silent:
                print("Ontology.RDFGraph - Error: RDFLib is required. Install it with 'pip install rdflib'. Returning None.")
                print("Error:", e)
            return None

        def expand(term):
            if term is None:
                return None
            if isinstance(term, str) and term.startswith('"'):
                return literal_from_turtle(term)
            if isinstance(term, str) and ":" in term and not term.startswith("http"):
                prefix, local = term.split(":", 1)
                if prefix in namespaces:
                    return URIRef(namespaces[prefix] + local)
            if isinstance(term, str) and (term.startswith("http://") or term.startswith("https://")):
                return URIRef(term)
            return URIRef(str(term))

        def literal_from_turtle(text):
            text = str(text)
            if "^^" not in text:
                return Literal(text[1:-1] if text.startswith('"') and text.endswith('"') else text)
            value, dtype = text.split("^^", 1)
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            dtype_ref = expand(dtype.strip())
            if str(dtype_ref) == str(XSD.integer):
                try:
                    return Literal(int(value), datatype=XSD.integer)
                except Exception:
                    return Literal(value, datatype=XSD.integer)
            if str(dtype_ref) == str(XSD.double):
                try:
                    return Literal(float(value), datatype=XSD.double)
                except Exception:
                    return Literal(value, datatype=XSD.double)
            if str(dtype_ref) == str(XSD.boolean):
                return Literal(str(value).lower() == "true", datatype=XSD.boolean)
            return Literal(value, datatype=dtype_ref)

        namespaces = dict(Ontology.NAMESPACES)
        namespaces[namespacePrefix] = instanceNamespace

        rdf_graph = RDFLibGraph()
        for prefix, uri in namespaces.items():
            rdf_graph.bind(prefix, Namespace(uri))

        try:
            if includeGraph and Ontology._is_graph_like(topology):
                triples = Ontology.GraphTriples(topology,
                                                includeDictionaries=includeDictionaries,
                                                includeBOT=includeBOT,
                                                namespacePrefix=namespacePrefix,
                                                silent=silent)
            else:
                triples = Ontology.Triples(topology,
                                           includeDictionaries=includeDictionaries,
                                           includeBOT=includeBOT,
                                           namespacePrefix=namespacePrefix,
                                           silent=silent)
            for s, p, o in triples:
                rdf_graph.add((expand(s), expand(p), expand(o)))
            return rdf_graph
        except Exception as e:
            if not silent:
                print("Ontology.RDFGraph - Error: Could not build RDF graph. Returning None.")
                print("Error:", e)
            return None

    @staticmethod
    def RDFString(topology,
                  format="turtle",
                  includeGraph=True,
                  includeDictionaries=True,
                  includeBOT=True,
                  namespacePrefix="inst",
                  instanceNamespace="http://w3id.org/topologicpy/instance#",
                  silent=False):
        """
        Serializes a topology or graph to an RDF string using RDFLib.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        format : str , optional
            RDFLib serialization format, e.g. "turtle", "xml", "json-ld", "nt". Default is "turtle".
        includeGraph : bool , optional
            If True and the input is a graph, vertices and edges are exported. Default is True.
        includeDictionaries : bool , optional
            If True, dictionary keys are exported. Default is True.
        includeBOT : bool , optional
            If True, BOT type triples are included where mapped. Default is True.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        instanceNamespace : str , optional
            The instance namespace URI. Default is "http://w3id.org/topologicpy/instance#".
        silent : bool , optional
            If True, warnings are suppressed. Default is False.

        Returns
        -------
        str
            The serialized RDF string, or None on failure.
        """

        g = Ontology.RDFGraph(topology,
                              includeGraph=includeGraph,
                              includeDictionaries=includeDictionaries,
                              includeBOT=includeBOT,
                              namespacePrefix=namespacePrefix,
                              instanceNamespace=instanceNamespace,
                              silent=silent)
        if g is None:
            return None
        try:
            result = g.serialize(format=format)
            if isinstance(result, bytes):
                return result.decode("utf-8")
            return str(result)
        except Exception as e:
            if not silent:
                print("Ontology.RDFString - Error: Could not serialize RDF graph. Returning None.")
                print("Error:", e)
            return None

    @staticmethod
    def ExportRDF(topology,
                  path,
                  format=None,
                  includeGraph=True,
                  includeDictionaries=True,
                  includeBOT=True,
                  namespacePrefix="inst",
                  instanceNamespace="http://w3id.org/topologicpy/instance#",
                  silent=False):
        """
        Exports a topology or graph to an RDF file using RDFLib.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        path : str
            The output file path.
        format : str , optional
            RDFLib serialization format. If None, it is inferred from the extension. Default is None.
        includeGraph : bool , optional
            If True and the input is a graph, vertices and edges are exported. Default is True.
        includeDictionaries : bool , optional
            If True, dictionary keys are exported. Default is True.
        includeBOT : bool , optional
            If True, BOT type triples are included where mapped. Default is True.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        instanceNamespace : str , optional
            The instance namespace URI. Default is "http://w3id.org/topologicpy/instance#".
        silent : bool , optional
            If True, warnings are suppressed. Default is False.

        Returns
        -------
        str
            The output path, or None on failure.
        """

        if path is None:
            if not silent:
                print("Ontology.ExportRDF - Error: The input path is None. Returning None.")
            return None
        if format is None:
            ext = str(path).lower().rsplit(".", 1)[-1] if "." in str(path) else "ttl"
            format = {
                "ttl": "turtle",
                "rdf": "xml",
                "xml": "xml",
                "jsonld": "json-ld",
                "json": "json-ld",
                "nt": "nt",
                "n3": "n3",
            }.get(ext, "turtle")
        g = Ontology.RDFGraph(topology,
                              includeGraph=includeGraph,
                              includeDictionaries=includeDictionaries,
                              includeBOT=includeBOT,
                              namespacePrefix=namespacePrefix,
                              instanceNamespace=instanceNamespace,
                              silent=silent)
        if g is None:
            return None
        try:
            g.serialize(destination=path, format=format)
            return path
        except Exception as e:
            if not silent:
                print("Ontology.ExportRDF - Error: Could not export RDF file. Returning None.")
                print("Error:", e)
            return None

    @staticmethod
    def GraphByRDFGraph(rdfGraph,
                        graphSubject=None,
                        namespacePrefix="inst",
                        tolerance=0.0001,
                        asTGraph: bool = False,
                        silent=False):
        """
        Reconstructs a TopologicPy graph from an RDFLib graph exported by
        ``Ontology.RDFGraph`` or ``Ontology.ExportRDF``.

        Both canonical and legacy edge-endpoint predicates are accepted:
        ``top:startsAt``/``top:endsAt`` and
        ``top:hasStartVertex``/``top:hasEndVertex``.
        """

        try:
            from rdflib import URIRef, Literal
            from rdflib.namespace import RDF, RDFS
            from topologicpy.Vertex import Vertex
            from topologicpy.Edge import Edge
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary
        except Exception as e:
            if not silent:
                print("Ontology.GraphByRDFGraph - Error: Missing dependency or TopologicPy class. Returning None.")
                print("Error:", e)
            return None

        if rdfGraph is None:
            if not silent:
                print("Ontology.GraphByRDFGraph - Error: The input rdfGraph is None. Returning None.")
            return None

        def uri(qname):
            if qname is None:
                return None
            if isinstance(qname, URIRef):
                return qname
            qname = str(qname)
            if qname.startswith("http://") or qname.startswith("https://"):
                return URIRef(qname)
            if ":" in qname:
                prefix, local = qname.split(":", 1)
                if prefix in Ontology.NAMESPACES:
                    return URIRef(Ontology.NAMESPACES[prefix] + local)
            return URIRef(qname)

        def qname(term):
            try:
                return rdfGraph.namespace_manager.normalizeUri(term).replace("<", "").replace(">", "")
            except Exception:
                return str(term)

        def local_key(term):
            q = qname(term)
            if ":" in q:
                return q.split(":", 1)[1]
            if "#" in q:
                return q.rsplit("#", 1)[1]
            if "/" in q:
                return q.rsplit("/", 1)[1]
            return q

        def literal_value(value):
            if isinstance(value, Literal):
                try:
                    return value.toPython()
                except Exception:
                    return str(value)
            return qname(value) if isinstance(value, URIRef) else str(value)

        def first_object(subject, predicates):
            for pred in predicates:
                objs = list(rdfGraph.objects(subject, pred))
                if objs:
                    return objs[0]
            return None

        hasNode = uri("top:hasNode")
        hasRelationship = uri("top:hasRelationship")
        startPredicates = [uri("top:startsAt"), uri("top:hasStartVertex")]
        endPredicates = [uri("top:endsAt"), uri("top:hasEndVertex")]
        xPredicates = [uri("top:hasX"), uri("top:x")]
        yPredicates = [uri("top:hasY"), uri("top:y")]
        zPredicates = [uri("top:hasZ"), uri("top:z")]
        categoryP = uri("top:category")

        graph_subject = uri(graphSubject) if graphSubject is not None else None
        if graph_subject is None:
            for s, _, _ in rdfGraph.triples((None, hasNode, None)):
                graph_subject = s
                break
        if graph_subject is None:
            if not silent:
                print("Ontology.GraphByRDFGraph - Error: Could not find a graph subject with top:hasNode. Returning None.")
            return None

        node_subjects = list(rdfGraph.objects(graph_subject, hasNode))
        rel_subjects = list(rdfGraph.objects(graph_subject, hasRelationship))

        def subject_props(subject):
            props = {Ontology.URI_KEY: qname(subject)}
            for obj in rdfGraph.objects(subject, RDF.type):
                q = qname(obj)
                if isinstance(q, str) and q.startswith("top:"):
                    q = Ontology.CanonicalClass(q, defaultValue=q)
                    props[Ontology.ONTOLOGY_CLASS_KEY] = q
                    expanded = Ontology.ExpandQName(q, defaultValue=None)
                    if expanded is not None:
                        props[Ontology.ONTOLOGY_URI_KEY] = expanded
                    cat = Ontology.CategoryByClass(q, defaultValue=None)
                    if cat is not None and Ontology.CATEGORY_KEY not in props:
                        props[Ontology.CATEGORY_KEY] = cat
                    break
            for obj in rdfGraph.objects(subject, RDFS.label):
                props[Ontology.LABEL_KEY] = literal_value(obj)
                break
            for obj in rdfGraph.objects(subject, categoryP):
                props[Ontology.CATEGORY_KEY] = literal_value(obj)
                break
            skip = {RDF.type, RDFS.label, categoryP, hasNode, hasRelationship}
            skip.update(startPredicates)
            skip.update(endPredicates)
            for _, p, o in rdfGraph.triples((subject, None, None)):
                if p in skip:
                    continue
                key = local_key(p)
                # Recover dictionary keys from canonical ontology properties.
                inv_aliases = {v: k for k, v in Ontology.PROPERTY_ALIASES.items() if k in {"x", "y", "z", "length", "area", "volume", "mantissa", "unit", "ontology_class", "ontology_uri", "ifc_class", "ifc_guid", "derived_from", "generated_by"}}
                key = inv_aliases.get(key, key)
                if key not in props:
                    props[key] = literal_value(o)
            return props

        vertices = []
        subject_to_index = {}
        subject_to_vertex = {}
        for i, subject in enumerate(node_subjects):
            props = subject_props(subject)
            x = props.get("x", None)
            y = props.get("y", None)
            z = props.get("z", None)
            if x is None:
                obj = first_object(subject, xPredicates)
                x = literal_value(obj) if obj is not None else float(i)
            if y is None:
                obj = first_object(subject, yPredicates)
                y = literal_value(obj) if obj is not None else 0.0
            if z is None:
                obj = first_object(subject, zPredicates)
                z = literal_value(obj) if obj is not None else 0.0
            try:
                x = float(x)
            except Exception:
                x = float(i)
            try:
                y = float(y)
            except Exception:
                y = 0.0
            try:
                z = float(z)
            except Exception:
                z = 0.0
            props.setdefault("id", Ontology._safe_local_name(qname(subject)))
            props.setdefault("x", x)
            props.setdefault("y", y)
            props.setdefault("z", z)
            subject_to_index[subject] = i
            if asTGraph:
                vertices.append(dict(props))
            else:
                v = Vertex.ByCoordinates(x, y, z)
                v = Topology.SetDictionary(v, Dictionary.ByPythonDictionary(props))
                subject_to_vertex[subject] = v
                vertices.append(v)

        edge_dictionaries = []
        edges = []
        for subject in rel_subjects:
            start_obj = first_object(subject, startPredicates)
            end_obj = first_object(subject, endPredicates)
            if start_obj is None or end_obj is None:
                continue
            props = subject_props(subject)
            props.setdefault("id", Ontology._safe_local_name(qname(subject)))
            if asTGraph:
                src = subject_to_index.get(start_obj)
                dst = subject_to_index.get(end_obj)
                if src is None or dst is None:
                    continue
                ed = dict(props)
                ed["src"] = src
                ed["dst"] = dst
                edge_dictionaries.append(ed)
            else:
                sv = subject_to_vertex.get(start_obj)
                ev = subject_to_vertex.get(end_obj)
                if sv is None or ev is None:
                    continue
                try:
                    e = Edge.ByStartVertexEndVertex(sv, ev)
                except Exception:
                    try:
                        e = Edge.ByVertices([sv, ev], tolerance=tolerance, silent=True)
                    except Exception:
                        e = None
                if e is None:
                    continue
                e = Topology.SetDictionary(e, Dictionary.ByPythonDictionary(props))
                edges.append(e)

        if not vertices:
            return None

        graph_props = subject_props(graph_subject)
        graph_props.setdefault(Ontology.ONTOLOGY_CLASS_KEY, "top:Graph")
        graph_props[Ontology.ONTOLOGY_CLASS_KEY] = Ontology.CanonicalClass(graph_props.get(Ontology.ONTOLOGY_CLASS_KEY), "top:Graph")
        graph_props.setdefault(Ontology.CATEGORY_KEY, "graph")

        if asTGraph:
            try:
                from topologicpy.TGraph import TGraph
                graph = TGraph.ByVerticesEdges(vertices=vertices,
                                               edges=edge_dictionaries,
                                               dictionary=graph_props,
                                               directed=True,
                                               allowSelfLoops=True,
                                               allowParallelEdges=True,
                                               tolerance=tolerance,
                                               silent=silent,
                                               ontology=False)
                return graph
            except Exception as e:
                if not silent:
                    print("Ontology.GraphByRDFGraph - Error: Could not create TGraph. Returning None.")
                    print("Error:", e)
                return None

        try:
            from topologicpy.Graph import Graph
            graph = Graph.ByVerticesEdges(vertices, edges, tolerance=tolerance, silent=silent)
        except TypeError:
            try:
                from topologicpy.Graph import Graph
                graph = Graph.ByVerticesEdges(vertices, edges)
            except Exception:
                graph = None
        if graph is None:
            return None

        graph = Topology.SetDictionary(graph, Dictionary.ByPythonDictionary(graph_props))
        return graph

    @staticmethod
    def GraphByRDFFile(path,
                       format=None,
                       graphSubject=None,
                       namespacePrefix="inst",
                       tolerance=0.0001,
                       asTGraph: bool = False,
                       silent=False):
        """
        Reconstructs a TopologicPy graph from an RDF/Turtle file.

        Parameters
        ----------
        path : str
            The RDF file path.
        format : str , optional
            RDFLib parser format. If None, inferred from the extension. Default is None.
        graphSubject : str , optional
            The graph subject URI or QName. Default is None.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        tolerance : float , optional
            Graph creation tolerance. Default is 0.0001.
        asTGraph : bool , optional
            If True, returns a topologicpy.TGraph. If False, returns a legacy Graph. Default is False.
        silent : bool , optional
            If True, warnings are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph or topologicpy.TGraph
            The reconstructed graph, or None on failure.
        """

        if path is None:
            if not silent:
                print("Ontology.GraphByRDFFile - Error: The input path is None. Returning None.")
            return None
        try:
            import rdflib
        except Exception as e:
            if not silent:
                print("Ontology.GraphByRDFFile - Error: RDFLib is required. Install it with 'pip install rdflib'. Returning None.")
                print("Error:", e)
            return None
        if format is None:
            ext = str(path).lower().rsplit(".", 1)[-1] if "." in str(path) else "ttl"
            format = {
                "ttl": "turtle",
                "rdf": "xml",
                "xml": "xml",
                "jsonld": "json-ld",
                "json": "json-ld",
                "nt": "nt",
                "n3": "n3",
            }.get(ext, "turtle")
        try:
            g = rdflib.Graph()
            for prefix, uri in Ontology.NAMESPACES.items():
                g.bind(prefix, uri)
            g.parse(path, format=format)
            return Ontology.GraphByRDFGraph(g,
                                            graphSubject=graphSubject,
                                            namespacePrefix=namespacePrefix,
                                            tolerance=tolerance,
                                            asTGraph=asTGraph,
                                            silent=silent)
        except Exception as e:
            if not silent:
                print("Ontology.GraphByRDFFile - Error: Could not parse RDF file. Returning None.")
                print("Error:", e)
            return None

    @staticmethod
    def GraphByTTL(path,
                   graphSubject=None,
                   namespacePrefix="inst",
                   tolerance=0.0001,
                   asTGraph: bool = False,
                   silent=False):
        """
        Reconstructs a TopologicPy graph from a Turtle file.

        Parameters
        ----------
        path : str
            The Turtle file path.
        graphSubject : str , optional
            The graph subject URI or QName. Default is None.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        tolerance : float , optional
            Graph creation tolerance. Default is 0.0001.
        asTGraph : bool , optional
            If True, returns a topologicpy.TGraph. If False, returns a legacy Graph. Default is False.
        silent : bool , optional
            If True, warnings are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph or topologicpy.TGraph
            The reconstructed graph, or None on failure.
        """

        return Ontology.GraphByRDFFile(path,
                                       format="turtle",
                                       graphSubject=graphSubject,
                                       namespacePrefix=namespacePrefix,
                                       tolerance=tolerance,
                                       asTGraph=asTGraph,
                                       silent=silent)

    @staticmethod
    def GraphByTTLString(ttlString,
                         graphSubject=None,
                         namespacePrefix="inst",
                         tolerance=0.0001,
                         asTGraph: bool = False,
                         silent=False):
        """
        Reconstructs a TopologicPy graph from a Turtle string.

        Parameters
        ----------
        ttlString : str
            The Turtle string.
        graphSubject : str , optional
            The graph subject URI or QName. Default is None.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        tolerance : float , optional
            Graph creation tolerance. Default is 0.0001.
        asTGraph : bool , optional
            If True, returns a topologicpy.TGraph. If False, returns a legacy Graph. Default is False.
        silent : bool , optional
            If True, warnings are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph or topologicpy.TGraph
            The reconstructed graph, or None on failure.
        """

        try:
            import rdflib
        except Exception as e:
            if not silent:
                print("Ontology.GraphByTTLString - Error: RDFLib is required. Install it with 'pip install rdflib'. Returning None.")
                print("Error:", e)
            return None
        if not isinstance(ttlString, str) or ttlString.strip() == "":
            if not silent:
                print("Ontology.GraphByTTLString - Error: The input ttlString is not valid. Returning None.")
            return None
        try:
            g = rdflib.Graph()
            for prefix, uri in Ontology.NAMESPACES.items():
                g.bind(prefix, uri)
            g.parse(data=ttlString, format="turtle")
            return Ontology.GraphByRDFGraph(g,
                                            graphSubject=graphSubject,
                                            namespacePrefix=namespacePrefix,
                                            tolerance=tolerance,
                                            asTGraph=asTGraph,
                                            silent=silent)
        except Exception as e:
            if not silent:
                print("Ontology.GraphByTTLString - Error: Could not parse Turtle string. Returning None.")
                print("Error:", e)
            return None

# -----------------------------------------------------------------------------


__NAMESPACES_005 = {'bot': 'https://w3id.org/bot#',
 'brick': 'https://brickschema.org/schema/Brick#',
 'dcterms': 'http://purl.org/dc/terms/',
 'dict': 'http://w3id.org/topologicpy/dictionary#',
 'geo': 'http://www.opengis.net/ont/geosparql#',
 'ifc': 'https://standards.buildingsmart.org/IFC/DEV/IFC4/ADD2_TC1/OWL#',
 'inst': 'http://w3id.org/topologicpy/instance#',
 'owl': 'http://www.w3.org/2002/07/owl#',
 'prov': 'http://www.w3.org/ns/prov#',
 'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
 'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
 'skos': 'http://www.w3.org/2004/02/skos/core#',
 'top': 'http://w3id.org/topologicpy#',
 'vann': 'http://purl.org/vocab/vann/',
 'xsd': 'http://www.w3.org/2001/XMLSchema#'}
__TOP_SUPERCLASSES_005 = {'top:AccessGraph': ['top:SpatialGraph'],
 'top:AdjacencyGraph': ['top:SpatialGraph'],
 'top:AnalysisGraph': ['top:Graph'],
 'top:AnalysisMetric': [],
 'top:Aperture': ['top:Element', 'top:Face'],
 'top:Attribute': [],
 'top:Beam': ['top:Element'],
 'top:Boundary': ['top:Topology'],
 'top:Building': ['top:Zone'],
 'top:Cell': ['top:Topology'],
 'top:CellComplex': ['top:Topology'],
 'top:CirculationGraph': ['top:SpatialGraph'],
 'top:CirculationZone': ['top:Zone'],
 'top:ClassificationReference': [],
 'top:Cluster': ['top:Topology'],
 'top:Column': ['top:Element'],
 'top:ConnectivityGraph': ['top:SpatialGraph'],
 'top:Context': [],
 'top:CurtainWall': ['top:Wall'],
 'top:Dictionary': [],
 'top:DirectedRelationship': ['top:Relationship'],
 'top:Door': ['top:Element'],
 'top:DualGraph': ['top:SpatialGraph'],
 'top:Edge': ['top:Topology'],
 'top:EdgeFeature': ['top:Attribute'],
 'top:Element': ['top:Topology'],
 'top:Equipment': ['top:Element'],
 'top:ExternalBoundary': ['top:Boundary'],
 'top:Face': ['top:Topology'],
 'top:FunctionalZone': ['top:Zone'],
 'top:Furniture': ['top:Element'],
 'top:Graph': [],
 'top:GraphDataset': [],
 'top:GraphFeature': ['top:Attribute'],
 'top:Grid': [],
 'top:HasseDiagramGraph': ['top:Graph'],
 'top:Interface': ['top:Face'],
 'top:InternalBoundary': ['top:Boundary'],
 'top:Isovist': ['top:AnalysisMetric'],
 'top:IsovistGraph': ['top:SpatialGraph'],
 'top:KnowledgeGraph': ['top:Graph'],
 'top:LineGraph': ['top:Graph'],
 'top:Material': [],
 'top:MaterialSet': [],
 'top:Matrix': [],
 'top:Member': ['top:Element'],
 'top:NavigationGraph': ['top:SpatialGraph'],
 'top:Node': ['top:Vertex'],
 'top:NodeFeature': ['top:Attribute'],
 'top:Opening': ['top:Element'],
 'top:Path': ['top:Graph'],
 'top:Point': ['top:Vertex'],
 'top:Port': [],
 'top:PrimalGraph': ['top:SpatialGraph'],
 'top:Project': [],
 'top:PropertySet': [],
 'top:QualityIssue': [],
 'top:Quantity': [],
 'top:QuotientGraph': ['top:Graph'],
 'top:Railing': ['top:Element'],
 'top:Relationship': ['top:Edge'],
 'top:Roof': ['top:Element'],
 'top:Room': ['top:Space'],
 'top:SemanticGraph': ['top:Graph'],
 'top:Sensor': ['top:Element'],
 'top:Shell': ['top:Topology'],
 'top:Site': ['top:Zone'],
 'top:Slab': ['top:Element'],
 'top:Space': ['top:Zone'],
 'top:SpaceSyntaxMetric': ['top:AnalysisMetric'],
 'top:SpatialGraph': ['top:Graph'],
 'top:Stair': ['top:Element'],
 'top:Storey': ['top:Zone'],
 'top:Surface': ['top:Face'],
 'top:System': [],
 'top:TGraph': ['top:Graph'],
 'top:ThermalZone': ['top:Space'],
 'top:Topology': [],
 'top:TreeGraph': ['top:Graph'],
 'top:UndirectedRelationship': ['top:Relationship'],
 'top:ValidationRule': [],
 'top:Vector': [],
 'top:Vertex': ['top:Topology'],
 'top:VisibilityGraph': ['top:SpatialGraph'],
 'top:Wall': ['top:Element'],
 'top:Window': ['top:Element'],
 'top:Wire': ['top:Topology'],
 'top:Zone': ['top:Cell']}
__CLASS_COMMENTS_005 = {'top:AccessGraph': 'A directed or undirected graph used to represent permitted access between spaces, rooms, zones, '
                    'portals, doors, apertures, circulation elements, or other traversable entities. Edges may encode '
                    'passability, access control, traversal direction, door type, cost, distance, or movement '
                    'constraints.',
 'top:AdjacencyGraph': 'An undirected graph used to represent adjacency between topological, geometric, spatial, or '
                       'semantic entities. Edges typically indicate that two entities touch, share a boundary, are '
                       'incident, are proximal within a specified tolerance, or satisfy a declared adjacency '
                       'predicate.',
 'top:AnalysisGraph': 'A directed or undirected graph used to represent entities and relationships prepared for '
                      'computational analysis. Vertices, edges, and graph-level dictionaries may encode analytical '
                      'attributes, weights, labels, features, metrics, simulation values, or machine-learning inputs '
                      'and outputs.',
 'top:AnalysisMetric': 'A computed metric associated with geometry, topology, graphs, buildings, systems, simulation, '
                       'performance analysis, or machine-learning workflows.',
 'top:Aperture': 'A topological or building-domain opening associated with a host topology or element. In TopologicPy '
                 'it is treated as an aperture/opening construct and may be represented geometrically by a face-like '
                 'boundary.',
 'top:Attribute': 'A semantic or analytical key-value attribute associated with a topology, graph, node, relationship, '
                  'dataset, metric, or external entity.',
 'top:Beam': 'A beam, girder, or linear structural member.',
 'top:Boundary': 'A topology used to represent the boundary of another topology, spatial region, element, or '
                 'analytical domain.',
 'top:Building': 'A building-level spatial container used to represent a whole building, facility, or IFC building.',
 'top:Cell': 'A three-dimensional TopologicPy topology bounded by faces or shells and representing a volumetric '
             'region.',
 'top:CellComplex': 'A TopologicPy topology composed of cells, typically with non-manifold adjacency through shared '
                    'faces.',
 'top:CirculationGraph': 'A directed or undirected graph used to represent movement, circulation, or route-choice '
                         'relationships among spaces, portals, corridors, stairs, entrances, exits, or waypoints. '
                         'Edges may encode route length, cost, capacity, direction, or accessibility.',
 'top:CirculationZone': 'A zone primarily used to represent movement, access, circulation, transition, or evacuation '
                        'functions.',
 'top:ClassificationReference': 'A classification reference used to represent external classification systems, codes, '
                                'taxonomies, or IFC classification references.',
 'top:Cluster': 'A heterogeneous TopologicPy topology used to group vertices, edges, wires, faces, shells, cells, cell '
                'complexes, or other clusters without requiring manifold connectivity.',
 'top:Column': 'A column or vertical structural member.',
 'top:ConnectivityGraph': 'A directed or undirected graph used to represent generic connectivity or reachability among '
                          'topological, spatial, building, system, or graph entities. Edges indicate that two entities '
                          'are connected according to an explicit computational or semantic rule.',
 'top:Context': 'A contextual reference used by TopologicPy to associate content with a host topology or subtopology.',
 'top:CurtainWall': 'A curtain-wall element represented as a specialised wall or façade system.',
 'top:Dictionary': 'A key-value metadata container used by TopologicPy to attach semantic, analytical, geometric, '
                   'provenance, and interoperability data to topologies, graphs, vertices, edges, and records.',
 'top:DirectedRelationship': 'A directed graph relationship used to represent an ordered source-to-target relation '
                             'between two nodes.',
 'top:Door': 'A door, access, or operable passage element, commonly associated with openings, access graphs, and '
             'navigation relationships.',
 'top:DualGraph': 'An undirected graph used to represent adjacency between higher-dimensional entities such as faces, '
                  'cells, rooms, spaces, zones, or regions. Each entity is represented as a vertex, and an edge '
                  'represents a shared boundary, interface, aperture, adjacency relation, or other declared connection '
                  'between two entities.',
 'top:Edge': 'A one-dimensional TopologicPy topology connecting a start vertex to an end vertex.',
 'top:EdgeFeature': 'A numerical, categorical, vector, or encoded feature associated with a graph edge or relationship '
                    'record.',
 'top:Element': 'A physical, conceptual, or analytical building/infrastructure element that may be represented by '
                'topology and aligned with BIM entities.',
 'top:Equipment': 'A building, MEP, operational, or service equipment element, optionally aligned with Brick equipment '
                  'classes when system semantics are available.',
 'top:ExternalBoundary': 'A boundary topology used to represent the outer boundary of another topology, region, '
                         'element, or analytical domain.',
 'top:Face': 'A two-dimensional TopologicPy topology bounded by an external wire and, optionally, one or more internal '
             'boundary wires.',
 'top:FunctionalZone': 'A zone grouped by use, function, programme, activity, department, or other non-geometric '
                       'classification.',
 'top:Furniture': 'A furniture or furnishing element.',
 'top:Graph': 'A directed, undirected, mixed, weighted, unweighted, simple, or multigraph representation used to '
              'represent entities and relationships emitted, consumed, analysed, or transformed by TopologicPy. More '
              'specific graph semantics are defined by subclasses.',
 'top:GraphDataset': 'A dataset used to represent graphs, nodes, edges, labels, masks, features, metadata, and '
                     'ontology annotations for graph analysis or graph machine learning.',
 'top:GraphFeature': 'A numerical, categorical, vector, or encoded feature associated with a graph-level record.',
 'top:Grid': 'A spatial subdivision, sampling, or reference structure used to generate or organise vertices, edges, '
             'cells, or analysis samples.',
 'top:HasseDiagramGraph': 'A directed acyclic graph used to represent a partially ordered set, typically by showing '
                          'only cover relations after transitive reduction. In TopologicPy, it may represent '
                          'hierarchical containment, decomposition, incidence, or boundary relationships among '
                          'topological entities.',
 'top:Interface': 'A shared boundary or interface between spaces, zones, elements, or topologies, including IFC '
                  'space-boundary-style relationships.',
 'top:InternalBoundary': 'A boundary topology used to represent a hole, void, inner loop, or internal boundary of '
                         'another topology, region, element, or analytical domain.',
 'top:Isovist': 'A visibility field, polygon, volume, or related visual metric computed from an observation point or '
                'region.',
 'top:IsovistGraph': 'A directed or undirected spatial graph used to represent isovists, visibility fields, or '
                     'relationships between observation points and visible spatial extents. Vertices may represent '
                     'viewpoints, isovist polygons, spatial samples, or visible objects, while edges may encode '
                     'visibility, overlap, containment, or intervisibility relationships.',
 'top:KnowledgeGraph': 'A directed semantic graph used to represent entities, attributes, relationships, and inferred '
                       'knowledge. In TopologicPy, it may encode topological objects, BIM entities, ontology classes, '
                       'RDF triples, rules, reasoning outputs, or links between geometry, topology, semantics, '
                       'analysis, and external knowledge sources.',
 'top:LineGraph': 'An undirected or directed graph used to represent edge-to-edge relationships derived from another '
                  'graph. Each vertex in the line graph represents an edge of the source graph, and each edge in the '
                  'line graph represents adjacency, incidence, continuity, succession, or compatibility between source '
                  'edges.',
 'top:Material': 'A material record used to represent material identity, composition, assignment, or IFC material '
                 'information.',
 'top:MaterialSet': 'A material-set record used to represent material lists, material layer sets, material constituent '
                    'sets, or other grouped material definitions.',
 'top:Matrix': 'A numerical matrix used to represent transformations, coordinate operations, or other matrix-valued '
               'quantities.',
 'top:Member': 'A generic structural, framing, or linear member when a more specific class such as Beam or Column is '
               'not asserted.',
 'top:NavigationGraph': 'A directed or undirected weighted graph used to represent navigable movement through spaces, '
                        'paths, portals, waypoints, circulation systems, or walkable regions. Edges typically encode '
                        'traversability and may store distance, travel time, slope, accessibility, obstruction, cost, '
                        'or agent-specific movement constraints.',
 'top:Node': 'A graph node represented by, or projected from, a TopologicPy vertex or TGraph vertex record. Nodes may '
             'carry dictionary metadata, ontology class information, labels, features, coordinates, and external '
             'identifiers.',
 'top:NodeFeature': 'A numerical, categorical, vector, or encoded feature associated with a graph node or vertex '
                    'record.',
 'top:Opening': 'An opening element, void, recess, penetration, or host-element subtraction associated with doors, '
                'windows, services, or coordination checks.',
 'top:Path': 'An ordered graph structure used to represent a sequence of nodes and relationships, typically returned '
             'by routing, shortest-path, circulation, or navigation methods.',
 'top:Point': 'A point-like spatial entity represented as a specialised vertex.',
 'top:Port': 'A distribution or connection port used to represent connection points on systems, equipment, MEP '
             'components, or IFC distribution ports.',
 'top:PrimalGraph': 'A directed or undirected graph used to represent direct physical, spatial, topological, or '
                    'semantic connectivity among entities in their original form. In spatial and topological '
                    'modelling, vertices and edges typically correspond directly to objects, points, spaces, elements, '
                    'or their explicit connections, before any dual transformation is applied.',
 'top:Project': 'A project-level container used to represent a model, dataset, building project, computational study, '
                'or IFC project.',
 'top:PropertySet': 'A property set used to represent grouped name-value properties, commonly mapped from IFC property '
                    'sets or related property-definition entities.',
 'top:QualityIssue': 'A validation, model-checking, coordination, geometry, topology, semantic, or data-quality issue '
                     'detected during analysis.',
 'top:Quantity': 'A quantity record used to represent measured or computed quantities, commonly mapped from IFC '
                 'element quantities.',
 'top:QuotientGraph': 'A directed or undirected graph used to represent a simplified graph obtained by grouping '
                      'vertices, edges, spaces, zones, components, or other entities according to an equivalence '
                      'relation or partition. Vertices represent groups or classes, and edges represent relationships '
                      'between those groups inherited from the source graph.',
 'top:Railing': 'A railing, guard, handrail, or balustrade element.',
 'top:Relationship': 'A graph relationship represented by, or projected from, a TopologicPy edge or TGraph edge '
                     'record. Relationships may carry source and target references, directionality, weights, '
                     'predicates, labels, features, and dictionary metadata.',
 'top:Roof': 'A roof or roof-like envelope element.',
 'top:Room': 'A room-like occupiable space represented as a specialised building space.',
 'top:SemanticGraph': 'A directed graph used to represent typed entities and typed relationships with explicit '
                      'semantic meaning. In TopologicPy, it may encode classes, instances, properties, labels, '
                      'attributes, ontology terms, BIM semantics, or RDF-style subject-predicate-object statements.',
 'top:Sensor': 'A sensor, observation point, or measurement device associated with a space, element, system, or '
               'analytical process.',
 'top:Shell': 'A two-dimensional TopologicPy topology composed of connected faces, typically forming an open or closed '
              'segmented surface.',
 'top:Site': 'A site-level spatial container used to represent land, context, parcels, infrastructure extents, or an '
             'IFC site.',
 'top:Slab': 'A slab, floor, ceiling, plate, or horizontal building element.',
 'top:Space': 'A bounded or occupiable spatial region used to represent rooms, spaces, compartments, or IFC spaces.',
 'top:SpaceSyntaxMetric': 'A spatial network metric used in space syntax, visibility analysis, access analysis, or '
                          'graph-based spatial analysis.',
 'top:SpatialGraph': 'A directed or undirected graph used to represent spatial entities and spatial relationships. '
                     'Relationships may include metric, topological, directional, containment, proximity, overlap, '
                     'visibility, accessibility, or connectivity predicates, including relations aligned with OGC '
                     'Simple Features, ISO 19107 spatial schema concepts, DE-9IM predicates, or RCC-8 qualitative '
                     'spatial relations.',
 'top:Stair': 'A stair, stair flight, or vertical circulation element.',
 'top:Storey': 'A storey-level spatial container used to represent a building level, floor, or IFC building storey.',
 'top:Surface': 'A surface-like spatial, analytical, or building-domain entity represented as a specialised face.',
 'top:System': 'A building, distribution, MEP, operational, or analytical system that groups elements, equipment, '
               'ports, spaces, or functions.',
 'top:TGraph': 'A TopologicPy-native graph representation used to represent indexed vertices, indexed edges, graph '
               'dictionaries, optional topology representations, and ontology-aware metadata. This class distinguishes '
               'the Python TGraph implementation from the more general top:Graph concept.',
 'top:ThermalZone': 'A thermal analysis zone represented as a specialised space used in environmental, energy, HVAC, '
                    'or comfort analysis.',
 'top:Topology': 'Superclass for TopologicPy topological entities, including vertices, edges, wires, faces, shells, '
                 'cells, cell complexes, clusters, apertures, and related topological specialisations.',
 'top:TreeGraph': 'A connected acyclic graph used to represent hierarchical, branching, containment, decomposition, '
                  'classification, or dependency structures. In TopologicPy and BIM contexts, it may represent project '
                  'breakdowns, spatial decomposition, assembly hierarchies, taxonomy structures, or rooted '
                  'parent-child relationships.',
 'top:UndirectedRelationship': 'An undirected graph relationship used to represent a symmetric or '
                               'direction-independent relation between two nodes.',
 'top:ValidationRule': 'A rule used to validate geometry, topology, semantics, IFC/BIM content, graph structure, '
                       'provenance, or analytical metadata.',
 'top:Vector': 'A mathematical vector used to represent direction, displacement, orientation, or other vector-valued '
               'quantities.',
 'top:Vertex': 'A zero-dimensional TopologicPy topology representing a point-like entity, commonly defined by X, Y, '
               'and Z coordinates.',
 'top:VisibilityGraph': 'An undirected or directed spatial graph used to represent line-of-sight visibility or '
                        'intervisibility between points, locations, spaces, surfaces, objects, or sampled positions. '
                        'Edges indicate that two entities are mutually visible or directionally visible, and may store '
                        'visibility distance, angular range, obstruction status, or visual weight.',
 'top:Wall': 'A wall or wall-like building element used to bound, separate, support, enclose, or subdivide spaces.',
 'top:Window': 'A window or transparent/translucent opening element, commonly associated with daylight, view, '
               'envelope, and façade relationships.',
 'top:Wire': 'A one-dimensional TopologicPy topology composed of connected edges, typically forming an open path or a '
             'closed boundary.',
 'top:Zone': 'A spatial region, bounded domain, or analytical zone used to organise spaces, buildings, storeys, sites, '
             'thermal zones, or functional regions.'}
__OBJECT_PROPERTIES_005 = {'top:adjacentTo': ('top:Topology',
                    'top:Topology',
                    'Associates two topologies, spaces, regions, or elements that are adjacent according to a declared '
                    'spatial, topological, or tolerance-based rule.'),
 'top:aggregates': ('top:Topology',
                    'top:Topology',
                    'Represents a whole-part, decomposition, or aggregation relationship, commonly mapped from IFC '
                    'aggregation or decomposition relations.'),
 'top:connects': ('top:Topology',
                  'top:Topology',
                  'Generic semantic connection used when a relationship is known but more specific semantics are '
                  'unavailable.'),
 'top:connectsPort': ('top:Port',
                      'top:Port',
                      'Connects two ports without asserting flow direction. This is the preferred direct mapping for '
                      'port-to-port connectivity such as IfcRelConnectsPorts.'),
 'top:connectsTo': ('top:Topology',
                    'top:Topology',
                    'Generic undirected topological or graph connectivity between two topologies, vertices, nodes, '
                    'elements, spaces, or other entities.'),
 'top:containsElement': ('top:Topology',
                         'top:Topology',
                         'Associates a spatial, topological, or semantic container with a contained topology, element, '
                         'space, or entity.'),
 'top:derivedFrom': ('owl:Thing',
                     'owl:Thing',
                     'Associates an entity, topology, graph, or record with the source entity, model, file, process, '
                     'or data object from which it was derived.'),
 'top:endsAt': ('owl:Thing',
                'top:Vertex',
                'Alias property for associating an edge or relationship with its end vertex or target node.'),
 'top:fillsOpening': ('top:Element',
                      'top:Opening',
                      'Associates an element such as a door, window, or service component with the opening it fills.'),
 'top:generatedBy': ('owl:Thing',
                     'owl:Thing',
                     'Associates an entity, topology, graph, or record with the method, script, notebook, process, or '
                     'software operation that generated it.'),
 'top:hasApproval': ('owl:Thing',
                     'owl:Thing',
                     'Associates an entity with an approval, review, authorisation, or sign-off record.'),
 'top:hasCell': ('top:Topology', 'top:Cell', 'Associates a topology with a constituent cell.'),
 'top:hasCellComplex': ('top:Cluster',
                        'top:CellComplex',
                        'Associates a cluster or model container with a constituent cell complex.'),
 'top:hasClassification': ('top:Topology',
                           'top:ClassificationReference',
                           'Associates a topology, element, system, or mapped BIM entity with a classification '
                           'reference.'),
 'top:hasConnectedPort': ('top:Element',
                          'top:Port',
                          'Associates an element, system component, or equipment item with a connected distribution or '
                          'connection port.'),
 'top:hasConstraint': ('owl:Thing',
                       'owl:Thing',
                       'Associates an entity with a rule, constraint, requirement, limit, or validation condition.'),
 'top:hasCoordinationIssue': ('top:Topology',
                              'top:Relationship',
                              'Associates an entity with a detected coordination, clash, validation, or quality '
                              'issue.'),
 'top:hasDictionary': ('owl:Thing',
                       'top:Dictionary',
                       'Associates a topology, graph, node, relationship, or record with a TopologicPy dictionary '
                       'containing metadata, attributes, semantics, analysis values, or provenance.'),
 'top:hasDocument': ('owl:Thing',
                     'owl:Thing',
                     'Associates an entity with a document reference, external file, specification, drawing, approval '
                     'package, or supporting document.'),
 'top:hasEdge': ('owl:Thing', 'top:Edge', 'Associates a topology or graph with an edge that belongs to it.'),
 'top:hasEndVertex': ('owl:Thing',
                      'top:Vertex',
                      'Associates an edge or relationship with its end vertex or target node.'),
 'top:hasExternalBoundary': ('top:Topology',
                             'top:Boundary',
                             'Associates a topology, region, element, or analytical domain with its external '
                             'boundary.'),
 'top:hasFace': ('top:Topology', 'top:Face', 'Associates a topology with a constituent face.'),
 'top:hasIFCType': ('top:Topology',
                    'owl:Thing',
                    'Associates an IFC occurrence or mapped topology with its IFC type object.'),
 'top:hasInternalBoundary': ('top:Topology',
                             'top:Boundary',
                             'Associates a topology, region, element, or analytical domain with an internal boundary, '
                             'hole, or void boundary.'),
 'top:hasMaterial': ('top:Topology',
                     'owl:Thing',
                     'Associates a topology, element, or mapped BIM entity with a material or material set.'),
 'top:hasMissingOpening': ('top:Topology',
                           'top:Element',
                           'Associates an element or topology with a coordination issue in which an expected opening '
                           'is absent.'),
 'top:hasNode': ('top:Graph', 'top:Node', 'Associates a graph with a node that belongs to it.'),
 'top:hasOpening': ('top:Element',
                    'top:Opening',
                    'Associates an element with an opening, void, penetration, or recess.'),
 'top:hasPredicate': ('top:Relationship',
                      'rdf:Property',
                      'Associates a TopologicPy relationship record with the RDF predicate that gives the relationship '
                      'its semantic meaning.'),
 'top:hasPropertySet': ('owl:Thing',
                        'top:PropertySet',
                        'Associates a topology, element, type, system, graph entity, or relationship with a property '
                        'set.'),
 'top:hasRelationship': ('top:Graph',
                         'top:Relationship',
                         'Associates a graph with a relationship or edge that belongs to it.'),
 'top:hasShell': ('top:Topology', 'top:Shell', 'Associates a topology with a constituent shell.'),
 'top:hasStartVertex': ('owl:Thing',
                        'top:Vertex',
                        'Associates an edge or relationship with its start vertex or source node.'),
 'top:hasSubTopology': ('top:Topology',
                        'top:Topology',
                        'Associates a topology with a contained or constituent subtopology.'),
 'top:hasTopology': ('owl:Thing',
                     'top:Topology',
                     'Associates an entity with a topology that geometrically or topologically represents it.'),
 'top:hasVertex': ('owl:Thing', 'top:Vertex', 'Associates a topology or graph with a vertex that belongs to it.'),
 'top:hasWire': ('top:Topology', 'top:Wire', 'Associates a topology with a constituent wire.'),
 'top:interfaceOf': ('top:Interface',
                     'top:Topology',
                     'Associates an interface with the topology, element, space, or zone that it bounds, separates, or '
                     'connects.'),
 'top:intersects': ('top:Topology',
                    'top:Topology',
                    'Associates two topologies, elements, spaces, or regions that geometrically or topologically '
                    'intersect according to a declared tolerance or spatial predicate.'),
 'top:isAggregatedBy': ('top:Topology',
                        'top:Topology',
                        'Inverse relation of top:aggregates, associating a part with its aggregate or whole.'),
 'top:isApprovalOf': ('owl:Thing', 'owl:Thing', 'Inverse relation of top:hasApproval.'),
 'top:isCellComplexOf': ('top:CellComplex',
                         'top:Cluster',
                         'Associates a cell complex with a containing cluster or model container.'),
 'top:isCellOf': ('top:Cell', 'top:Topology', 'Associates a cell with its parent topology.'),
 'top:isClassificationOf': ('top:ClassificationReference',
                            'top:Topology',
                            'Inverse relation of top:hasClassification.'),
 'top:isConnectedPortOf': ('top:Port',
                           'top:Port',
                           'Inverse or companion relation for top:connectsPort where a directional statement is '
                           'required by an export process.'),
 'top:isConnectedTo': ('top:Topology',
                       'top:Topology',
                       'Alias property for generic semantic or topological connection.'),
 'top:isConstraintOf': ('owl:Thing', 'owl:Thing', 'Inverse relation of top:hasConstraint.'),
 'top:isDocumentOf': ('owl:Thing', 'owl:Thing', 'Inverse relation of top:hasDocument.'),
 'top:isEdgeOf': ('top:Edge', 'owl:Thing', 'Associates an edge with the topology or graph to which it belongs.'),
 'top:isFaceOf': ('top:Face', 'top:Topology', 'Associates a face with its parent topology.'),
 'top:isFilledBy': ('top:Opening',
                    'top:Element',
                    'Inverse relation of top:fillsOpening, associating an opening with the element that fills it.'),
 'top:isIFCTypeOf': ('owl:Thing', 'top:Topology', 'Inverse relation of top:hasIFCType.'),
 'top:isMaterialOf': ('owl:Thing', 'top:Topology', 'Inverse relation of top:hasMaterial.'),
 'top:isOpeningIn': ('top:Opening',
                     'top:Element',
                     'Inverse relation of top:hasOpening, associating an opening with its host element.'),
 'top:isPartOf': ('top:Topology',
                  'top:Topology',
                  'Associates a topology, element, space, or entity with a containing or aggregating whole.'),
 'top:isPropertySetOf': ('top:PropertySet', 'owl:Thing', 'Inverse relation of top:hasPropertySet.'),
 'top:isServedBy': ('top:Topology',
                    'owl:Thing',
                    'Associates a spatial structure with the system or equipment item that serves it.'),
 'top:isShellOf': ('top:Shell', 'top:Topology', 'Associates a shell with its parent topology.'),
 'top:isSubTopologyOf': ('top:Topology', 'top:Topology', 'Associates a topology with a containing or parent topology.'),
 'top:isTopologyOf': ('top:Topology',
                      'owl:Thing',
                      'Inverse relation of top:hasTopology, associating a topology with the entity it represents.'),
 'top:isVertexOf': ('top:Vertex', 'owl:Thing', 'Associates a vertex with the topology or graph to which it belongs.'),
 'top:isWireOf': ('top:Wire', 'top:Topology', 'Associates a wire with its parent topology.'),
 'top:locatedIn': ('top:Topology',
                   'top:Topology',
                   'Associates a topology, element, node, or entity with the containing, nearest, or inferred spatial '
                   'structure derived by geometric or semantic analysis.'),
 'top:passesThrough': ('top:Topology',
                       'top:Topology',
                       'Indicates that one topology, element, or system component passes through another topology, '
                       'element, space, or region.'),
 'top:requiresOpening': ('top:Topology',
                         'top:Element',
                         'Indicates that an element, system component, route, or topology requires an opening through '
                         'another element.'),
 'top:servesBuilding': ('owl:Thing',
                        'top:Building',
                        'Associates a system or equipment item with a building it serves.'),
 'top:servesSpatialStructure': ('owl:Thing',
                                'top:Topology',
                                'Associates a system or equipment item with the spatial structure it serves, such as a '
                                'site, building, storey, space, or zone.'),
 'top:startsAt': ('owl:Thing',
                  'top:Vertex',
                  'Alias property for associating an edge or relationship with its start vertex or source node.'),
 'top:violatesCoordinationRule': ('top:Topology',
                                  'top:Relationship',
                                  'Associates an entity with a violated coordination rule, model-checking rule, or '
                                  'relationship record.')}
__DATA_PROPERTIES_005 = {'top:area': ('top:Topology',
              'xsd:double',
              'The area of a face, shell, cell, cell complex, surface, spatial region, or other area-bearing topology '
              'or analytical record.'),
 'top:category': ('owl:Thing',
                  'xsd:string',
                  'A broad category value emitted from TopologicPy dictionaries, such as topology, graph, space, '
                  'element, equipment, interface, project, metadata, mathematics, or analysis.'),
 'top:createdAt': ('owl:Thing', 'xsd:dateTime', 'The creation timestamp of an entity, topology, graph, or record.'),
 'top:description': ('owl:Thing', 'xsd:string', 'A human-readable description emitted from a TopologicPy dictionary.'),
 'top:dstId': ('top:Relationship',
               'xsd:integer',
               'The destination node index of a graph relationship record, retained as structural metadata when '
               'exported.'),
 'top:feature': ('owl:Thing',
                 'xsd:string',
                 'A scalar or symbolic feature value associated with an entity, topology, graph, node, or '
                 'relationship.'),
 'top:featureVector': ('owl:Thing',
                       'xsd:string',
                       'A feature vector associated with an entity, topology, graph, node, or relationship.'),
 'top:generatedByMethod': ('owl:Thing',
                           'xsd:string',
                           'The method, function, script, notebook, or software operation that generated an entity, '
                           'topology, graph, or record, stored as a literal value.'),
 'top:hasArea': ('top:Topology',
                 'xsd:double',
                 'The area of a face, shell, cell, cell complex, surface, spatial region, or other area-bearing '
                 'topology or analytical record.'),
 'top:hasFeature': ('owl:Thing', 'xsd:string', 'Deprecated alias of top:feature.'),
 'top:hasFeatureVector': ('owl:Thing', 'xsd:string', 'Deprecated alias of top:featureVector.'),
 'top:hasLength': ('top:Topology',
                   'xsd:double',
                   'The length of an edge, wire, path, graph edge, or other length-bearing topology or analytical '
                   'record.'),
 'top:hasMantissa': ('owl:Thing',
                     'xsd:integer',
                     'The number of decimal places used to round, serialize, compare, or report numeric values.'),
 'top:hasUnit': ('owl:Thing',
                 'xsd:string',
                 'The unit of measurement associated with a value, topology, graph, metric, or record.'),
 'top:hasVolume': ('top:Topology',
                   'xsd:double',
                   'The volume of a cell, cell complex, zone, space, or other volume-bearing topology or analytical '
                   'record.'),
 'top:hasWeight': ('owl:Thing', 'xsd:double', 'Deprecated alias of top:weight.'),
 'top:hasX': ('top:Vertex', 'xsd:double', 'The X coordinate of a vertex, point, node, or graph vertex record.'),
 'top:hasY': ('top:Vertex', 'xsd:double', 'The Y coordinate of a vertex, point, node, or graph vertex record.'),
 'top:hasZ': ('top:Vertex', 'xsd:double', 'The Z coordinate of a vertex, point, node, or graph vertex record.'),
 'top:ifcClass': ('owl:Thing',
                  'xsd:string',
                  'The IFC entity class name associated with a topology, graph entity, or record, commonly stored '
                  'under the dictionary key ifc_class.'),
 'top:ifcGUID': ('owl:Thing',
                 'xsd:string',
                 'The IFC GlobalId associated with a topology, graph entity, or record, commonly stored under the '
                 'dictionary key ifc_guid.'),
 'top:ifcName': ('owl:Thing', 'xsd:string', 'The IFC Name associated with a topology, graph entity, or record.'),
 'top:ifcStepId': ('owl:Thing', 'xsd:integer', 'The file-local STEP entity id associated with an IFC entity.'),
 'top:ifcStepKey': ('owl:Thing',
                    'xsd:string',
                    'The file-local STEP entity key, such as #123, associated with an IFC entity.'),
 'top:ifcType': ('owl:Thing',
                 'xsd:string',
                 'The IFC entity type or declared type associated with a topology, graph entity, or record.'),
 'top:index': ('owl:Thing',
               'xsd:integer',
               'A file-local, graph-local, or collection-local ordinal index used for deterministic ordering or '
               'reconstruction.'),
 'top:label': ('owl:Thing',
               'xsd:string',
               'A human-readable label emitted from a TopologicPy dictionary when represented as data rather than '
               'rdfs:label.'),
 'top:length': ('top:Topology',
                'xsd:double',
                'The length of an edge, wire, path, graph edge, or other length-bearing topology or analytical '
                'record.'),
 'top:mantissa': ('owl:Thing',
                  'xsd:integer',
                  'Alias data property for mantissa when TopologicPy dictionary export emits the raw key mantissa.'),
 'top:modifiedAt': ('owl:Thing',
                    'xsd:dateTime',
                    'The last modification timestamp of an entity, topology, graph, or record.'),
 'top:name': ('owl:Thing',
              'xsd:string',
              'A human-readable name emitted from a TopologicPy dictionary when not represented as rdfs:label.'),
 'top:ontologyClass': ('owl:Thing',
                       'xsd:string',
                       'The ontology class QName or URI recorded in a TopologicPy dictionary, commonly stored under '
                       'the dictionary key ontology_class.'),
 'top:ontologyURI': ('owl:Thing',
                     'xsd:anyURI',
                     'The expanded ontology URI recorded in a TopologicPy dictionary, commonly stored under the '
                     'dictionary key ontology_uri.'),
 'top:relationship': ('owl:Thing',
                      'xsd:string',
                      'A general-purpose relationship label emitted from TopologicPy dictionaries when a more specific '
                      'ontology predicate is not available.'),
 'top:source': ('owl:Thing',
                'xsd:string',
                'The source file, model, database, method, or process associated with an entity, topology, graph, or '
                'record.'),
 'top:srcId': ('top:Relationship',
               'xsd:integer',
               'The source node index of a graph relationship record, retained as structural metadata when exported.'),
 'top:unit': ('owl:Thing',
              'xsd:string',
              'Alias data property for unit when TopologicPy dictionary export emits the raw key unit.'),
 'top:uuid': ('owl:Thing',
              'xsd:string',
              'A stable UUID or UUID-like identifier associated with an entity, topology, graph, node, or '
              'relationship.'),
 'top:volume': ('top:Topology',
                'xsd:double',
                'The volume of a cell, cell complex, zone, space, or other volume-bearing topology or analytical '
                'record.'),
 'top:weight': ('owl:Thing', 'xsd:double', 'A numeric graph, path, analysis, or relationship weight.'),
 'top:x': ('top:Vertex',
           'xsd:double',
           'Alias data property for the X coordinate when TopologicPy dictionary export emits the raw key x.'),
 'top:y': ('top:Vertex',
           'xsd:double',
           'Alias data property for the Y coordinate when TopologicPy dictionary export emits the raw key y.'),
 'top:z': ('top:Vertex',
           'xsd:double',
           'Alias data property for the Z coordinate when TopologicPy dictionary export emits the raw key z.')}
__DEPRECATED_PROPERTIES_005 = ['top:hasArea',
 'top:hasEndVertex',
 'top:hasFeature',
 'top:hasFeatureVector',
 'top:hasLength',
 'top:hasMantissa',
 'top:hasStartVertex',
 'top:hasUnit',
 'top:hasVolume',
 'top:hasWeight',
 'top:hasX',
 'top:hasY',
 'top:hasZ']
__ONTOLOGY_TRIPLES_005 = [('http://w3id.org/topologicpy', 'dcterms:contributor', '"Wassim Jabi"@en'),
 ('http://w3id.org/topologicpy', 'dcterms:created', '"2026-05-24"^^xsd:date'),
 ('http://w3id.org/topologicpy', 'dcterms:creator', '"TopologicPy Project"@en'),
 ('http://w3id.org/topologicpy',
  'dcterms:description',
  '"An OWL/RDFS vocabulary for TopologicPy topology, graph structures, BIM/IFC/BOT/Brick alignment, dictionary '
  'metadata, graph-machine-learning metadata, provenance, and analysis results."@en'),
 ('http://w3id.org/topologicpy', 'dcterms:hasVersion', '"0.4.2"'),
 ('http://w3id.org/topologicpy', 'dcterms:issued', '"2026-05-24"^^xsd:date'),
 ('http://w3id.org/topologicpy', 'dcterms:license', 'https://www.gnu.org/licenses/agpl-3.0.en.html'),
 ('http://w3id.org/topologicpy', 'dcterms:modified', '"2026-07-02"^^xsd:date'),
 ('http://w3id.org/topologicpy', 'dcterms:title', '"TopologicPy Ontology"@en'),
 ('http://w3id.org/topologicpy', 'owl:versionInfo', '"0.4.2"'),
 ('http://w3id.org/topologicpy', 'rdf:type', 'owl:Ontology'),
 ('http://w3id.org/topologicpy',
  'rdfs:comment',
  '"The canonical namespace prefix is top:. BOT, Brick, GeoSPARQL, IFC, PROV-O, SKOS, RDF, RDFS, OWL, and XSD terms '
  'are referenced for interoperability but are not redefined here. Arbitrary user dictionary keys should use the dict: '
  'namespace, not the top: ontology namespace."@en'),
 ('http://w3id.org/topologicpy', 'rdfs:label', '"TopologicPy Ontology"@en'),
 ('http://w3id.org/topologicpy', 'vann:preferredNamespacePrefix', '"top"'),
 ('http://w3id.org/topologicpy', 'vann:preferredNamespaceUri', '"http://w3id.org/topologicpy#"'),
 ('top:AccessGraph', 'rdf:type', 'owl:Class'),
 ('top:AccessGraph',
  'rdfs:comment',
  '"A directed or undirected graph used to represent permitted access between spaces, rooms, zones, portals, doors, '
  'apertures, circulation elements, or other traversable entities. Edges may encode passability, access control, '
  'traversal direction, door type, cost, distance, or movement constraints."@en'),
 ('top:AccessGraph', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:AccessGraph', 'rdfs:label', '"Access Graph"@en'),
 ('top:AccessGraph', 'rdfs:subClassOf', 'top:SpatialGraph'),
 ('top:AccessGraph', 'skos:altLabel', '"AccessGraph"@en'),
 ('top:AdjacencyGraph', 'rdf:type', 'owl:Class'),
 ('top:AdjacencyGraph',
  'rdfs:comment',
  '"An undirected graph used to represent adjacency between topological, geometric, spatial, or semantic entities. '
  'Edges typically indicate that two entities touch, share a boundary, are incident, are proximal within a specified '
  'tolerance, or satisfy a declared adjacency predicate."@en'),
 ('top:AdjacencyGraph', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:AdjacencyGraph', 'rdfs:label', '"Adjacency Graph"@en'),
 ('top:AdjacencyGraph', 'rdfs:subClassOf', 'top:SpatialGraph'),
 ('top:AdjacencyGraph', 'skos:altLabel', '"AdjacencyGraph"@en'),
 ('top:AnalysisGraph', 'rdf:type', 'owl:Class'),
 ('top:AnalysisGraph',
  'rdfs:comment',
  '"A directed or undirected graph used to represent entities and relationships prepared for computational analysis. '
  'Vertices, edges, and graph-level dictionaries may encode analytical attributes, weights, labels, features, metrics, '
  'simulation values, or machine-learning inputs and outputs."@en'),
 ('top:AnalysisGraph', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:AnalysisGraph', 'rdfs:label', '"Analysis Graph"@en'),
 ('top:AnalysisGraph', 'rdfs:subClassOf', 'top:Graph'),
 ('top:AnalysisGraph', 'skos:altLabel', '"AnalysisGraph"@en'),
 ('top:AnalysisMetric', 'rdf:type', 'owl:Class'),
 ('top:AnalysisMetric',
  'rdfs:comment',
  '"A computed metric associated with geometry, topology, graphs, buildings, systems, simulation, performance '
  'analysis, or machine-learning workflows."@en'),
 ('top:AnalysisMetric', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:AnalysisMetric', 'rdfs:label', '"Analysis Metric"@en'),
 ('top:AnalysisMetric', 'skos:altLabel', '"AnalysisMetric"@en'),
 ('top:Aperture', 'rdf:type', 'owl:Class'),
 ('top:Aperture',
  'rdfs:comment',
  '"A topological or building-domain opening associated with a host topology or element. In TopologicPy it is treated '
  'as an aperture/opening construct and may be represented geometrically by a face-like boundary."@en'),
 ('top:Aperture', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Aperture', 'rdfs:label', '"Aperture"@en'),
 ('top:Aperture', 'rdfs:subClassOf', 'bot:Element'),
 ('top:Aperture', 'rdfs:subClassOf', 'top:Element'),
 ('top:Aperture', 'rdfs:subClassOf', 'top:Face'),
 ('top:Attribute', 'rdf:type', 'owl:Class'),
 ('top:Attribute',
  'rdfs:comment',
  '"A semantic or analytical key-value attribute associated with a topology, graph, node, relationship, dataset, '
  'metric, or external entity."@en'),
 ('top:Attribute', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Attribute', 'rdfs:label', '"Attribute"@en'),
 ('top:Beam', 'rdf:type', 'owl:Class'),
 ('top:Beam', 'rdfs:comment', '"A beam, girder, or linear structural member."@en'),
 ('top:Beam', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Beam', 'rdfs:label', '"Beam"@en'),
 ('top:Beam', 'rdfs:subClassOf', 'bot:Element'),
 ('top:Beam', 'rdfs:subClassOf', 'top:Element'),
 ('top:Beam', 'skos:closeMatch', 'ifc:IfcBeam'),
 ('top:Boundary', 'rdf:type', 'owl:Class'),
 ('top:Boundary',
  'rdfs:comment',
  '"A topology used to represent the boundary of another topology, spatial region, element, or analytical domain."@en'),
 ('top:Boundary', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Boundary', 'rdfs:label', '"Boundary"@en'),
 ('top:Boundary', 'rdfs:subClassOf', 'top:Topology'),
 ('top:Building', 'rdf:type', 'owl:Class'),
 ('top:Building',
  'rdfs:comment',
  '"A building-level spatial container used to represent a whole building, facility, or IFC building."@en'),
 ('top:Building', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Building', 'rdfs:label', '"Building"@en'),
 ('top:Building', 'rdfs:subClassOf', 'bot:Building'),
 ('top:Building', 'rdfs:subClassOf', 'top:Zone'),
 ('top:Building', 'skos:closeMatch', 'ifc:IfcBuilding'),
 ('top:Cell', 'rdf:type', 'owl:Class'),
 ('top:Cell',
  'rdfs:comment',
  '"A three-dimensional TopologicPy topology bounded by faces or shells and representing a volumetric region."@en'),
 ('top:Cell', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Cell', 'rdfs:label', '"Cell"@en'),
 ('top:Cell', 'rdfs:subClassOf', 'top:Topology'),
 ('top:CellComplex', 'rdf:type', 'owl:Class'),
 ('top:CellComplex',
  'rdfs:comment',
  '"A TopologicPy topology composed of cells, typically with non-manifold adjacency through shared faces."@en'),
 ('top:CellComplex', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:CellComplex', 'rdfs:label', '"Cell Complex"@en'),
 ('top:CellComplex', 'rdfs:subClassOf', 'top:Topology'),
 ('top:CellComplex', 'skos:altLabel', '"CellComplex"@en'),
 ('top:CirculationGraph', 'rdf:type', 'owl:Class'),
 ('top:CirculationGraph',
  'rdfs:comment',
  '"A directed or undirected graph used to represent movement, circulation, or route-choice relationships among '
  'spaces, portals, corridors, stairs, entrances, exits, or waypoints. Edges may encode route length, cost, capacity, '
  'direction, or accessibility."@en'),
 ('top:CirculationGraph', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:CirculationGraph', 'rdfs:label', '"Circulation Graph"@en'),
 ('top:CirculationGraph', 'rdfs:subClassOf', 'top:SpatialGraph'),
 ('top:CirculationGraph', 'skos:altLabel', '"CirculationGraph"@en'),
 ('top:CirculationZone', 'rdf:type', 'owl:Class'),
 ('top:CirculationZone',
  'rdfs:comment',
  '"A zone primarily used to represent movement, access, circulation, transition, or evacuation functions."@en'),
 ('top:CirculationZone', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:CirculationZone', 'rdfs:label', '"Circulation Zone"@en'),
 ('top:CirculationZone', 'rdfs:subClassOf', 'bot:Zone'),
 ('top:CirculationZone', 'rdfs:subClassOf', 'top:Zone'),
 ('top:CirculationZone', 'skos:altLabel', '"CirculationZone"@en'),
 ('top:ClassificationReference', 'rdf:type', 'owl:Class'),
 ('top:ClassificationReference',
  'rdfs:comment',
  '"A classification reference used to represent external classification systems, codes, taxonomies, or IFC '
  'classification references."@en'),
 ('top:ClassificationReference', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:ClassificationReference', 'rdfs:label', '"Classification Reference"@en'),
 ('top:ClassificationReference', 'skos:altLabel', '"ClassificationReference"@en'),
 ('top:ClassificationReference', 'skos:closeMatch', 'ifc:IfcClassificationReference'),
 ('top:Cluster', 'rdf:type', 'owl:Class'),
 ('top:Cluster',
  'rdfs:comment',
  '"A heterogeneous TopologicPy topology used to group vertices, edges, wires, faces, shells, cells, cell complexes, '
  'or other clusters without requiring manifold connectivity."@en'),
 ('top:Cluster', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Cluster', 'rdfs:label', '"Cluster"@en'),
 ('top:Cluster', 'rdfs:subClassOf', 'top:Topology'),
 ('top:Column', 'rdf:type', 'owl:Class'),
 ('top:Column', 'rdfs:comment', '"A column or vertical structural member."@en'),
 ('top:Column', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Column', 'rdfs:label', '"Column"@en'),
 ('top:Column', 'rdfs:subClassOf', 'bot:Element'),
 ('top:Column', 'rdfs:subClassOf', 'top:Element'),
 ('top:Column', 'skos:closeMatch', 'ifc:IfcColumn'),
 ('top:ConnectivityGraph', 'rdf:type', 'owl:Class'),
 ('top:ConnectivityGraph',
  'rdfs:comment',
  '"A directed or undirected graph used to represent generic connectivity or reachability among topological, spatial, '
  'building, system, or graph entities. Edges indicate that two entities are connected according to an explicit '
  'computational or semantic rule."@en'),
 ('top:ConnectivityGraph', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:ConnectivityGraph', 'rdfs:label', '"Connectivity Graph"@en'),
 ('top:ConnectivityGraph', 'rdfs:subClassOf', 'top:SpatialGraph'),
 ('top:ConnectivityGraph', 'skos:altLabel', '"ConnectivityGraph"@en'),
 ('top:Context', 'rdf:type', 'owl:Class'),
 ('top:Context',
  'rdfs:comment',
  '"A contextual reference used by TopologicPy to associate content with a host topology or subtopology."@en'),
 ('top:Context', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Context', 'rdfs:label', '"Context"@en'),
 ('top:CurtainWall', 'rdf:type', 'owl:Class'),
 ('top:CurtainWall', 'rdfs:comment', '"A curtain-wall element represented as a specialised wall or façade system."@en'),
 ('top:CurtainWall', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:CurtainWall', 'rdfs:label', '"Curtain Wall"@en'),
 ('top:CurtainWall', 'rdfs:subClassOf', 'bot:Element'),
 ('top:CurtainWall', 'rdfs:subClassOf', 'top:Wall'),
 ('top:CurtainWall', 'skos:altLabel', '"CurtainWall"@en'),
 ('top:CurtainWall', 'skos:closeMatch', 'ifc:IfcCurtainWall'),
 ('top:Dictionary', 'rdf:type', 'owl:Class'),
 ('top:Dictionary',
  'rdfs:comment',
  '"A key-value metadata container used by TopologicPy to attach semantic, analytical, geometric, provenance, and '
  'interoperability data to topologies, graphs, vertices, edges, and records."@en'),
 ('top:Dictionary', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Dictionary', 'rdfs:label', '"Dictionary"@en'),
 ('top:DirectedRelationship', 'rdf:type', 'owl:Class'),
 ('top:DirectedRelationship',
  'rdfs:comment',
  '"A directed graph relationship used to represent an ordered source-to-target relation between two nodes."@en'),
 ('top:DirectedRelationship', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:DirectedRelationship', 'rdfs:label', '"Directed Relationship"@en'),
 ('top:DirectedRelationship', 'rdfs:subClassOf', 'top:Relationship'),
 ('top:DirectedRelationship', 'skos:altLabel', '"DirectedRelationship"@en'),
 ('top:Door', 'rdf:type', 'owl:Class'),
 ('top:Door',
  'rdfs:comment',
  '"A door, access, or operable passage element, commonly associated with openings, access graphs, and navigation '
  'relationships."@en'),
 ('top:Door', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Door', 'rdfs:label', '"Door"@en'),
 ('top:Door', 'rdfs:subClassOf', 'bot:Element'),
 ('top:Door', 'rdfs:subClassOf', 'top:Element'),
 ('top:Door', 'skos:closeMatch', 'ifc:IfcDoor'),
 ('top:DualGraph', 'rdf:type', 'owl:Class'),
 ('top:DualGraph',
  'rdfs:comment',
  '"An undirected graph used to represent adjacency between higher-dimensional entities such as faces, cells, rooms, '
  'spaces, zones, or regions. Each entity is represented as a vertex, and an edge represents a shared boundary, '
  'interface, aperture, adjacency relation, or other declared connection between two entities."@en'),
 ('top:DualGraph', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:DualGraph', 'rdfs:label', '"Dual Graph"@en'),
 ('top:DualGraph', 'rdfs:subClassOf', 'top:SpatialGraph'),
 ('top:DualGraph', 'skos:altLabel', '"DualGraph"@en'),
 ('top:Edge', 'rdf:type', 'owl:Class'),
 ('top:Edge',
  'rdfs:comment',
  '"A one-dimensional TopologicPy topology connecting a start vertex to an end vertex."@en'),
 ('top:Edge', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Edge', 'rdfs:label', '"Edge"@en'),
 ('top:Edge', 'rdfs:subClassOf', 'top:Topology'),
 ('top:EdgeFeature', 'rdf:type', 'owl:Class'),
 ('top:EdgeFeature',
  'rdfs:comment',
  '"A numerical, categorical, vector, or encoded feature associated with a graph edge or relationship record."@en'),
 ('top:EdgeFeature', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:EdgeFeature', 'rdfs:label', '"Edge Feature"@en'),
 ('top:EdgeFeature', 'rdfs:subClassOf', 'top:Attribute'),
 ('top:EdgeFeature', 'skos:altLabel', '"EdgeFeature"@en'),
 ('top:Element', 'rdf:type', 'owl:Class'),
 ('top:Element',
  'rdfs:comment',
  '"A physical, conceptual, or analytical building/infrastructure element that may be represented by topology and '
  'aligned with BIM entities."@en'),
 ('top:Element', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Element', 'rdfs:label', '"Element"@en'),
 ('top:Element', 'rdfs:subClassOf', 'bot:Element'),
 ('top:Element', 'rdfs:subClassOf', 'top:Topology'),
 ('top:Element', 'skos:closeMatch', 'ifc:IfcElement'),
 ('top:Equipment', 'rdf:type', 'owl:Class'),
 ('top:Equipment',
  'rdfs:comment',
  '"A building, MEP, operational, or service equipment element, optionally aligned with Brick equipment classes when '
  'system semantics are available."@en'),
 ('top:Equipment', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Equipment', 'rdfs:label', '"Equipment"@en'),
 ('top:Equipment', 'rdfs:subClassOf', 'brick:Equipment'),
 ('top:Equipment', 'rdfs:subClassOf', 'top:Element'),
 ('top:Equipment', 'skos:closeMatch', 'ifc:IfcDistributionElement'),
 ('top:ExternalBoundary', 'rdf:type', 'owl:Class'),
 ('top:ExternalBoundary',
  'rdfs:comment',
  '"A boundary topology used to represent the outer boundary of another topology, region, element, or analytical '
  'domain."@en'),
 ('top:ExternalBoundary', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:ExternalBoundary', 'rdfs:label', '"External Boundary"@en'),
 ('top:ExternalBoundary', 'rdfs:subClassOf', 'top:Boundary'),
 ('top:ExternalBoundary', 'skos:altLabel', '"ExternalBoundary"@en'),
 ('top:Face', 'rdf:type', 'owl:Class'),
 ('top:Face',
  'rdfs:comment',
  '"A two-dimensional TopologicPy topology bounded by an external wire and, optionally, one or more internal boundary '
  'wires."@en'),
 ('top:Face', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Face', 'rdfs:label', '"Face"@en'),
 ('top:Face', 'rdfs:subClassOf', 'top:Topology'),
 ('top:FunctionalZone', 'rdf:type', 'owl:Class'),
 ('top:FunctionalZone',
  'rdfs:comment',
  '"A zone grouped by use, function, programme, activity, department, or other non-geometric classification."@en'),
 ('top:FunctionalZone', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:FunctionalZone', 'rdfs:label', '"Functional Zone"@en'),
 ('top:FunctionalZone', 'rdfs:subClassOf', 'bot:Zone'),
 ('top:FunctionalZone', 'rdfs:subClassOf', 'top:Zone'),
 ('top:FunctionalZone', 'skos:altLabel', '"FunctionalZone"@en'),
 ('top:Furniture', 'rdf:type', 'owl:Class'),
 ('top:Furniture', 'rdfs:comment', '"A furniture or furnishing element."@en'),
 ('top:Furniture', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Furniture', 'rdfs:label', '"Furniture"@en'),
 ('top:Furniture', 'rdfs:subClassOf', 'bot:Element'),
 ('top:Furniture', 'rdfs:subClassOf', 'top:Element'),
 ('top:Furniture', 'skos:closeMatch', 'ifc:IfcFurnishingElement'),
 ('top:Graph', 'owl:equivalentClass', 'top:TGraph'),
 ('top:Graph', 'rdf:type', 'owl:Class'),
 ('top:Graph',
  'rdfs:comment',
  '"A directed, undirected, mixed, weighted, unweighted, simple, or multigraph representation used to represent '
  'entities and relationships emitted, consumed, analysed, or transformed by TopologicPy. More specific graph '
  'semantics are defined by subclasses."@en'),
 ('top:Graph', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Graph', 'rdfs:label', '"Graph"@en'),
 ('top:GraphDataset', 'rdf:type', 'owl:Class'),
 ('top:GraphDataset',
  'rdfs:comment',
  '"A dataset used to represent graphs, nodes, edges, labels, masks, features, metadata, and ontology annotations for '
  'graph analysis or graph machine learning."@en'),
 ('top:GraphDataset', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:GraphDataset', 'rdfs:label', '"Graph Dataset"@en'),
 ('top:GraphDataset', 'skos:altLabel', '"GraphDataset"@en'),
 ('top:GraphFeature', 'rdf:type', 'owl:Class'),
 ('top:GraphFeature',
  'rdfs:comment',
  '"A numerical, categorical, vector, or encoded feature associated with a graph-level record."@en'),
 ('top:GraphFeature', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:GraphFeature', 'rdfs:label', '"Graph Feature"@en'),
 ('top:GraphFeature', 'rdfs:subClassOf', 'top:Attribute'),
 ('top:GraphFeature', 'skos:altLabel', '"GraphFeature"@en'),
 ('top:Grid', 'rdf:type', 'owl:Class'),
 ('top:Grid',
  'rdfs:comment',
  '"A spatial subdivision, sampling, or reference structure used to generate or organise vertices, edges, cells, or '
  'analysis samples."@en'),
 ('top:Grid', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Grid', 'rdfs:label', '"Grid"@en'),
 ('top:HasseDiagramGraph', 'rdf:type', 'owl:Class'),
 ('top:HasseDiagramGraph',
  'rdfs:comment',
  '"A directed acyclic graph used to represent a partially ordered set, typically by showing only cover relations '
  'after transitive reduction. In TopologicPy, it may represent hierarchical containment, decomposition, incidence, or '
  'boundary relationships among topological entities."@en'),
 ('top:HasseDiagramGraph', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:HasseDiagramGraph', 'rdfs:label', '"Hasse Diagram Graph"@en'),
 ('top:HasseDiagramGraph', 'rdfs:subClassOf', 'top:Graph'),
 ('top:HasseDiagramGraph', 'skos:altLabel', '"HasseDiagramGraph"@en'),
 ('top:Interface', 'rdf:type', 'owl:Class'),
 ('top:Interface',
  'rdfs:comment',
  '"A shared boundary or interface between spaces, zones, elements, or topologies, including IFC space-boundary-style '
  'relationships."@en'),
 ('top:Interface', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Interface', 'rdfs:label', '"Interface"@en'),
 ('top:Interface', 'rdfs:subClassOf', 'bot:Interface'),
 ('top:Interface', 'rdfs:subClassOf', 'top:Face'),
 ('top:Interface', 'skos:closeMatch', 'ifc:IfcRelSpaceBoundary'),
 ('top:InternalBoundary', 'rdf:type', 'owl:Class'),
 ('top:InternalBoundary',
  'rdfs:comment',
  '"A boundary topology used to represent a hole, void, inner loop, or internal boundary of another topology, region, '
  'element, or analytical domain."@en'),
 ('top:InternalBoundary', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:InternalBoundary', 'rdfs:label', '"Internal Boundary"@en'),
 ('top:InternalBoundary', 'rdfs:subClassOf', 'top:Boundary'),
 ('top:InternalBoundary', 'skos:altLabel', '"InternalBoundary"@en'),
 ('top:Isovist', 'rdf:type', 'owl:Class'),
 ('top:Isovist',
  'rdfs:comment',
  '"A visibility field, polygon, volume, or related visual metric computed from an observation point or region."@en'),
 ('top:Isovist', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Isovist', 'rdfs:label', '"Isovist"@en'),
 ('top:Isovist', 'rdfs:subClassOf', 'top:AnalysisMetric'),
 ('top:IsovistGraph', 'rdf:type', 'owl:Class'),
 ('top:IsovistGraph',
  'rdfs:comment',
  '"A directed or undirected spatial graph used to represent isovists, visibility fields, or relationships between '
  'observation points and visible spatial extents. Vertices may represent viewpoints, isovist polygons, spatial '
  'samples, or visible objects, while edges may encode visibility, overlap, containment, or intervisibility '
  'relationships."@en'),
 ('top:IsovistGraph', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:IsovistGraph', 'rdfs:label', '"Isovist Graph"@en'),
 ('top:IsovistGraph', 'rdfs:subClassOf', 'top:SpatialGraph'),
 ('top:IsovistGraph', 'skos:altLabel', '"IsovistGraph"@en'),
 ('top:KnowledgeGraph', 'rdf:type', 'owl:Class'),
 ('top:KnowledgeGraph',
  'rdfs:comment',
  '"A directed semantic graph used to represent entities, attributes, relationships, and inferred knowledge. In '
  'TopologicPy, it may encode topological objects, BIM entities, ontology classes, RDF triples, rules, reasoning '
  'outputs, or links between geometry, topology, semantics, analysis, and external knowledge sources."@en'),
 ('top:KnowledgeGraph', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:KnowledgeGraph', 'rdfs:label', '"Knowledge Graph"@en'),
 ('top:KnowledgeGraph', 'rdfs:subClassOf', 'top:Graph'),
 ('top:KnowledgeGraph', 'skos:altLabel', '"KnowledgeGraph"@en'),
 ('top:LineGraph', 'rdf:type', 'owl:Class'),
 ('top:LineGraph',
  'rdfs:comment',
  '"An undirected or directed graph used to represent edge-to-edge relationships derived from another graph. Each '
  'vertex in the line graph represents an edge of the source graph, and each edge in the line graph represents '
  'adjacency, incidence, continuity, succession, or compatibility between source edges."@en'),
 ('top:LineGraph', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:LineGraph', 'rdfs:label', '"Line Graph"@en'),
 ('top:LineGraph', 'rdfs:subClassOf', 'top:Graph'),
 ('top:LineGraph', 'skos:altLabel', '"LineGraph"@en'),
 ('top:Material', 'rdf:type', 'owl:Class'),
 ('top:Material',
  'rdfs:comment',
  '"A material record used to represent material identity, composition, assignment, or IFC material information."@en'),
 ('top:Material', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Material', 'rdfs:label', '"Material"@en'),
 ('top:Material', 'skos:closeMatch', 'ifc:IfcMaterial'),
 ('top:MaterialSet', 'rdf:type', 'owl:Class'),
 ('top:MaterialSet',
  'rdfs:comment',
  '"A material-set record used to represent material lists, material layer sets, material constituent sets, or other '
  'grouped material definitions."@en'),
 ('top:MaterialSet', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:MaterialSet', 'rdfs:label', '"Material Set"@en'),
 ('top:MaterialSet', 'skos:altLabel', '"MaterialSet"@en'),
 ('top:MaterialSet', 'skos:closeMatch', 'ifc:IfcMaterialLayerSet'),
 ('top:Matrix', 'rdf:type', 'owl:Class'),
 ('top:Matrix',
  'rdfs:comment',
  '"A numerical matrix used to represent transformations, coordinate operations, or other matrix-valued '
  'quantities."@en'),
 ('top:Matrix', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Matrix', 'rdfs:label', '"Matrix"@en'),
 ('top:Member', 'rdf:type', 'owl:Class'),
 ('top:Member',
  'rdfs:comment',
  '"A generic structural, framing, or linear member when a more specific class such as Beam or Column is not '
  'asserted."@en'),
 ('top:Member', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Member', 'rdfs:label', '"Member"@en'),
 ('top:Member', 'rdfs:subClassOf', 'bot:Element'),
 ('top:Member', 'rdfs:subClassOf', 'top:Element'),
 ('top:Member', 'skos:closeMatch', 'ifc:IfcMember'),
 ('top:NavigationGraph', 'rdf:type', 'owl:Class'),
 ('top:NavigationGraph',
  'rdfs:comment',
  '"A directed or undirected weighted graph used to represent navigable movement through spaces, paths, portals, '
  'waypoints, circulation systems, or walkable regions. Edges typically encode traversability and may store distance, '
  'travel time, slope, accessibility, obstruction, cost, or agent-specific movement constraints."@en'),
 ('top:NavigationGraph', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:NavigationGraph', 'rdfs:label', '"Navigation Graph"@en'),
 ('top:NavigationGraph', 'rdfs:subClassOf', 'top:SpatialGraph'),
 ('top:NavigationGraph', 'skos:altLabel', '"NavigationGraph"@en'),
 ('top:Node', 'rdf:type', 'owl:Class'),
 ('top:Node',
  'rdfs:comment',
  '"A graph node represented by, or projected from, a TopologicPy vertex or TGraph vertex record. Nodes may carry '
  'dictionary metadata, ontology class information, labels, features, coordinates, and external identifiers."@en'),
 ('top:Node', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Node', 'rdfs:label', '"Node"@en'),
 ('top:Node', 'rdfs:subClassOf', 'top:Vertex'),
 ('top:NodeFeature', 'rdf:type', 'owl:Class'),
 ('top:NodeFeature',
  'rdfs:comment',
  '"A numerical, categorical, vector, or encoded feature associated with a graph node or vertex record."@en'),
 ('top:NodeFeature', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:NodeFeature', 'rdfs:label', '"Node Feature"@en'),
 ('top:NodeFeature', 'rdfs:subClassOf', 'top:Attribute'),
 ('top:NodeFeature', 'skos:altLabel', '"NodeFeature"@en'),
 ('top:Opening', 'rdf:type', 'owl:Class'),
 ('top:Opening',
  'rdfs:comment',
  '"An opening element, void, recess, penetration, or host-element subtraction associated with doors, windows, '
  'services, or coordination checks."@en'),
 ('top:Opening', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Opening', 'rdfs:label', '"Opening"@en'),
 ('top:Opening', 'rdfs:subClassOf', 'bot:Element'),
 ('top:Opening', 'rdfs:subClassOf', 'top:Element'),
 ('top:Opening', 'skos:closeMatch', 'ifc:IfcOpeningElement'),
 ('top:Path', 'rdf:type', 'owl:Class'),
 ('top:Path',
  'rdfs:comment',
  '"An ordered graph structure used to represent a sequence of nodes and relationships, typically returned by routing, '
  'shortest-path, circulation, or navigation methods."@en'),
 ('top:Path', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Path', 'rdfs:label', '"Path"@en'),
 ('top:Path', 'rdfs:subClassOf', 'top:Graph'),
 ('top:Point', 'rdf:type', 'owl:Class'),
 ('top:Point', 'rdfs:comment', '"A point-like spatial entity represented as a specialised vertex."@en'),
 ('top:Point', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Point', 'rdfs:label', '"Point"@en'),
 ('top:Point', 'rdfs:subClassOf', 'top:Vertex'),
 ('top:Port', 'rdf:type', 'owl:Class'),
 ('top:Port',
  'rdfs:comment',
  '"A distribution or connection port used to represent connection points on systems, equipment, MEP components, or '
  'IFC distribution ports."@en'),
 ('top:Port', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Port', 'rdfs:label', '"Port"@en'),
 ('top:PrimalGraph', 'rdf:type', 'owl:Class'),
 ('top:PrimalGraph',
  'rdfs:comment',
  '"A directed or undirected graph used to represent direct physical, spatial, topological, or semantic connectivity '
  'among entities in their original form. In spatial and topological modelling, vertices and edges typically '
  'correspond directly to objects, points, spaces, elements, or their explicit connections, before any dual '
  'transformation is applied."@en'),
 ('top:PrimalGraph', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:PrimalGraph', 'rdfs:label', '"Primal Graph"@en'),
 ('top:PrimalGraph', 'rdfs:subClassOf', 'top:SpatialGraph'),
 ('top:PrimalGraph', 'skos:altLabel', '"PrimalGraph"@en'),
 ('top:Project', 'rdf:type', 'owl:Class'),
 ('top:Project',
  'rdfs:comment',
  '"A project-level container used to represent a model, dataset, building project, computational study, or IFC '
  'project."@en'),
 ('top:Project', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Project', 'rdfs:label', '"Project"@en'),
 ('top:Project', 'rdfs:subClassOf', 'prov:Entity'),
 ('top:Project', 'skos:closeMatch', 'ifc:IfcProject'),
 ('top:PropertySet', 'rdf:type', 'owl:Class'),
 ('top:PropertySet',
  'rdfs:comment',
  '"A property set used to represent grouped name-value properties, commonly mapped from IFC property sets or related '
  'property-definition entities."@en'),
 ('top:PropertySet', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:PropertySet', 'rdfs:label', '"Property Set"@en'),
 ('top:PropertySet', 'skos:altLabel', '"PropertySet"@en'),
 ('top:PropertySet', 'skos:closeMatch', 'ifc:IfcPropertySet'),
 ('top:QualityIssue', 'rdf:type', 'owl:Class'),
 ('top:QualityIssue',
  'rdfs:comment',
  '"A validation, model-checking, coordination, geometry, topology, semantic, or data-quality issue detected during '
  'analysis."@en'),
 ('top:QualityIssue', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:QualityIssue', 'rdfs:label', '"Quality Issue"@en'),
 ('top:QualityIssue', 'skos:altLabel', '"QualityIssue"@en'),
 ('top:Quantity', 'rdf:type', 'owl:Class'),
 ('top:Quantity',
  'rdfs:comment',
  '"A quantity record used to represent measured or computed quantities, commonly mapped from IFC element '
  'quantities."@en'),
 ('top:Quantity', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Quantity', 'rdfs:label', '"Quantity"@en'),
 ('top:Quantity', 'skos:closeMatch', 'ifc:IfcElementQuantity'),
 ('top:QuotientGraph', 'rdf:type', 'owl:Class'),
 ('top:QuotientGraph',
  'rdfs:comment',
  '"A directed or undirected graph used to represent a simplified graph obtained by grouping vertices, edges, spaces, '
  'zones, components, or other entities according to an equivalence relation or partition. Vertices represent groups '
  'or classes, and edges represent relationships between those groups inherited from the source graph."@en'),
 ('top:QuotientGraph', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:QuotientGraph', 'rdfs:label', '"Quotient Graph"@en'),
 ('top:QuotientGraph', 'rdfs:subClassOf', 'top:Graph'),
 ('top:QuotientGraph', 'skos:altLabel', '"QuotientGraph"@en'),
 ('top:Railing', 'rdf:type', 'owl:Class'),
 ('top:Railing', 'rdfs:comment', '"A railing, guard, handrail, or balustrade element."@en'),
 ('top:Railing', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Railing', 'rdfs:label', '"Railing"@en'),
 ('top:Railing', 'rdfs:subClassOf', 'bot:Element'),
 ('top:Railing', 'rdfs:subClassOf', 'top:Element'),
 ('top:Railing', 'skos:closeMatch', 'ifc:IfcRailing'),
 ('top:Relationship', 'rdf:type', 'owl:Class'),
 ('top:Relationship',
  'rdfs:comment',
  '"A graph relationship represented by, or projected from, a TopologicPy edge or TGraph edge record. Relationships '
  'may carry source and target references, directionality, weights, predicates, labels, features, and dictionary '
  'metadata."@en'),
 ('top:Relationship', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Relationship', 'rdfs:label', '"Relationship"@en'),
 ('top:Relationship', 'rdfs:subClassOf', 'top:Edge'),
 ('top:Roof', 'rdf:type', 'owl:Class'),
 ('top:Roof', 'rdfs:comment', '"A roof or roof-like envelope element."@en'),
 ('top:Roof', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Roof', 'rdfs:label', '"Roof"@en'),
 ('top:Roof', 'rdfs:subClassOf', 'bot:Element'),
 ('top:Roof', 'rdfs:subClassOf', 'top:Element'),
 ('top:Roof', 'skos:closeMatch', 'ifc:IfcRoof'),
 ('top:Room', 'rdf:type', 'owl:Class'),
 ('top:Room', 'rdfs:comment', '"A room-like occupiable space represented as a specialised building space."@en'),
 ('top:Room', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Room', 'rdfs:label', '"Room"@en'),
 ('top:Room', 'rdfs:subClassOf', 'bot:Space'),
 ('top:Room', 'rdfs:subClassOf', 'top:Space'),
 ('top:Room', 'skos:closeMatch', 'ifc:IfcSpace'),
 ('top:SemanticGraph', 'rdf:type', 'owl:Class'),
 ('top:SemanticGraph',
  'rdfs:comment',
  '"A directed graph used to represent typed entities and typed relationships with explicit semantic meaning. In '
  'TopologicPy, it may encode classes, instances, properties, labels, attributes, ontology terms, BIM semantics, or '
  'RDF-style subject-predicate-object statements."@en'),
 ('top:SemanticGraph', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:SemanticGraph', 'rdfs:label', '"Semantic Graph"@en'),
 ('top:SemanticGraph', 'rdfs:subClassOf', 'top:Graph'),
 ('top:SemanticGraph', 'skos:altLabel', '"SemanticGraph"@en'),
 ('top:Sensor', 'rdf:type', 'owl:Class'),
 ('top:Sensor',
  'rdfs:comment',
  '"A sensor, observation point, or measurement device associated with a space, element, system, or analytical '
  'process."@en'),
 ('top:Sensor', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Sensor', 'rdfs:label', '"Sensor"@en'),
 ('top:Sensor', 'rdfs:subClassOf', 'brick:Point'),
 ('top:Sensor', 'rdfs:subClassOf', 'top:Element'),
 ('top:Shell', 'rdf:type', 'owl:Class'),
 ('top:Shell',
  'rdfs:comment',
  '"A two-dimensional TopologicPy topology composed of connected faces, typically forming an open or closed segmented '
  'surface."@en'),
 ('top:Shell', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Shell', 'rdfs:label', '"Shell"@en'),
 ('top:Shell', 'rdfs:subClassOf', 'top:Topology'),
 ('top:Site', 'rdf:type', 'owl:Class'),
 ('top:Site',
  'rdfs:comment',
  '"A site-level spatial container used to represent land, context, parcels, infrastructure extents, or an IFC '
  'site."@en'),
 ('top:Site', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Site', 'rdfs:label', '"Site"@en'),
 ('top:Site', 'rdfs:subClassOf', 'bot:Site'),
 ('top:Site', 'rdfs:subClassOf', 'top:Zone'),
 ('top:Site', 'skos:closeMatch', 'ifc:IfcSite'),
 ('top:Slab', 'rdf:type', 'owl:Class'),
 ('top:Slab', 'rdfs:comment', '"A slab, floor, ceiling, plate, or horizontal building element."@en'),
 ('top:Slab', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Slab', 'rdfs:label', '"Slab"@en'),
 ('top:Slab', 'rdfs:subClassOf', 'bot:Element'),
 ('top:Slab', 'rdfs:subClassOf', 'top:Element'),
 ('top:Slab', 'skos:closeMatch', 'ifc:IfcSlab'),
 ('top:Space', 'rdf:type', 'owl:Class'),
 ('top:Space',
  'rdfs:comment',
  '"A bounded or occupiable spatial region used to represent rooms, spaces, compartments, or IFC spaces."@en'),
 ('top:Space', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Space', 'rdfs:label', '"Space"@en'),
 ('top:Space', 'rdfs:subClassOf', 'bot:Space'),
 ('top:Space', 'rdfs:subClassOf', 'top:Zone'),
 ('top:Space', 'skos:closeMatch', 'ifc:IfcSpace'),
 ('top:SpaceSyntaxMetric', 'rdf:type', 'owl:Class'),
 ('top:SpaceSyntaxMetric',
  'rdfs:comment',
  '"A spatial network metric used in space syntax, visibility analysis, access analysis, or graph-based spatial '
  'analysis."@en'),
 ('top:SpaceSyntaxMetric', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:SpaceSyntaxMetric', 'rdfs:label', '"Space Syntax Metric"@en'),
 ('top:SpaceSyntaxMetric', 'rdfs:subClassOf', 'top:AnalysisMetric'),
 ('top:SpaceSyntaxMetric', 'skos:altLabel', '"SpaceSyntaxMetric"@en'),
 ('top:SpatialGraph', 'rdf:type', 'owl:Class'),
 ('top:SpatialGraph',
  'rdfs:comment',
  '"A directed or undirected graph used to represent spatial entities and spatial relationships. Relationships may '
  'include metric, topological, directional, containment, proximity, overlap, visibility, accessibility, or '
  'connectivity predicates, including relations aligned with OGC Simple Features, ISO 19107 spatial schema concepts, '
  'DE-9IM predicates, or RCC-8 qualitative spatial relations."@en'),
 ('top:SpatialGraph', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:SpatialGraph', 'rdfs:label', '"Spatial Graph"@en'),
 ('top:SpatialGraph', 'rdfs:subClassOf', 'top:Graph'),
 ('top:SpatialGraph', 'skos:altLabel', '"SpatialGraph"@en'),
 ('top:Stair', 'rdf:type', 'owl:Class'),
 ('top:Stair', 'rdfs:comment', '"A stair, stair flight, or vertical circulation element."@en'),
 ('top:Stair', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Stair', 'rdfs:label', '"Stair"@en'),
 ('top:Stair', 'rdfs:subClassOf', 'bot:Element'),
 ('top:Stair', 'rdfs:subClassOf', 'top:Element'),
 ('top:Stair', 'skos:closeMatch', 'ifc:IfcStair'),
 ('top:Storey', 'rdf:type', 'owl:Class'),
 ('top:Storey',
  'rdfs:comment',
  '"A storey-level spatial container used to represent a building level, floor, or IFC building storey."@en'),
 ('top:Storey', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Storey', 'rdfs:label', '"Storey"@en'),
 ('top:Storey', 'rdfs:subClassOf', 'bot:Storey'),
 ('top:Storey', 'rdfs:subClassOf', 'top:Zone'),
 ('top:Storey', 'skos:closeMatch', 'ifc:IfcBuildingStorey'),
 ('top:Surface', 'rdf:type', 'owl:Class'),
 ('top:Surface',
  'rdfs:comment',
  '"A surface-like spatial, analytical, or building-domain entity represented as a specialised face."@en'),
 ('top:Surface', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Surface', 'rdfs:label', '"Surface"@en'),
 ('top:Surface', 'rdfs:subClassOf', 'top:Face'),
 ('top:System', 'rdf:type', 'owl:Class'),
 ('top:System',
  'rdfs:comment',
  '"A building, distribution, MEP, operational, or analytical system that groups elements, equipment, ports, spaces, '
  'or functions."@en'),
 ('top:System', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:System', 'rdfs:label', '"System"@en'),
 ('top:System', 'skos:closeMatch', 'ifc:IfcSystem'),
 ('top:TGraph', 'owl:equivalentClass', 'top:Graph'),
 ('top:TGraph', 'rdf:type', 'owl:Class'),
 ('top:TGraph',
  'rdfs:comment',
  '"A TopologicPy-native graph representation used to represent indexed vertices, indexed edges, graph dictionaries, '
  'optional topology representations, and ontology-aware metadata. This class distinguishes the Python TGraph '
  'implementation from the more general top:Graph concept."@en'),
 ('top:TGraph', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:TGraph', 'rdfs:label', '"T Graph"@en'),
 ('top:TGraph', 'rdfs:subClassOf', 'top:Graph'),
 ('top:TGraph', 'skos:altLabel', '"TGraph"@en'),
 ('top:ThermalZone', 'rdf:type', 'owl:Class'),
 ('top:ThermalZone',
  'rdfs:comment',
  '"A thermal analysis zone represented as a specialised space used in environmental, energy, HVAC, or comfort '
  'analysis."@en'),
 ('top:ThermalZone', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:ThermalZone', 'rdfs:label', '"Thermal Zone"@en'),
 ('top:ThermalZone', 'rdfs:subClassOf', 'bot:Zone'),
 ('top:ThermalZone', 'rdfs:subClassOf', 'top:Space'),
 ('top:ThermalZone', 'skos:altLabel', '"ThermalZone"@en'),
 ('top:Topology', 'rdf:type', 'owl:Class'),
 ('top:Topology',
  'rdfs:comment',
  '"Superclass for TopologicPy topological entities, including vertices, edges, wires, faces, shells, cells, cell '
  'complexes, clusters, apertures, and related topological specialisations."@en'),
 ('top:Topology', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Topology', 'rdfs:label', '"Topology"@en'),
 ('top:TreeGraph', 'rdf:type', 'owl:Class'),
 ('top:TreeGraph',
  'rdfs:comment',
  '"A connected acyclic graph used to represent hierarchical, branching, containment, decomposition, classification, '
  'or dependency structures. In TopologicPy and BIM contexts, it may represent project breakdowns, spatial '
  'decomposition, assembly hierarchies, taxonomy structures, or rooted parent-child relationships."@en'),
 ('top:TreeGraph', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:TreeGraph', 'rdfs:label', '"Tree Graph"@en'),
 ('top:TreeGraph', 'rdfs:subClassOf', 'top:Graph'),
 ('top:TreeGraph', 'skos:altLabel', '"TreeGraph"@en'),
 ('top:UndirectedRelationship', 'rdf:type', 'owl:Class'),
 ('top:UndirectedRelationship',
  'rdfs:comment',
  '"An undirected graph relationship used to represent a symmetric or direction-independent relation between two '
  'nodes."@en'),
 ('top:UndirectedRelationship', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:UndirectedRelationship', 'rdfs:label', '"Undirected Relationship"@en'),
 ('top:UndirectedRelationship', 'rdfs:subClassOf', 'top:Relationship'),
 ('top:UndirectedRelationship', 'skos:altLabel', '"UndirectedRelationship"@en'),
 ('top:ValidationRule', 'rdf:type', 'owl:Class'),
 ('top:ValidationRule',
  'rdfs:comment',
  '"A rule used to validate geometry, topology, semantics, IFC/BIM content, graph structure, provenance, or analytical '
  'metadata."@en'),
 ('top:ValidationRule', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:ValidationRule', 'rdfs:label', '"Validation Rule"@en'),
 ('top:ValidationRule', 'skos:altLabel', '"ValidationRule"@en'),
 ('top:Vector', 'rdf:type', 'owl:Class'),
 ('top:Vector',
  'rdfs:comment',
  '"A mathematical vector used to represent direction, displacement, orientation, or other vector-valued '
  'quantities."@en'),
 ('top:Vector', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Vector', 'rdfs:label', '"Vector"@en'),
 ('top:Vertex', 'rdf:type', 'owl:Class'),
 ('top:Vertex',
  'rdfs:comment',
  '"A zero-dimensional TopologicPy topology representing a point-like entity, commonly defined by X, Y, and Z '
  'coordinates."@en'),
 ('top:Vertex', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Vertex', 'rdfs:label', '"Vertex"@en'),
 ('top:Vertex', 'rdfs:subClassOf', 'top:Topology'),
 ('top:VisibilityGraph', 'rdf:type', 'owl:Class'),
 ('top:VisibilityGraph',
  'rdfs:comment',
  '"An undirected or directed spatial graph used to represent line-of-sight visibility or intervisibility between '
  'points, locations, spaces, surfaces, objects, or sampled positions. Edges indicate that two entities are mutually '
  'visible or directionally visible, and may store visibility distance, angular range, obstruction status, or visual '
  'weight."@en'),
 ('top:VisibilityGraph', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:VisibilityGraph', 'rdfs:label', '"Visibility Graph"@en'),
 ('top:VisibilityGraph', 'rdfs:subClassOf', 'top:SpatialGraph'),
 ('top:VisibilityGraph', 'skos:altLabel', '"VisibilityGraph"@en'),
 ('top:Wall', 'rdf:type', 'owl:Class'),
 ('top:Wall',
  'rdfs:comment',
  '"A wall or wall-like building element used to bound, separate, support, enclose, or subdivide spaces."@en'),
 ('top:Wall', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Wall', 'rdfs:label', '"Wall"@en'),
 ('top:Wall', 'rdfs:subClassOf', 'bot:Element'),
 ('top:Wall', 'rdfs:subClassOf', 'top:Element'),
 ('top:Wall', 'skos:closeMatch', 'ifc:IfcWall'),
 ('top:Window', 'rdf:type', 'owl:Class'),
 ('top:Window',
  'rdfs:comment',
  '"A window or transparent/translucent opening element, commonly associated with daylight, view, envelope, and façade '
  'relationships."@en'),
 ('top:Window', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Window', 'rdfs:label', '"Window"@en'),
 ('top:Window', 'rdfs:subClassOf', 'bot:Element'),
 ('top:Window', 'rdfs:subClassOf', 'top:Element'),
 ('top:Window', 'skos:closeMatch', 'ifc:IfcWindow'),
 ('top:Wire', 'rdf:type', 'owl:Class'),
 ('top:Wire',
  'rdfs:comment',
  '"A one-dimensional TopologicPy topology composed of connected edges, typically forming an open path or a closed '
  'boundary."@en'),
 ('top:Wire', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Wire', 'rdfs:label', '"Wire"@en'),
 ('top:Wire', 'rdfs:subClassOf', 'top:Topology'),
 ('top:Zone', 'rdf:type', 'owl:Class'),
 ('top:Zone',
  'rdfs:comment',
  '"A spatial region, bounded domain, or analytical zone used to organise spaces, buildings, storeys, sites, thermal '
  'zones, or functional regions."@en'),
 ('top:Zone', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:Zone', 'rdfs:label', '"Zone"@en'),
 ('top:Zone', 'rdfs:subClassOf', 'bot:Zone'),
 ('top:Zone', 'rdfs:subClassOf', 'top:Cell'),
 ('top:Zone', 'skos:closeMatch', 'ifc:IfcZone'),
 ('top:adjacentTo', 'rdf:type', 'owl:ObjectProperty'),
 ('top:adjacentTo', 'rdf:type', 'owl:SymmetricProperty'),
 ('top:adjacentTo',
  'rdfs:comment',
  '"Associates two topologies, spaces, regions, or elements that are adjacent according to a declared spatial, '
  'topological, or tolerance-based rule."@en'),
 ('top:adjacentTo', 'rdfs:domain', 'top:Topology'),
 ('top:adjacentTo', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:adjacentTo', 'rdfs:label', '"Adjacent To"@en'),
 ('top:adjacentTo', 'rdfs:range', 'top:Topology'),
 ('top:adjacentTo', 'rdfs:subPropertyOf', 'bot:adjacentTo'),
 ('top:adjacentTo', 'rdfs:subPropertyOf', 'bot:adjacentZone'),
 ('top:adjacentTo', 'skos:altLabel', '"adjacentTo"@en'),
 ('top:aggregates', 'owl:inverseOf', 'top:isAggregatedBy'),
 ('top:aggregates', 'rdf:type', 'owl:ObjectProperty'),
 ('top:aggregates',
  'rdfs:comment',
  '"Represents a whole-part, decomposition, or aggregation relationship, commonly mapped from IFC aggregation or '
  'decomposition relations."@en'),
 ('top:aggregates', 'rdfs:domain', 'top:Topology'),
 ('top:aggregates', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:aggregates', 'rdfs:label', '"Aggregates"@en'),
 ('top:aggregates', 'rdfs:range', 'top:Topology'),
 ('top:aggregates', 'skos:altLabel', '"aggregates"@en'),
 ('top:area', 'owl:equivalentProperty', 'top:hasArea'),
 ('top:area', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:area',
  'rdfs:comment',
  '"The area of a face, shell, cell, cell complex, surface, spatial region, or other area-bearing topology or '
  'analytical record."@en'),
 ('top:area', 'rdfs:domain', 'top:Topology'),
 ('top:area', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:area', 'rdfs:label', '"Area"@en'),
 ('top:area', 'rdfs:range', 'xsd:double'),
 ('top:area', 'skos:altLabel', '"area"@en'),
 ('top:category', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:category',
  'rdfs:comment',
  '"A broad category value emitted from TopologicPy dictionaries, such as topology, graph, space, element, equipment, '
  'interface, project, metadata, mathematics, or analysis."@en'),
 ('top:category', 'rdfs:domain', 'owl:Thing'),
 ('top:category', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:category', 'rdfs:label', '"Category"@en'),
 ('top:category', 'rdfs:range', 'xsd:string'),
 ('top:category', 'skos:altLabel', '"category"@en'),
 ('top:connects', 'rdf:type', 'owl:ObjectProperty'),
 ('top:connects', 'rdf:type', 'owl:SymmetricProperty'),
 ('top:connects',
  'rdfs:comment',
  '"Generic semantic connection used when a relationship is known but more specific semantics are unavailable."@en'),
 ('top:connects', 'rdfs:domain', 'top:Topology'),
 ('top:connects', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:connects', 'rdfs:label', '"Connects"@en'),
 ('top:connects', 'rdfs:range', 'top:Topology'),
 ('top:connects', 'skos:altLabel', '"connects"@en'),
 ('top:connectsPort', 'owl:inverseOf', 'top:isConnectedPortOf'),
 ('top:connectsPort', 'rdf:type', 'owl:ObjectProperty'),
 ('top:connectsPort', 'rdf:type', 'owl:SymmetricProperty'),
 ('top:connectsPort',
  'rdfs:comment',
  '"Connects two ports without asserting flow direction. This is the preferred direct mapping for port-to-port '
  'connectivity such as IfcRelConnectsPorts."@en'),
 ('top:connectsPort', 'rdfs:domain', 'top:Port'),
 ('top:connectsPort', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:connectsPort', 'rdfs:label', '"Connects Port"@en'),
 ('top:connectsPort', 'rdfs:range', 'top:Port'),
 ('top:connectsPort', 'skos:altLabel', '"connectsPort"@en'),
 ('top:connectsTo', 'rdf:type', 'owl:ObjectProperty'),
 ('top:connectsTo', 'rdf:type', 'owl:SymmetricProperty'),
 ('top:connectsTo',
  'rdfs:comment',
  '"Generic undirected topological or graph connectivity between two topologies, vertices, nodes, elements, spaces, or '
  'other entities."@en'),
 ('top:connectsTo', 'rdfs:domain', 'top:Topology'),
 ('top:connectsTo', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:connectsTo', 'rdfs:label', '"Connects To"@en'),
 ('top:connectsTo', 'rdfs:range', 'top:Topology'),
 ('top:connectsTo', 'rdfs:subPropertyOf', 'bot:connectsTo'),
 ('top:connectsTo', 'skos:altLabel', '"connectsTo"@en'),
 ('top:containsElement', 'owl:inverseOf', 'top:isPartOf'),
 ('top:containsElement', 'rdf:type', 'owl:ObjectProperty'),
 ('top:containsElement',
  'rdfs:comment',
  '"Associates a spatial, topological, or semantic container with a contained topology, element, space, or '
  'entity."@en'),
 ('top:containsElement', 'rdfs:domain', 'top:Topology'),
 ('top:containsElement', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:containsElement', 'rdfs:label', '"Contains Element"@en'),
 ('top:containsElement', 'rdfs:range', 'top:Topology'),
 ('top:containsElement', 'rdfs:subPropertyOf', 'bot:containsElement'),
 ('top:containsElement', 'skos:altLabel', '"containsElement"@en'),
 ('top:createdAt', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:createdAt', 'rdfs:comment', '"The creation timestamp of an entity, topology, graph, or record."@en'),
 ('top:createdAt', 'rdfs:domain', 'owl:Thing'),
 ('top:createdAt', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:createdAt', 'rdfs:label', '"Created At"@en'),
 ('top:createdAt', 'rdfs:range', 'xsd:dateTime'),
 ('top:createdAt', 'skos:altLabel', '"createdAt"@en'),
 ('top:derivedFrom', 'rdf:type', 'owl:ObjectProperty'),
 ('top:derivedFrom',
  'rdfs:comment',
  '"Associates an entity, topology, graph, or record with the source entity, model, file, process, or data object from '
  'which it was derived."@en'),
 ('top:derivedFrom', 'rdfs:domain', 'owl:Thing'),
 ('top:derivedFrom', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:derivedFrom', 'rdfs:label', '"Derived From"@en'),
 ('top:derivedFrom', 'rdfs:range', 'owl:Thing'),
 ('top:derivedFrom', 'skos:altLabel', '"derivedFrom"@en'),
 ('top:description', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:description', 'rdfs:comment', '"A human-readable description emitted from a TopologicPy dictionary."@en'),
 ('top:description', 'rdfs:domain', 'owl:Thing'),
 ('top:description', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:description', 'rdfs:label', '"Description"@en'),
 ('top:description', 'rdfs:range', 'xsd:string'),
 ('top:description', 'skos:altLabel', '"description"@en'),
 ('top:dstId', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:dstId',
  'rdfs:comment',
  '"The destination node index of a graph relationship record, retained as structural metadata when exported."@en'),
 ('top:dstId', 'rdfs:domain', 'top:Relationship'),
 ('top:dstId', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:dstId', 'rdfs:label', '"Destination ID"@en'),
 ('top:dstId', 'rdfs:range', 'xsd:integer'),
 ('top:dstId', 'skos:altLabel', '"dstId"@en'),
 ('top:endsAt', 'owl:equivalentProperty', 'top:hasEndVertex'),
 ('top:endsAt', 'rdf:type', 'owl:ObjectProperty'),
 ('top:endsAt',
  'rdfs:comment',
  '"Alias property for associating an edge or relationship with its end vertex or target node."@en'),
 ('top:endsAt', 'rdfs:domain', 'owl:Thing'),
 ('top:endsAt', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:endsAt', 'rdfs:label', '"Ends At"@en'),
 ('top:endsAt', 'rdfs:range', 'top:Vertex'),
 ('top:endsAt', 'skos:altLabel', '"endsAt"@en'),
 ('top:feature', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:feature',
  'rdfs:comment',
  '"A scalar or symbolic feature value associated with an entity, topology, graph, node, or relationship."@en'),
 ('top:feature', 'rdfs:domain', 'owl:Thing'),
 ('top:feature', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:feature', 'rdfs:label', '"Feature"@en'),
 ('top:feature', 'rdfs:range', 'xsd:string'),
 ('top:feature', 'skos:altLabel', '"feature"@en'),
 ('top:featureVector', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:featureVector',
  'rdfs:comment',
  '"A feature vector associated with an entity, topology, graph, node, or relationship."@en'),
 ('top:featureVector', 'rdfs:domain', 'owl:Thing'),
 ('top:featureVector', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:featureVector', 'rdfs:label', '"Feature Vector"@en'),
 ('top:featureVector', 'rdfs:range', 'xsd:string'),
 ('top:featureVector', 'skos:altLabel', '"featureVector"@en'),
 ('top:fillsOpening', 'owl:inverseOf', 'top:isFilledBy'),
 ('top:fillsOpening', 'rdf:type', 'owl:ObjectProperty'),
 ('top:fillsOpening',
  'rdfs:comment',
  '"Associates an element such as a door, window, or service component with the opening it fills."@en'),
 ('top:fillsOpening', 'rdfs:domain', 'top:Element'),
 ('top:fillsOpening', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:fillsOpening', 'rdfs:label', '"Fills Opening"@en'),
 ('top:fillsOpening', 'rdfs:range', 'top:Opening'),
 ('top:fillsOpening', 'skos:altLabel', '"fillsOpening"@en'),
 ('top:generatedBy', 'rdf:type', 'owl:ObjectProperty'),
 ('top:generatedBy',
  'rdfs:comment',
  '"Associates an entity, topology, graph, or record with the method, script, notebook, process, or software operation '
  'that generated it."@en'),
 ('top:generatedBy', 'rdfs:domain', 'owl:Thing'),
 ('top:generatedBy', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:generatedBy', 'rdfs:label', '"Generated By"@en'),
 ('top:generatedBy', 'rdfs:range', 'owl:Thing'),
 ('top:generatedBy', 'skos:altLabel', '"generatedBy"@en'),
 ('top:generatedByMethod', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:generatedByMethod',
  'rdfs:comment',
  '"The method, function, script, notebook, or software operation that generated an entity, topology, graph, or '
  'record, stored as a literal value."@en'),
 ('top:generatedByMethod', 'rdfs:domain', 'owl:Thing'),
 ('top:generatedByMethod', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:generatedByMethod', 'rdfs:label', '"Generated By Method"@en'),
 ('top:generatedByMethod', 'rdfs:range', 'xsd:string'),
 ('top:generatedByMethod', 'skos:altLabel', '"generatedByMethod"@en'),
 ('top:hasApproval', 'owl:inverseOf', 'top:isApprovalOf'),
 ('top:hasApproval', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasApproval',
  'rdfs:comment',
  '"Associates an entity with an approval, review, authorisation, or sign-off record."@en'),
 ('top:hasApproval', 'rdfs:domain', 'owl:Thing'),
 ('top:hasApproval', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasApproval', 'rdfs:label', '"Has Approval"@en'),
 ('top:hasApproval', 'rdfs:range', 'owl:Thing'),
 ('top:hasApproval', 'skos:altLabel', '"hasApproval"@en'),
 ('top:hasArea', 'owl:deprecated', '"true"^^xsd:boolean'),
 ('top:hasArea', 'owl:equivalentProperty', 'top:area'),
 ('top:hasArea', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:hasArea',
  'rdfs:comment',
  '"The area of a face, shell, cell, cell complex, surface, spatial region, or other area-bearing topology or '
  'analytical record."@en'),
 ('top:hasArea', 'rdfs:domain', 'top:Topology'),
 ('top:hasArea', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasArea', 'rdfs:label', '"Has Area"@en'),
 ('top:hasArea', 'rdfs:range', 'xsd:double'),
 ('top:hasArea', 'skos:altLabel', '"hasArea"@en'),
 ('top:hasCell', 'owl:inverseOf', 'top:isCellOf'),
 ('top:hasCell', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasCell', 'rdfs:comment', '"Associates a topology with a constituent cell."@en'),
 ('top:hasCell', 'rdfs:domain', 'top:Topology'),
 ('top:hasCell', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasCell', 'rdfs:label', '"Has Cell"@en'),
 ('top:hasCell', 'rdfs:range', 'top:Cell'),
 ('top:hasCell', 'rdfs:subPropertyOf', 'top:hasSubTopology'),
 ('top:hasCell', 'skos:altLabel', '"hasCell"@en'),
 ('top:hasCellComplex', 'owl:inverseOf', 'top:isCellComplexOf'),
 ('top:hasCellComplex', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasCellComplex',
  'rdfs:comment',
  '"Associates a cluster or model container with a constituent cell complex."@en'),
 ('top:hasCellComplex', 'rdfs:domain', 'top:Cluster'),
 ('top:hasCellComplex', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasCellComplex', 'rdfs:label', '"Has Cell Complex"@en'),
 ('top:hasCellComplex', 'rdfs:range', 'top:CellComplex'),
 ('top:hasCellComplex', 'rdfs:subPropertyOf', 'top:hasSubTopology'),
 ('top:hasCellComplex', 'skos:altLabel', '"hasCellComplex"@en'),
 ('top:hasClassification', 'owl:inverseOf', 'top:isClassificationOf'),
 ('top:hasClassification', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasClassification',
  'rdfs:comment',
  '"Associates a topology, element, system, or mapped BIM entity with a classification reference."@en'),
 ('top:hasClassification', 'rdfs:domain', 'top:Topology'),
 ('top:hasClassification', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasClassification', 'rdfs:label', '"Has Classification"@en'),
 ('top:hasClassification', 'rdfs:range', 'top:ClassificationReference'),
 ('top:hasClassification', 'skos:altLabel', '"hasClassification"@en'),
 ('top:hasConnectedPort', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasConnectedPort',
  'rdfs:comment',
  '"Associates an element, system component, or equipment item with a connected distribution or connection port."@en'),
 ('top:hasConnectedPort', 'rdfs:domain', 'top:Element'),
 ('top:hasConnectedPort', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasConnectedPort', 'rdfs:label', '"Has Connected Port"@en'),
 ('top:hasConnectedPort', 'rdfs:range', 'top:Port'),
 ('top:hasConnectedPort', 'skos:altLabel', '"hasConnectedPort"@en'),
 ('top:hasConstraint', 'owl:inverseOf', 'top:isConstraintOf'),
 ('top:hasConstraint', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasConstraint',
  'rdfs:comment',
  '"Associates an entity with a rule, constraint, requirement, limit, or validation condition."@en'),
 ('top:hasConstraint', 'rdfs:domain', 'owl:Thing'),
 ('top:hasConstraint', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasConstraint', 'rdfs:label', '"Has Constraint"@en'),
 ('top:hasConstraint', 'rdfs:range', 'owl:Thing'),
 ('top:hasConstraint', 'skos:altLabel', '"hasConstraint"@en'),
 ('top:hasCoordinationIssue', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasCoordinationIssue',
  'rdfs:comment',
  '"Associates an entity with a detected coordination, clash, validation, or quality issue."@en'),
 ('top:hasCoordinationIssue', 'rdfs:domain', 'top:Topology'),
 ('top:hasCoordinationIssue', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasCoordinationIssue', 'rdfs:label', '"Has Coordination Issue"@en'),
 ('top:hasCoordinationIssue', 'rdfs:range', 'top:Relationship'),
 ('top:hasCoordinationIssue', 'skos:altLabel', '"hasCoordinationIssue"@en'),
 ('top:hasDictionary', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasDictionary',
  'rdfs:comment',
  '"Associates a topology, graph, node, relationship, or record with a TopologicPy dictionary containing metadata, '
  'attributes, semantics, analysis values, or provenance."@en'),
 ('top:hasDictionary', 'rdfs:domain', 'owl:Thing'),
 ('top:hasDictionary', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasDictionary', 'rdfs:label', '"Has Dictionary"@en'),
 ('top:hasDictionary', 'rdfs:range', 'top:Dictionary'),
 ('top:hasDictionary', 'skos:altLabel', '"hasDictionary"@en'),
 ('top:hasDocument', 'owl:inverseOf', 'top:isDocumentOf'),
 ('top:hasDocument', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasDocument',
  'rdfs:comment',
  '"Associates an entity with a document reference, external file, specification, drawing, approval package, or '
  'supporting document."@en'),
 ('top:hasDocument', 'rdfs:domain', 'owl:Thing'),
 ('top:hasDocument', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasDocument', 'rdfs:label', '"Has Document"@en'),
 ('top:hasDocument', 'rdfs:range', 'owl:Thing'),
 ('top:hasDocument', 'skos:altLabel', '"hasDocument"@en'),
 ('top:hasEdge', 'owl:inverseOf', 'top:isEdgeOf'),
 ('top:hasEdge', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasEdge', 'rdfs:comment', '"Associates a topology or graph with an edge that belongs to it."@en'),
 ('top:hasEdge', 'rdfs:domain', 'owl:Thing'),
 ('top:hasEdge', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasEdge', 'rdfs:label', '"Has Edge"@en'),
 ('top:hasEdge', 'rdfs:range', 'top:Edge'),
 ('top:hasEdge', 'rdfs:subPropertyOf', 'top:hasSubTopology'),
 ('top:hasEdge', 'skos:altLabel', '"hasEdge"@en'),
 ('top:hasEndVertex', 'owl:deprecated', '"true"^^xsd:boolean'),
 ('top:hasEndVertex', 'owl:equivalentProperty', 'top:endsAt'),
 ('top:hasEndVertex', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasEndVertex', 'rdfs:comment', '"Associates an edge or relationship with its end vertex or target node."@en'),
 ('top:hasEndVertex', 'rdfs:domain', 'owl:Thing'),
 ('top:hasEndVertex', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasEndVertex', 'rdfs:label', '"Has End Vertex"@en'),
 ('top:hasEndVertex', 'rdfs:range', 'top:Vertex'),
 ('top:hasEndVertex', 'skos:altLabel', '"hasEndVertex"@en'),
 ('top:hasExternalBoundary', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasExternalBoundary',
  'rdfs:comment',
  '"Associates a topology, region, element, or analytical domain with its external boundary."@en'),
 ('top:hasExternalBoundary', 'rdfs:domain', 'top:Topology'),
 ('top:hasExternalBoundary', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasExternalBoundary', 'rdfs:label', '"Has External Boundary"@en'),
 ('top:hasExternalBoundary', 'rdfs:range', 'top:Boundary'),
 ('top:hasExternalBoundary', 'skos:altLabel', '"hasExternalBoundary"@en'),
 ('top:hasFace', 'owl:inverseOf', 'top:isFaceOf'),
 ('top:hasFace', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasFace', 'rdfs:comment', '"Associates a topology with a constituent face."@en'),
 ('top:hasFace', 'rdfs:domain', 'top:Topology'),
 ('top:hasFace', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasFace', 'rdfs:label', '"Has Face"@en'),
 ('top:hasFace', 'rdfs:range', 'top:Face'),
 ('top:hasFace', 'rdfs:subPropertyOf', 'top:hasSubTopology'),
 ('top:hasFace', 'skos:altLabel', '"hasFace"@en'),
 ('top:hasFeature', 'owl:deprecated', '"true"^^xsd:boolean'),
 ('top:hasFeature', 'owl:equivalentProperty', 'top:feature'),
 ('top:hasFeature', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:hasFeature', 'rdfs:comment', '"Deprecated alias of top:feature."@en'),
 ('top:hasFeature', 'rdfs:domain', 'owl:Thing'),
 ('top:hasFeature', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasFeature', 'rdfs:label', '"Has Feature"@en'),
 ('top:hasFeature', 'rdfs:range', 'xsd:string'),
 ('top:hasFeature', 'skos:altLabel', '"hasFeature"@en'),
 ('top:hasFeatureVector', 'owl:deprecated', '"true"^^xsd:boolean'),
 ('top:hasFeatureVector', 'owl:equivalentProperty', 'top:featureVector'),
 ('top:hasFeatureVector', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:hasFeatureVector', 'rdfs:comment', '"Deprecated alias of top:featureVector."@en'),
 ('top:hasFeatureVector', 'rdfs:domain', 'owl:Thing'),
 ('top:hasFeatureVector', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasFeatureVector', 'rdfs:label', '"Has Feature Vector"@en'),
 ('top:hasFeatureVector', 'rdfs:range', 'xsd:string'),
 ('top:hasFeatureVector', 'skos:altLabel', '"hasFeatureVector"@en'),
 ('top:hasIFCType', 'owl:inverseOf', 'top:isIFCTypeOf'),
 ('top:hasIFCType', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasIFCType', 'rdfs:comment', '"Associates an IFC occurrence or mapped topology with its IFC type object."@en'),
 ('top:hasIFCType', 'rdfs:domain', 'top:Topology'),
 ('top:hasIFCType', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasIFCType', 'rdfs:label', '"Has IFC Type"@en'),
 ('top:hasIFCType', 'rdfs:range', 'owl:Thing'),
 ('top:hasIFCType', 'skos:altLabel', '"hasIFCType"@en'),
 ('top:hasInternalBoundary', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasInternalBoundary',
  'rdfs:comment',
  '"Associates a topology, region, element, or analytical domain with an internal boundary, hole, or void '
  'boundary."@en'),
 ('top:hasInternalBoundary', 'rdfs:domain', 'top:Topology'),
 ('top:hasInternalBoundary', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasInternalBoundary', 'rdfs:label', '"Has Internal Boundary"@en'),
 ('top:hasInternalBoundary', 'rdfs:range', 'top:Boundary'),
 ('top:hasInternalBoundary', 'skos:altLabel', '"hasInternalBoundary"@en'),
 ('top:hasLength', 'owl:deprecated', '"true"^^xsd:boolean'),
 ('top:hasLength', 'owl:equivalentProperty', 'top:length'),
 ('top:hasLength', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:hasLength',
  'rdfs:comment',
  '"The length of an edge, wire, path, graph edge, or other length-bearing topology or analytical record."@en'),
 ('top:hasLength', 'rdfs:domain', 'top:Topology'),
 ('top:hasLength', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasLength', 'rdfs:label', '"Has Length"@en'),
 ('top:hasLength', 'rdfs:range', 'xsd:double'),
 ('top:hasLength', 'skos:altLabel', '"hasLength"@en'),
 ('top:hasMantissa', 'owl:deprecated', '"true"^^xsd:boolean'),
 ('top:hasMantissa', 'owl:equivalentProperty', 'top:mantissa'),
 ('top:hasMantissa', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:hasMantissa',
  'rdfs:comment',
  '"The number of decimal places used to round, serialize, compare, or report numeric values."@en'),
 ('top:hasMantissa', 'rdfs:domain', 'owl:Thing'),
 ('top:hasMantissa', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasMantissa', 'rdfs:label', '"Has Mantissa"@en'),
 ('top:hasMantissa', 'rdfs:range', 'xsd:integer'),
 ('top:hasMantissa', 'skos:altLabel', '"hasMantissa"@en'),
 ('top:hasMaterial', 'owl:inverseOf', 'top:isMaterialOf'),
 ('top:hasMaterial', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasMaterial',
  'rdfs:comment',
  '"Associates a topology, element, or mapped BIM entity with a material or material set."@en'),
 ('top:hasMaterial', 'rdfs:domain', 'top:Topology'),
 ('top:hasMaterial', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasMaterial', 'rdfs:label', '"Has Material"@en'),
 ('top:hasMaterial', 'rdfs:range', 'owl:Thing'),
 ('top:hasMaterial', 'skos:altLabel', '"hasMaterial"@en'),
 ('top:hasMissingOpening', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasMissingOpening',
  'rdfs:comment',
  '"Associates an element or topology with a coordination issue in which an expected opening is absent."@en'),
 ('top:hasMissingOpening', 'rdfs:domain', 'top:Topology'),
 ('top:hasMissingOpening', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasMissingOpening', 'rdfs:label', '"Has Missing Opening"@en'),
 ('top:hasMissingOpening', 'rdfs:range', 'top:Element'),
 ('top:hasMissingOpening', 'skos:altLabel', '"hasMissingOpening"@en'),
 ('top:hasNode', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasNode', 'rdfs:comment', '"Associates a graph with a node that belongs to it."@en'),
 ('top:hasNode', 'rdfs:domain', 'top:Graph'),
 ('top:hasNode', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasNode', 'rdfs:label', '"Has Node"@en'),
 ('top:hasNode', 'rdfs:range', 'top:Node'),
 ('top:hasNode', 'skos:altLabel', '"hasNode"@en'),
 ('top:hasOpening', 'owl:inverseOf', 'top:isOpeningIn'),
 ('top:hasOpening', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasOpening', 'rdfs:comment', '"Associates an element with an opening, void, penetration, or recess."@en'),
 ('top:hasOpening', 'rdfs:domain', 'top:Element'),
 ('top:hasOpening', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasOpening', 'rdfs:label', '"Has Opening"@en'),
 ('top:hasOpening', 'rdfs:range', 'top:Opening'),
 ('top:hasOpening', 'skos:altLabel', '"hasOpening"@en'),
 ('top:hasPredicate', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasPredicate',
  'rdfs:comment',
  '"Associates a TopologicPy relationship record with the RDF predicate that gives the relationship its semantic '
  'meaning."@en'),
 ('top:hasPredicate', 'rdfs:domain', 'top:Relationship'),
 ('top:hasPredicate', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasPredicate', 'rdfs:label', '"Has Predicate"@en'),
 ('top:hasPredicate', 'rdfs:range', 'rdf:Property'),
 ('top:hasPredicate', 'skos:altLabel', '"hasPredicate"@en'),
 ('top:hasPropertySet', 'owl:inverseOf', 'top:isPropertySetOf'),
 ('top:hasPropertySet', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasPropertySet',
  'rdfs:comment',
  '"Associates a topology, element, type, system, graph entity, or relationship with a property set."@en'),
 ('top:hasPropertySet', 'rdfs:domain', 'owl:Thing'),
 ('top:hasPropertySet', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasPropertySet', 'rdfs:label', '"Has Property Set"@en'),
 ('top:hasPropertySet', 'rdfs:range', 'top:PropertySet'),
 ('top:hasPropertySet', 'skos:altLabel', '"hasPropertySet"@en'),
 ('top:hasRelationship', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasRelationship', 'rdfs:comment', '"Associates a graph with a relationship or edge that belongs to it."@en'),
 ('top:hasRelationship', 'rdfs:domain', 'top:Graph'),
 ('top:hasRelationship', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasRelationship', 'rdfs:label', '"Has Relationship"@en'),
 ('top:hasRelationship', 'rdfs:range', 'top:Relationship'),
 ('top:hasRelationship', 'skos:altLabel', '"hasRelationship"@en'),
 ('top:hasShell', 'owl:inverseOf', 'top:isShellOf'),
 ('top:hasShell', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasShell', 'rdfs:comment', '"Associates a topology with a constituent shell."@en'),
 ('top:hasShell', 'rdfs:domain', 'top:Topology'),
 ('top:hasShell', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasShell', 'rdfs:label', '"Has Shell"@en'),
 ('top:hasShell', 'rdfs:range', 'top:Shell'),
 ('top:hasShell', 'rdfs:subPropertyOf', 'top:hasSubTopology'),
 ('top:hasShell', 'skos:altLabel', '"hasShell"@en'),
 ('top:hasStartVertex', 'owl:deprecated', '"true"^^xsd:boolean'),
 ('top:hasStartVertex', 'owl:equivalentProperty', 'top:startsAt'),
 ('top:hasStartVertex', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasStartVertex',
  'rdfs:comment',
  '"Associates an edge or relationship with its start vertex or source node."@en'),
 ('top:hasStartVertex', 'rdfs:domain', 'owl:Thing'),
 ('top:hasStartVertex', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasStartVertex', 'rdfs:label', '"Has Start Vertex"@en'),
 ('top:hasStartVertex', 'rdfs:range', 'top:Vertex'),
 ('top:hasStartVertex', 'skos:altLabel', '"hasStartVertex"@en'),
 ('top:hasSubTopology', 'owl:inverseOf', 'top:isSubTopologyOf'),
 ('top:hasSubTopology', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasSubTopology', 'rdfs:comment', '"Associates a topology with a contained or constituent subtopology."@en'),
 ('top:hasSubTopology', 'rdfs:domain', 'top:Topology'),
 ('top:hasSubTopology', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasSubTopology', 'rdfs:label', '"Has Sub Topology"@en'),
 ('top:hasSubTopology', 'rdfs:range', 'top:Topology'),
 ('top:hasSubTopology', 'skos:altLabel', '"hasSubTopology"@en'),
 ('top:hasTopology', 'owl:inverseOf', 'top:isTopologyOf'),
 ('top:hasTopology', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasTopology',
  'rdfs:comment',
  '"Associates an entity with a topology that geometrically or topologically represents it."@en'),
 ('top:hasTopology', 'rdfs:domain', 'owl:Thing'),
 ('top:hasTopology', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasTopology', 'rdfs:label', '"Has Topology"@en'),
 ('top:hasTopology', 'rdfs:range', 'top:Topology'),
 ('top:hasTopology', 'skos:altLabel', '"hasTopology"@en'),
 ('top:hasUnit', 'owl:deprecated', '"true"^^xsd:boolean'),
 ('top:hasUnit', 'owl:equivalentProperty', 'top:unit'),
 ('top:hasUnit', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:hasUnit',
  'rdfs:comment',
  '"The unit of measurement associated with a value, topology, graph, metric, or record."@en'),
 ('top:hasUnit', 'rdfs:domain', 'owl:Thing'),
 ('top:hasUnit', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasUnit', 'rdfs:label', '"Has Unit"@en'),
 ('top:hasUnit', 'rdfs:range', 'xsd:string'),
 ('top:hasUnit', 'skos:altLabel', '"hasUnit"@en'),
 ('top:hasVertex', 'owl:inverseOf', 'top:isVertexOf'),
 ('top:hasVertex', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasVertex', 'rdfs:comment', '"Associates a topology or graph with a vertex that belongs to it."@en'),
 ('top:hasVertex', 'rdfs:domain', 'owl:Thing'),
 ('top:hasVertex', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasVertex', 'rdfs:label', '"Has Vertex"@en'),
 ('top:hasVertex', 'rdfs:range', 'top:Vertex'),
 ('top:hasVertex', 'rdfs:subPropertyOf', 'top:hasSubTopology'),
 ('top:hasVertex', 'skos:altLabel', '"hasVertex"@en'),
 ('top:hasVolume', 'owl:deprecated', '"true"^^xsd:boolean'),
 ('top:hasVolume', 'owl:equivalentProperty', 'top:volume'),
 ('top:hasVolume', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:hasVolume',
  'rdfs:comment',
  '"The volume of a cell, cell complex, zone, space, or other volume-bearing topology or analytical record."@en'),
 ('top:hasVolume', 'rdfs:domain', 'top:Topology'),
 ('top:hasVolume', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasVolume', 'rdfs:label', '"Has Volume"@en'),
 ('top:hasVolume', 'rdfs:range', 'xsd:double'),
 ('top:hasVolume', 'skos:altLabel', '"hasVolume"@en'),
 ('top:hasWeight', 'owl:deprecated', '"true"^^xsd:boolean'),
 ('top:hasWeight', 'owl:equivalentProperty', 'top:weight'),
 ('top:hasWeight', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:hasWeight', 'rdfs:comment', '"Deprecated alias of top:weight."@en'),
 ('top:hasWeight', 'rdfs:domain', 'owl:Thing'),
 ('top:hasWeight', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasWeight', 'rdfs:label', '"Has Weight"@en'),
 ('top:hasWeight', 'rdfs:range', 'xsd:double'),
 ('top:hasWeight', 'skos:altLabel', '"hasWeight"@en'),
 ('top:hasWire', 'owl:inverseOf', 'top:isWireOf'),
 ('top:hasWire', 'rdf:type', 'owl:ObjectProperty'),
 ('top:hasWire', 'rdfs:comment', '"Associates a topology with a constituent wire."@en'),
 ('top:hasWire', 'rdfs:domain', 'top:Topology'),
 ('top:hasWire', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasWire', 'rdfs:label', '"Has Wire"@en'),
 ('top:hasWire', 'rdfs:range', 'top:Wire'),
 ('top:hasWire', 'rdfs:subPropertyOf', 'top:hasSubTopology'),
 ('top:hasWire', 'skos:altLabel', '"hasWire"@en'),
 ('top:hasX', 'owl:deprecated', '"true"^^xsd:boolean'),
 ('top:hasX', 'owl:equivalentProperty', 'top:x'),
 ('top:hasX', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:hasX', 'rdfs:comment', '"The X coordinate of a vertex, point, node, or graph vertex record."@en'),
 ('top:hasX', 'rdfs:domain', 'top:Vertex'),
 ('top:hasX', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasX', 'rdfs:label', '"Has X"@en'),
 ('top:hasX', 'rdfs:range', 'xsd:double'),
 ('top:hasX', 'skos:altLabel', '"hasX"@en'),
 ('top:hasY', 'owl:deprecated', '"true"^^xsd:boolean'),
 ('top:hasY', 'owl:equivalentProperty', 'top:y'),
 ('top:hasY', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:hasY', 'rdfs:comment', '"The Y coordinate of a vertex, point, node, or graph vertex record."@en'),
 ('top:hasY', 'rdfs:domain', 'top:Vertex'),
 ('top:hasY', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasY', 'rdfs:label', '"Has Y"@en'),
 ('top:hasY', 'rdfs:range', 'xsd:double'),
 ('top:hasY', 'skos:altLabel', '"hasY"@en'),
 ('top:hasZ', 'owl:deprecated', '"true"^^xsd:boolean'),
 ('top:hasZ', 'owl:equivalentProperty', 'top:z'),
 ('top:hasZ', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:hasZ', 'rdfs:comment', '"The Z coordinate of a vertex, point, node, or graph vertex record."@en'),
 ('top:hasZ', 'rdfs:domain', 'top:Vertex'),
 ('top:hasZ', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:hasZ', 'rdfs:label', '"Has Z"@en'),
 ('top:hasZ', 'rdfs:range', 'xsd:double'),
 ('top:hasZ', 'skos:altLabel', '"hasZ"@en'),
 ('top:ifcClass', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:ifcClass',
  'rdfs:comment',
  '"The IFC entity class name associated with a topology, graph entity, or record, commonly stored under the '
  'dictionary key ifc_class."@en'),
 ('top:ifcClass', 'rdfs:domain', 'owl:Thing'),
 ('top:ifcClass', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:ifcClass', 'rdfs:label', '"IFC Class"@en'),
 ('top:ifcClass', 'rdfs:range', 'xsd:string'),
 ('top:ifcClass', 'skos:altLabel', '"ifcClass"@en'),
 ('top:ifcGUID', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:ifcGUID',
  'rdfs:comment',
  '"The IFC GlobalId associated with a topology, graph entity, or record, commonly stored under the dictionary key '
  'ifc_guid."@en'),
 ('top:ifcGUID', 'rdfs:domain', 'owl:Thing'),
 ('top:ifcGUID', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:ifcGUID', 'rdfs:label', '"IFC GUID"@en'),
 ('top:ifcGUID', 'rdfs:range', 'xsd:string'),
 ('top:ifcGUID', 'skos:altLabel', '"ifcGUID"@en'),
 ('top:ifcName', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:ifcName', 'rdfs:comment', '"The IFC Name associated with a topology, graph entity, or record."@en'),
 ('top:ifcName', 'rdfs:domain', 'owl:Thing'),
 ('top:ifcName', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:ifcName', 'rdfs:label', '"IFC Name"@en'),
 ('top:ifcName', 'rdfs:range', 'xsd:string'),
 ('top:ifcName', 'skos:altLabel', '"ifcName"@en'),
 ('top:ifcStepId', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:ifcStepId', 'rdfs:comment', '"The file-local STEP entity id associated with an IFC entity."@en'),
 ('top:ifcStepId', 'rdfs:domain', 'owl:Thing'),
 ('top:ifcStepId', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:ifcStepId', 'rdfs:label', '"IFC STEP ID"@en'),
 ('top:ifcStepId', 'rdfs:range', 'xsd:integer'),
 ('top:ifcStepId', 'skos:altLabel', '"ifcStepId"@en'),
 ('top:ifcStepKey', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:ifcStepKey',
  'rdfs:comment',
  '"The file-local STEP entity key, such as #123, associated with an IFC entity."@en'),
 ('top:ifcStepKey', 'rdfs:domain', 'owl:Thing'),
 ('top:ifcStepKey', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:ifcStepKey', 'rdfs:label', '"IFC STEP Key"@en'),
 ('top:ifcStepKey', 'rdfs:range', 'xsd:string'),
 ('top:ifcStepKey', 'skos:altLabel', '"ifcStepKey"@en'),
 ('top:ifcType', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:ifcType',
  'rdfs:comment',
  '"The IFC entity type or declared type associated with a topology, graph entity, or record."@en'),
 ('top:ifcType', 'rdfs:domain', 'owl:Thing'),
 ('top:ifcType', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:ifcType', 'rdfs:label', '"IFC Type"@en'),
 ('top:ifcType', 'rdfs:range', 'xsd:string'),
 ('top:ifcType', 'skos:altLabel', '"ifcType"@en'),
 ('top:index', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:index',
  'rdfs:comment',
  '"A file-local, graph-local, or collection-local ordinal index used for deterministic ordering or '
  'reconstruction."@en'),
 ('top:index', 'rdfs:domain', 'owl:Thing'),
 ('top:index', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:index', 'rdfs:label', '"Index"@en'),
 ('top:index', 'rdfs:range', 'xsd:integer'),
 ('top:index', 'skos:altLabel', '"index"@en'),
 ('top:interfaceOf', 'rdf:type', 'owl:ObjectProperty'),
 ('top:interfaceOf',
  'rdfs:comment',
  '"Associates an interface with the topology, element, space, or zone that it bounds, separates, or connects."@en'),
 ('top:interfaceOf', 'rdfs:domain', 'top:Interface'),
 ('top:interfaceOf', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:interfaceOf', 'rdfs:label', '"Interface Of"@en'),
 ('top:interfaceOf', 'rdfs:range', 'top:Topology'),
 ('top:interfaceOf', 'rdfs:subPropertyOf', 'bot:interfaceOf'),
 ('top:interfaceOf', 'skos:altLabel', '"interfaceOf"@en'),
 ('top:intersects', 'rdf:type', 'owl:ObjectProperty'),
 ('top:intersects', 'rdf:type', 'owl:SymmetricProperty'),
 ('top:intersects',
  'rdfs:comment',
  '"Associates two topologies, elements, spaces, or regions that geometrically or topologically intersect according to '
  'a declared tolerance or spatial predicate."@en'),
 ('top:intersects', 'rdfs:domain', 'top:Topology'),
 ('top:intersects', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:intersects', 'rdfs:label', '"Intersects"@en'),
 ('top:intersects', 'rdfs:range', 'top:Topology'),
 ('top:intersects', 'skos:altLabel', '"intersects"@en'),
 ('top:isAggregatedBy', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isAggregatedBy',
  'rdfs:comment',
  '"Inverse relation of top:aggregates, associating a part with its aggregate or whole."@en'),
 ('top:isAggregatedBy', 'rdfs:domain', 'top:Topology'),
 ('top:isAggregatedBy', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isAggregatedBy', 'rdfs:label', '"Is Aggregated By"@en'),
 ('top:isAggregatedBy', 'rdfs:range', 'top:Topology'),
 ('top:isAggregatedBy', 'skos:altLabel', '"isAggregatedBy"@en'),
 ('top:isApprovalOf', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isApprovalOf', 'rdfs:comment', '"Inverse relation of top:hasApproval."@en'),
 ('top:isApprovalOf', 'rdfs:domain', 'owl:Thing'),
 ('top:isApprovalOf', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isApprovalOf', 'rdfs:label', '"Is Approval Of"@en'),
 ('top:isApprovalOf', 'rdfs:range', 'owl:Thing'),
 ('top:isApprovalOf', 'skos:altLabel', '"isApprovalOf"@en'),
 ('top:isCellComplexOf', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isCellComplexOf',
  'rdfs:comment',
  '"Associates a cell complex with a containing cluster or model container."@en'),
 ('top:isCellComplexOf', 'rdfs:domain', 'top:CellComplex'),
 ('top:isCellComplexOf', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isCellComplexOf', 'rdfs:label', '"Is Cell Complex Of"@en'),
 ('top:isCellComplexOf', 'rdfs:range', 'top:Cluster'),
 ('top:isCellComplexOf', 'rdfs:subPropertyOf', 'top:isSubTopologyOf'),
 ('top:isCellComplexOf', 'skos:altLabel', '"isCellComplexOf"@en'),
 ('top:isCellOf', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isCellOf', 'rdfs:comment', '"Associates a cell with its parent topology."@en'),
 ('top:isCellOf', 'rdfs:domain', 'top:Cell'),
 ('top:isCellOf', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isCellOf', 'rdfs:label', '"Is Cell Of"@en'),
 ('top:isCellOf', 'rdfs:range', 'top:Topology'),
 ('top:isCellOf', 'rdfs:subPropertyOf', 'top:isSubTopologyOf'),
 ('top:isCellOf', 'skos:altLabel', '"isCellOf"@en'),
 ('top:isClassificationOf', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isClassificationOf', 'rdfs:comment', '"Inverse relation of top:hasClassification."@en'),
 ('top:isClassificationOf', 'rdfs:domain', 'top:ClassificationReference'),
 ('top:isClassificationOf', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isClassificationOf', 'rdfs:label', '"Is Classification Of"@en'),
 ('top:isClassificationOf', 'rdfs:range', 'top:Topology'),
 ('top:isClassificationOf', 'skos:altLabel', '"isClassificationOf"@en'),
 ('top:isConnectedPortOf', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isConnectedPortOf',
  'rdfs:comment',
  '"Inverse or companion relation for top:connectsPort where a directional statement is required by an export '
  'process."@en'),
 ('top:isConnectedPortOf', 'rdfs:domain', 'top:Port'),
 ('top:isConnectedPortOf', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isConnectedPortOf', 'rdfs:label', '"Is Connected Port Of"@en'),
 ('top:isConnectedPortOf', 'rdfs:range', 'top:Port'),
 ('top:isConnectedPortOf', 'skos:altLabel', '"isConnectedPortOf"@en'),
 ('top:isConnectedTo', 'owl:equivalentProperty', 'top:connects'),
 ('top:isConnectedTo', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isConnectedTo', 'rdf:type', 'owl:SymmetricProperty'),
 ('top:isConnectedTo', 'rdfs:comment', '"Alias property for generic semantic or topological connection."@en'),
 ('top:isConnectedTo', 'rdfs:domain', 'top:Topology'),
 ('top:isConnectedTo', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isConnectedTo', 'rdfs:label', '"Is Connected To"@en'),
 ('top:isConnectedTo', 'rdfs:range', 'top:Topology'),
 ('top:isConnectedTo', 'skos:altLabel', '"isConnectedTo"@en'),
 ('top:isConstraintOf', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isConstraintOf', 'rdfs:comment', '"Inverse relation of top:hasConstraint."@en'),
 ('top:isConstraintOf', 'rdfs:domain', 'owl:Thing'),
 ('top:isConstraintOf', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isConstraintOf', 'rdfs:label', '"Is Constraint Of"@en'),
 ('top:isConstraintOf', 'rdfs:range', 'owl:Thing'),
 ('top:isConstraintOf', 'skos:altLabel', '"isConstraintOf"@en'),
 ('top:isDocumentOf', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isDocumentOf', 'rdfs:comment', '"Inverse relation of top:hasDocument."@en'),
 ('top:isDocumentOf', 'rdfs:domain', 'owl:Thing'),
 ('top:isDocumentOf', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isDocumentOf', 'rdfs:label', '"Is Document Of"@en'),
 ('top:isDocumentOf', 'rdfs:range', 'owl:Thing'),
 ('top:isDocumentOf', 'skos:altLabel', '"isDocumentOf"@en'),
 ('top:isEdgeOf', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isEdgeOf', 'rdfs:comment', '"Associates an edge with the topology or graph to which it belongs."@en'),
 ('top:isEdgeOf', 'rdfs:domain', 'top:Edge'),
 ('top:isEdgeOf', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isEdgeOf', 'rdfs:label', '"Is Edge Of"@en'),
 ('top:isEdgeOf', 'rdfs:range', 'owl:Thing'),
 ('top:isEdgeOf', 'rdfs:subPropertyOf', 'top:isSubTopologyOf'),
 ('top:isEdgeOf', 'skos:altLabel', '"isEdgeOf"@en'),
 ('top:isFaceOf', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isFaceOf', 'rdfs:comment', '"Associates a face with its parent topology."@en'),
 ('top:isFaceOf', 'rdfs:domain', 'top:Face'),
 ('top:isFaceOf', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isFaceOf', 'rdfs:label', '"Is Face Of"@en'),
 ('top:isFaceOf', 'rdfs:range', 'top:Topology'),
 ('top:isFaceOf', 'rdfs:subPropertyOf', 'top:isSubTopologyOf'),
 ('top:isFaceOf', 'skos:altLabel', '"isFaceOf"@en'),
 ('top:isFilledBy', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isFilledBy',
  'rdfs:comment',
  '"Inverse relation of top:fillsOpening, associating an opening with the element that fills it."@en'),
 ('top:isFilledBy', 'rdfs:domain', 'top:Opening'),
 ('top:isFilledBy', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isFilledBy', 'rdfs:label', '"Is Filled By"@en'),
 ('top:isFilledBy', 'rdfs:range', 'top:Element'),
 ('top:isFilledBy', 'skos:altLabel', '"isFilledBy"@en'),
 ('top:isIFCTypeOf', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isIFCTypeOf', 'rdfs:comment', '"Inverse relation of top:hasIFCType."@en'),
 ('top:isIFCTypeOf', 'rdfs:domain', 'owl:Thing'),
 ('top:isIFCTypeOf', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isIFCTypeOf', 'rdfs:label', '"Is IFC Type Of"@en'),
 ('top:isIFCTypeOf', 'rdfs:range', 'top:Topology'),
 ('top:isIFCTypeOf', 'skos:altLabel', '"isIFCTypeOf"@en'),
 ('top:isMaterialOf', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isMaterialOf', 'rdfs:comment', '"Inverse relation of top:hasMaterial."@en'),
 ('top:isMaterialOf', 'rdfs:domain', 'owl:Thing'),
 ('top:isMaterialOf', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isMaterialOf', 'rdfs:label', '"Is Material Of"@en'),
 ('top:isMaterialOf', 'rdfs:range', 'top:Topology'),
 ('top:isMaterialOf', 'skos:altLabel', '"isMaterialOf"@en'),
 ('top:isOpeningIn', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isOpeningIn',
  'rdfs:comment',
  '"Inverse relation of top:hasOpening, associating an opening with its host element."@en'),
 ('top:isOpeningIn', 'rdfs:domain', 'top:Opening'),
 ('top:isOpeningIn', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isOpeningIn', 'rdfs:label', '"Is Opening In"@en'),
 ('top:isOpeningIn', 'rdfs:range', 'top:Element'),
 ('top:isOpeningIn', 'skos:altLabel', '"isOpeningIn"@en'),
 ('top:isPartOf', 'owl:inverseOf', 'top:containsElement'),
 ('top:isPartOf', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isPartOf',
  'rdfs:comment',
  '"Associates a topology, element, space, or entity with a containing or aggregating whole."@en'),
 ('top:isPartOf', 'rdfs:domain', 'top:Topology'),
 ('top:isPartOf', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isPartOf', 'rdfs:label', '"Is Part Of"@en'),
 ('top:isPartOf', 'rdfs:range', 'top:Topology'),
 ('top:isPartOf', 'rdfs:subPropertyOf', 'bot:isPartOf'),
 ('top:isPartOf', 'skos:altLabel', '"isPartOf"@en'),
 ('top:isPropertySetOf', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isPropertySetOf', 'rdfs:comment', '"Inverse relation of top:hasPropertySet."@en'),
 ('top:isPropertySetOf', 'rdfs:domain', 'top:PropertySet'),
 ('top:isPropertySetOf', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isPropertySetOf', 'rdfs:label', '"Is Property Set Of"@en'),
 ('top:isPropertySetOf', 'rdfs:range', 'owl:Thing'),
 ('top:isPropertySetOf', 'skos:altLabel', '"isPropertySetOf"@en'),
 ('top:isServedBy', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isServedBy',
  'rdfs:comment',
  '"Associates a spatial structure with the system or equipment item that serves it."@en'),
 ('top:isServedBy', 'rdfs:domain', 'top:Topology'),
 ('top:isServedBy', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isServedBy', 'rdfs:label', '"Is Served By"@en'),
 ('top:isServedBy', 'rdfs:range', 'owl:Thing'),
 ('top:isServedBy', 'skos:altLabel', '"isServedBy"@en'),
 ('top:isShellOf', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isShellOf', 'rdfs:comment', '"Associates a shell with its parent topology."@en'),
 ('top:isShellOf', 'rdfs:domain', 'top:Shell'),
 ('top:isShellOf', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isShellOf', 'rdfs:label', '"Is Shell Of"@en'),
 ('top:isShellOf', 'rdfs:range', 'top:Topology'),
 ('top:isShellOf', 'rdfs:subPropertyOf', 'top:isSubTopologyOf'),
 ('top:isShellOf', 'skos:altLabel', '"isShellOf"@en'),
 ('top:isSubTopologyOf', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isSubTopologyOf', 'rdfs:comment', '"Associates a topology with a containing or parent topology."@en'),
 ('top:isSubTopologyOf', 'rdfs:domain', 'top:Topology'),
 ('top:isSubTopologyOf', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isSubTopologyOf', 'rdfs:label', '"Is Sub Topology Of"@en'),
 ('top:isSubTopologyOf', 'rdfs:range', 'top:Topology'),
 ('top:isSubTopologyOf', 'skos:altLabel', '"isSubTopologyOf"@en'),
 ('top:isTopologyOf', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isTopologyOf',
  'rdfs:comment',
  '"Inverse relation of top:hasTopology, associating a topology with the entity it represents."@en'),
 ('top:isTopologyOf', 'rdfs:domain', 'top:Topology'),
 ('top:isTopologyOf', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isTopologyOf', 'rdfs:label', '"Is Topology Of"@en'),
 ('top:isTopologyOf', 'rdfs:range', 'owl:Thing'),
 ('top:isTopologyOf', 'skos:altLabel', '"isTopologyOf"@en'),
 ('top:isVertexOf', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isVertexOf', 'rdfs:comment', '"Associates a vertex with the topology or graph to which it belongs."@en'),
 ('top:isVertexOf', 'rdfs:domain', 'top:Vertex'),
 ('top:isVertexOf', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isVertexOf', 'rdfs:label', '"Is Vertex Of"@en'),
 ('top:isVertexOf', 'rdfs:range', 'owl:Thing'),
 ('top:isVertexOf', 'rdfs:subPropertyOf', 'top:isSubTopologyOf'),
 ('top:isVertexOf', 'skos:altLabel', '"isVertexOf"@en'),
 ('top:isWireOf', 'rdf:type', 'owl:ObjectProperty'),
 ('top:isWireOf', 'rdfs:comment', '"Associates a wire with its parent topology."@en'),
 ('top:isWireOf', 'rdfs:domain', 'top:Wire'),
 ('top:isWireOf', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:isWireOf', 'rdfs:label', '"Is Wire Of"@en'),
 ('top:isWireOf', 'rdfs:range', 'top:Topology'),
 ('top:isWireOf', 'rdfs:subPropertyOf', 'top:isSubTopologyOf'),
 ('top:isWireOf', 'skos:altLabel', '"isWireOf"@en'),
 ('top:label', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:label',
  'rdfs:comment',
  '"A human-readable label emitted from a TopologicPy dictionary when represented as data rather than rdfs:label."@en'),
 ('top:label', 'rdfs:domain', 'owl:Thing'),
 ('top:label', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:label', 'rdfs:label', '"Label"@en'),
 ('top:label', 'rdfs:range', 'xsd:string'),
 ('top:label', 'skos:altLabel', '"label"@en'),
 ('top:length', 'owl:equivalentProperty', 'top:hasLength'),
 ('top:length', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:length',
  'rdfs:comment',
  '"The length of an edge, wire, path, graph edge, or other length-bearing topology or analytical record."@en'),
 ('top:length', 'rdfs:domain', 'top:Topology'),
 ('top:length', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:length', 'rdfs:label', '"Length"@en'),
 ('top:length', 'rdfs:range', 'xsd:double'),
 ('top:length', 'skos:altLabel', '"length"@en'),
 ('top:locatedIn', 'rdf:type', 'owl:ObjectProperty'),
 ('top:locatedIn',
  'rdfs:comment',
  '"Associates a topology, element, node, or entity with the containing, nearest, or inferred spatial structure '
  'derived by geometric or semantic analysis."@en'),
 ('top:locatedIn', 'rdfs:domain', 'top:Topology'),
 ('top:locatedIn', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:locatedIn', 'rdfs:label', '"Located In"@en'),
 ('top:locatedIn', 'rdfs:range', 'top:Topology'),
 ('top:locatedIn', 'skos:altLabel', '"locatedIn"@en'),
 ('top:mantissa', 'owl:equivalentProperty', 'top:hasMantissa'),
 ('top:mantissa', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:mantissa',
  'rdfs:comment',
  '"Alias data property for mantissa when TopologicPy dictionary export emits the raw key mantissa."@en'),
 ('top:mantissa', 'rdfs:domain', 'owl:Thing'),
 ('top:mantissa', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:mantissa', 'rdfs:label', '"Mantissa"@en'),
 ('top:mantissa', 'rdfs:range', 'xsd:integer'),
 ('top:mantissa', 'skos:altLabel', '"mantissa"@en'),
 ('top:modifiedAt', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:modifiedAt', 'rdfs:comment', '"The last modification timestamp of an entity, topology, graph, or record."@en'),
 ('top:modifiedAt', 'rdfs:domain', 'owl:Thing'),
 ('top:modifiedAt', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:modifiedAt', 'rdfs:label', '"Modified At"@en'),
 ('top:modifiedAt', 'rdfs:range', 'xsd:dateTime'),
 ('top:modifiedAt', 'skos:altLabel', '"modifiedAt"@en'),
 ('top:name', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:name',
  'rdfs:comment',
  '"A human-readable name emitted from a TopologicPy dictionary when not represented as rdfs:label."@en'),
 ('top:name', 'rdfs:domain', 'owl:Thing'),
 ('top:name', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:name', 'rdfs:label', '"Name"@en'),
 ('top:name', 'rdfs:range', 'xsd:string'),
 ('top:name', 'skos:altLabel', '"name"@en'),
 ('top:ontologyClass', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:ontologyClass',
  'rdfs:comment',
  '"The ontology class QName or URI recorded in a TopologicPy dictionary, commonly stored under the dictionary key '
  'ontology_class."@en'),
 ('top:ontologyClass', 'rdfs:domain', 'owl:Thing'),
 ('top:ontologyClass', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:ontologyClass', 'rdfs:label', '"Ontology Class"@en'),
 ('top:ontologyClass', 'rdfs:range', 'xsd:string'),
 ('top:ontologyClass', 'skos:altLabel', '"ontologyClass"@en'),
 ('top:ontologyURI', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:ontologyURI',
  'rdfs:comment',
  '"The expanded ontology URI recorded in a TopologicPy dictionary, commonly stored under the dictionary key '
  'ontology_uri."@en'),
 ('top:ontologyURI', 'rdfs:domain', 'owl:Thing'),
 ('top:ontologyURI', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:ontologyURI', 'rdfs:label', '"Ontology URI"@en'),
 ('top:ontologyURI', 'rdfs:range', 'xsd:anyURI'),
 ('top:ontologyURI', 'skos:altLabel', '"ontologyURI"@en'),
 ('top:passesThrough', 'rdf:type', 'owl:ObjectProperty'),
 ('top:passesThrough',
  'rdfs:comment',
  '"Indicates that one topology, element, or system component passes through another topology, element, space, or '
  'region."@en'),
 ('top:passesThrough', 'rdfs:domain', 'top:Topology'),
 ('top:passesThrough', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:passesThrough', 'rdfs:label', '"Passes Through"@en'),
 ('top:passesThrough', 'rdfs:range', 'top:Topology'),
 ('top:passesThrough', 'skos:altLabel', '"passesThrough"@en'),
 ('top:relationship', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:relationship',
  'rdfs:comment',
  '"A general-purpose relationship label emitted from TopologicPy dictionaries when a more specific ontology predicate '
  'is not available."@en'),
 ('top:relationship', 'rdfs:domain', 'owl:Thing'),
 ('top:relationship', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:relationship', 'rdfs:label', '"Relationship"@en'),
 ('top:relationship', 'rdfs:range', 'xsd:string'),
 ('top:relationship', 'skos:altLabel', '"relationship"@en'),
 ('top:requiresOpening', 'rdf:type', 'owl:ObjectProperty'),
 ('top:requiresOpening',
  'rdfs:comment',
  '"Indicates that an element, system component, route, or topology requires an opening through another element."@en'),
 ('top:requiresOpening', 'rdfs:domain', 'top:Topology'),
 ('top:requiresOpening', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:requiresOpening', 'rdfs:label', '"Requires Opening"@en'),
 ('top:requiresOpening', 'rdfs:range', 'top:Element'),
 ('top:requiresOpening', 'skos:altLabel', '"requiresOpening"@en'),
 ('top:servesBuilding', 'rdf:type', 'owl:ObjectProperty'),
 ('top:servesBuilding', 'rdfs:comment', '"Associates a system or equipment item with a building it serves."@en'),
 ('top:servesBuilding', 'rdfs:domain', 'owl:Thing'),
 ('top:servesBuilding', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:servesBuilding', 'rdfs:label', '"Serves Building"@en'),
 ('top:servesBuilding', 'rdfs:range', 'top:Building'),
 ('top:servesBuilding', 'rdfs:subPropertyOf', 'top:servesSpatialStructure'),
 ('top:servesBuilding', 'skos:altLabel', '"servesBuilding"@en'),
 ('top:servesSpatialStructure', 'owl:inverseOf', 'top:isServedBy'),
 ('top:servesSpatialStructure', 'rdf:type', 'owl:ObjectProperty'),
 ('top:servesSpatialStructure',
  'rdfs:comment',
  '"Associates a system or equipment item with the spatial structure it serves, such as a site, building, storey, '
  'space, or zone."@en'),
 ('top:servesSpatialStructure', 'rdfs:domain', 'owl:Thing'),
 ('top:servesSpatialStructure', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:servesSpatialStructure', 'rdfs:label', '"Serves Spatial Structure"@en'),
 ('top:servesSpatialStructure', 'rdfs:range', 'top:Topology'),
 ('top:servesSpatialStructure', 'skos:altLabel', '"servesSpatialStructure"@en'),
 ('top:source', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:source',
  'rdfs:comment',
  '"The source file, model, database, method, or process associated with an entity, topology, graph, or record."@en'),
 ('top:source', 'rdfs:domain', 'owl:Thing'),
 ('top:source', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:source', 'rdfs:label', '"Source"@en'),
 ('top:source', 'rdfs:range', 'xsd:string'),
 ('top:source', 'skos:altLabel', '"source"@en'),
 ('top:srcId', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:srcId',
  'rdfs:comment',
  '"The source node index of a graph relationship record, retained as structural metadata when exported."@en'),
 ('top:srcId', 'rdfs:domain', 'top:Relationship'),
 ('top:srcId', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:srcId', 'rdfs:label', '"Source ID"@en'),
 ('top:srcId', 'rdfs:range', 'xsd:integer'),
 ('top:srcId', 'skos:altLabel', '"srcId"@en'),
 ('top:startsAt', 'owl:equivalentProperty', 'top:hasStartVertex'),
 ('top:startsAt', 'rdf:type', 'owl:ObjectProperty'),
 ('top:startsAt',
  'rdfs:comment',
  '"Alias property for associating an edge or relationship with its start vertex or source node."@en'),
 ('top:startsAt', 'rdfs:domain', 'owl:Thing'),
 ('top:startsAt', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:startsAt', 'rdfs:label', '"Starts At"@en'),
 ('top:startsAt', 'rdfs:range', 'top:Vertex'),
 ('top:startsAt', 'skos:altLabel', '"startsAt"@en'),
 ('top:unit', 'owl:equivalentProperty', 'top:hasUnit'),
 ('top:unit', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:unit',
  'rdfs:comment',
  '"Alias data property for unit when TopologicPy dictionary export emits the raw key unit."@en'),
 ('top:unit', 'rdfs:domain', 'owl:Thing'),
 ('top:unit', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:unit', 'rdfs:label', '"Unit"@en'),
 ('top:unit', 'rdfs:range', 'xsd:string'),
 ('top:unit', 'skos:altLabel', '"unit"@en'),
 ('top:uuid', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:uuid',
  'rdfs:comment',
  '"A stable UUID or UUID-like identifier associated with an entity, topology, graph, node, or relationship."@en'),
 ('top:uuid', 'rdfs:domain', 'owl:Thing'),
 ('top:uuid', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:uuid', 'rdfs:label', '"UUID"@en'),
 ('top:uuid', 'rdfs:range', 'xsd:string'),
 ('top:uuid', 'skos:altLabel', '"uuid"@en'),
 ('top:violatesCoordinationRule', 'rdf:type', 'owl:ObjectProperty'),
 ('top:violatesCoordinationRule',
  'rdfs:comment',
  '"Associates an entity with a violated coordination rule, model-checking rule, or relationship record."@en'),
 ('top:violatesCoordinationRule', 'rdfs:domain', 'top:Topology'),
 ('top:violatesCoordinationRule', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:violatesCoordinationRule', 'rdfs:label', '"Violates Coordination Rule"@en'),
 ('top:violatesCoordinationRule', 'rdfs:range', 'top:Relationship'),
 ('top:violatesCoordinationRule', 'skos:altLabel', '"violatesCoordinationRule"@en'),
 ('top:volume', 'owl:equivalentProperty', 'top:hasVolume'),
 ('top:volume', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:volume',
  'rdfs:comment',
  '"The volume of a cell, cell complex, zone, space, or other volume-bearing topology or analytical record."@en'),
 ('top:volume', 'rdfs:domain', 'top:Topology'),
 ('top:volume', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:volume', 'rdfs:label', '"Volume"@en'),
 ('top:volume', 'rdfs:range', 'xsd:double'),
 ('top:volume', 'skos:altLabel', '"volume"@en'),
 ('top:weight', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:weight', 'rdfs:comment', '"A numeric graph, path, analysis, or relationship weight."@en'),
 ('top:weight', 'rdfs:domain', 'owl:Thing'),
 ('top:weight', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:weight', 'rdfs:label', '"Weight"@en'),
 ('top:weight', 'rdfs:range', 'xsd:double'),
 ('top:weight', 'skos:altLabel', '"weight"@en'),
 ('top:x', 'owl:equivalentProperty', 'top:hasX'),
 ('top:x', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:x',
  'rdfs:comment',
  '"Alias data property for the X coordinate when TopologicPy dictionary export emits the raw key x."@en'),
 ('top:x', 'rdfs:domain', 'top:Vertex'),
 ('top:x', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:x', 'rdfs:label', '"X"@en'),
 ('top:x', 'rdfs:range', 'xsd:double'),
 ('top:x', 'skos:altLabel', '"x"@en'),
 ('top:y', 'owl:equivalentProperty', 'top:hasY'),
 ('top:y', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:y',
  'rdfs:comment',
  '"Alias data property for the Y coordinate when TopologicPy dictionary export emits the raw key y."@en'),
 ('top:y', 'rdfs:domain', 'top:Vertex'),
 ('top:y', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:y', 'rdfs:label', '"Y"@en'),
 ('top:y', 'rdfs:range', 'xsd:double'),
 ('top:y', 'skos:altLabel', '"y"@en'),
 ('top:z', 'owl:equivalentProperty', 'top:hasZ'),
 ('top:z', 'rdf:type', 'owl:DatatypeProperty'),
 ('top:z',
  'rdfs:comment',
  '"Alias data property for the Z coordinate when TopologicPy dictionary export emits the raw key z."@en'),
 ('top:z', 'rdfs:domain', 'top:Vertex'),
 ('top:z', 'rdfs:isDefinedBy', 'http://w3id.org/topologicpy'),
 ('top:z', 'rdfs:label', '"Z"@en'),
 ('top:z', 'rdfs:range', 'xsd:double'),
 ('top:z', 'skos:altLabel', '"z"@en')]


# -----------------------------------------------------------------------------
# Ontology 005 corrections
# -----------------------------------------------------------------------------
# This block aligns the class with topologicpy_005.ttl. It preserves the public
# API of Ontology.py while replacing the vocabulary, RDF serialisation, and TTL
# generation with the corrected 005 ontology model.

def _Ontology005_Apply():
    import re

    Ontology.NAMESPACES = __NAMESPACES_005.copy()
    Ontology.TOP_SUPERCLASSES = __TOP_SUPERCLASSES_005.copy()
    Ontology.CLASS_COMMENTS = __CLASS_COMMENTS_005.copy()
    Ontology.OBJECT_PROPERTIES = __OBJECT_PROPERTIES_005.copy()
    Ontology.DATA_PROPERTIES = __DATA_PROPERTIES_005.copy()
    Ontology.DEPRECATED_PROPERTIES = set(__DEPRECATED_PROPERTIES_005)
    Ontology.KNOWN_TOP_PROPERTIES = set(Ontology.OBJECT_PROPERTIES.keys()) | set(Ontology.DATA_PROPERTIES.keys())
    Ontology.KNOWN_TOP_CLASSES = set(Ontology.TOP_SUPERCLASSES.keys())
    Ontology.ONTOLOGY_TRIPLES_005 = list(__ONTOLOGY_TRIPLES_005)

    Ontology.PROPERTY_ALIASES = {
        "hasStartVertex": "startsAt", "hasEndVertex": "endsAt", "startVertex": "startsAt", "endVertex": "endsAt",
        "hasVertices": "hasVertex", "hasEdges": "hasEdge", "hasWires": "hasWire", "hasFaces": "hasFace",
        "hasShells": "hasShell", "hasCells": "hasCell", "hasCellComplexes": "hasCellComplex",
        "connectedTo": "connectsTo", "isConnectedTo": "connectsTo", "adjacent": "adjacentTo",
        "contains": "containsElement", "containedIn": "isPartOf",
        "ontology_class": "ontologyClass", "ontology_uri": "ontologyURI", "ontologyURI": "ontologyURI",
        "ontologyClass": "ontologyClass", "ifc_class": "ifcClass", "ifcClass": "ifcClass",
        "ifc_guid": "ifcGUID", "ifcGlobalId": "ifcGUID", "ifcGlobalID": "ifcGUID", "IFC_global_id": "ifcGUID",
        "IFC_guid": "ifcGUID", "IFC_GUID": "ifcGUID", "GlobalId": "ifcGUID", "global_id": "ifcGUID", "guid": "ifcGUID",
        "IFC_id": "ifcStepId", "ifc_id": "ifcStepId", "ifcId": "ifcStepId", "ifc_step_id": "ifcStepId",
        "IFC_key": "ifcStepKey", "ifc_key": "ifcStepKey", "ifc_step_key": "ifcStepKey",
        "IFC_name": "ifcName", "ifc_name": "ifcName", "ifcName": "ifcName",
        "IFC_type": "ifcType", "ifc_type": "ifcType", "ifcType": "ifcType",
        "created_at": "createdAt", "modified_at": "modifiedAt", "updated_at": "modifiedAt", "updatedAt": "modifiedAt",
        "derived_from": "derivedFrom", "derivedFrom": "derivedFrom",
        "generated_by": "generatedByMethod", "generatedByMethod": "generatedByMethod", "generatedBy": "generatedBy",
        "source": "source", "description": "description", "name": "name", "label": "label", "category": "category",
        "relationship": "relationship",
        "hasX": "x", "hasY": "y", "hasZ": "z", "x": "x", "y": "y", "z": "z",
        "hasLength": "length", "hasArea": "area", "hasVolume": "volume", "hasMantissa": "mantissa", "hasUnit": "unit",
        "length": "length", "area": "area", "volume": "volume", "mantissa": "mantissa", "unit": "unit",
        "index": "index", "uuid": "uuid", "src": "srcId", "dst": "dstId", "src_id": "srcId", "dst_id": "dstId",
        "srcId": "srcId", "dstId": "dstId", "weight": "weight", "feature": "feature",
        "feature_vector": "featureVector", "featureVector": "featureVector",
    }

    Ontology.CLASS_ALIASES = {"top:TGraph": "top:Graph", "TGraph": "top:Graph", "Graph": "top:Graph"}

    Ontology.INTERNAL_EXPORT_KEYS = {
        "active", "directed", "dictionary_mode", "dictionaryMode", "import_mode", "importMode",
        "color", "colour", "ontology_predicate", "ontologyPredicate", "inverse_predicate", "inversePredicate",
        "ifc_relationship", "ifcRelationship", "relationship_predicate", "relationshipPredicate", "asTopologic", "representation",
    }

    def _literal(value, datatype=None, language=None):
        if isinstance(value, bool) and datatype is None and language is None:
            return '"' + str(value).lower() + '"^^xsd:boolean'
        if isinstance(value, int) and not isinstance(value, bool) and datatype is None and language is None:
            return '"' + str(value) + '"^^xsd:integer'
        if isinstance(value, float) and datatype is None and language is None:
            return '"' + repr(value) + '"^^xsd:double'
        text = "" if value is None else str(value)
        text = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        lit = '"' + text + '"'
        if language:
            return lit + "@" + str(language).strip()
        if datatype:
            return lit + "^^" + str(datatype).strip()
        return lit

    def _safe_local_name(value):
        if value is None:
            return "unnamed"
        s = str(value).strip()
        if not s:
            return "unnamed"
        s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        if not s:
            s = "unnamed"
        if s[0].isdigit():
            s = "id_" + s
        return s

    def _is_uri_or_qname(value):
        if not isinstance(value, str):
            return False
        text = value.strip()
        if text.startswith("<") and text.endswith(">"):
            return True
        if text.startswith("http://") or text.startswith("https://"):
            return True
        if ":" in text:
            prefix, local = text.split(":", 1)
            return bool(local) and prefix in Ontology.NAMESPACES
        return False

    def _expand_qname(qname, defaultValue=None):
        if not isinstance(qname, str):
            return defaultValue
        text = qname.strip()
        if text.startswith("<") and text.endswith(">"):
            return text[1:-1]
        if text.startswith("http://") or text.startswith("https://"):
            return text
        if ":" not in text:
            return defaultValue
        prefix, local = text.split(":", 1)
        ns = Ontology.NAMESPACES.get(prefix)
        if ns is None:
            return defaultValue
        return ns + local

    def _property_qname(key, defaultPrefix="top"):
        if key is None:
            return None
        text = str(key).strip()
        if text == "":
            return None
        if ":" in text:
            prefix, local = text.split(":", 1)
            if prefix != "top":
                return text
            canonical = Ontology.PROPERTY_ALIASES.get(local, local)
            candidate = "top:" + _safe_local_name(canonical)
            if candidate in Ontology.KNOWN_TOP_PROPERTIES:
                return candidate
            return "dict:" + _safe_local_name(canonical)
        canonical = Ontology.PROPERTY_ALIASES.get(text, text)
        candidate = "top:" + _safe_local_name(canonical)
        if candidate in Ontology.KNOWN_TOP_PROPERTIES:
            return candidate
        return "dict:" + _safe_local_name(canonical)

    def _canonical_class(ontologyClass, defaultValue=None):
        if ontologyClass is None:
            return defaultValue
        cls = str(ontologyClass).strip()
        if not cls:
            return defaultValue
        if ":" not in cls and ("top:" + cls) in Ontology.KNOWN_TOP_CLASSES:
            cls = "top:" + cls
        return Ontology.CLASS_ALIASES.get(cls, cls)

    def _triple_object(predicate, value):
        if value is None:
            return None
        predicate = _property_qname(predicate) if predicate and ":" not in str(predicate) else str(predicate)
        if predicate in Ontology.OBJECT_PROPERTIES:
            if _is_uri_or_qname(value):
                return str(value).strip()
            return "inst:" + _safe_local_name(value)
        return _literal(value)

    def _record_identity(obj, role="item", fallbackIndex=None, prefix="inst"):
        d = Ontology._record_dictionary(obj) or {}
        for key in [Ontology.URI_KEY, "uri", "URI"]:
            value = d.get(key)
            if isinstance(value, str) and value.strip():
                value = value.strip()
                if _is_uri_or_qname(value):
                    return value
                return prefix + ":" + _safe_local_name(value)
        for key in ["uuid", "UUID", "id", "ID", Ontology.IFC_GUID_KEY, "ifc_guid", "ifcGUID", "IFC_global_id", "GlobalId"]:
            value = d.get(key)
            if value not in [None, ""]:
                return prefix + ":" + _safe_local_name(str(value))
        if isinstance(obj, dict):
            for key in ["uuid", "id", "index"]:
                value = obj.get(key)
                if value not in [None, ""]:
                    return prefix + ":" + role + "_" + _safe_local_name(str(value))
        if fallbackIndex is not None:
            return prefix + ":" + role + "_" + _safe_local_name(str(fallbackIndex))
        try:
            from topologicpy.Topology import Topology
            uid = Topology.UUID(obj, silent=True)
            if uid not in [None, ""]:
                return prefix + ":" + role + "_" + _safe_local_name(uid)
        except Exception:
            pass
        return prefix + ":" + role + "_" + _safe_local_name(id(obj))

    def _uri_for_topology(topology, prefix="inst"):
        return _record_identity(topology, role="item", fallbackIndex=None, prefix=prefix)

    def _dictionary_items(d):
        if isinstance(d, dict):
            return list(d.items())
        try:
            from topologicpy.Dictionary import Dictionary
            return [(k, Dictionary.ValueAtKey(d, k, None)) for k in Dictionary.Keys(d)]
        except Exception:
            return []

    def _triples(topology, subject=None, includeDictionaries=True, includeBOT=True, includeType=True,
                 includeLabel=True, includeCategory=True, namespacePrefix="inst", silent=False):
        if topology is None:
            if not silent:
                print("Ontology.Triples - Error: The input topology is None. Returning an empty list.")
            return []
        triples = []
        if subject is None:
            subject = _uri_for_topology(topology, prefix=namespacePrefix)
        ontologyClass = _canonical_class(Ontology.Class(topology), defaultValue=Ontology.Class(topology))
        label = Ontology.Label(topology)
        category = Ontology.Category(topology)
        if includeType and ontologyClass is not None:
            triples.append((subject, "rdf:type", ontologyClass))
            if includeBOT:
                botClass = Ontology.BOTClassByClass(ontologyClass)
                if botClass is not None:
                    triples.append((subject, "rdf:type", botClass))
        if includeLabel and label not in [None, ""]:
            triples.append((subject, "rdfs:label", _literal(label)))
        if includeCategory and category not in [None, ""]:
            triples.append((subject, "top:category", _literal(category)))
        if includeDictionaries:
            d = Ontology._dictionary(topology)
            skip_keys = {Ontology.ONTOLOGY_CLASS_KEY, Ontology.ONTOLOGY_URI_KEY, Ontology.LABEL_KEY,
                         Ontology.CATEGORY_KEY, Ontology.URI_KEY, "uri", "URI"} | set(Ontology.INTERNAL_EXPORT_KEYS)
            for key, value in _dictionary_items(d):
                if key in skip_keys or value is None:
                    continue
                predicate = _property_qname(key)
                if predicate is None:
                    continue
                values = value if isinstance(value, (list, tuple, set)) else [value]
                for v in values:
                    obj = _triple_object(predicate, v)
                    if obj is not None:
                        triples.append((subject, predicate, obj))
        return triples

    def _graph_triples(graph, includeVertices=True, includeEdges=True, includeDictionaries=True,
                       includeBOT=True, namespacePrefix="inst", silent=False):
        triples = []
        if graph is None:
            if not silent:
                print("Ontology.GraphTriples - Error: The input graph is None. Returning an empty list.")
            return triples
        try:
            graph_subject = _record_identity(graph, role="graph", fallbackIndex="graph", prefix=namespacePrefix)
            triples.extend(_triples(graph, subject=graph_subject, includeDictionaries=includeDictionaries,
                                    includeBOT=includeBOT, namespacePrefix=namespacePrefix, silent=True))
            vertex_records = Ontology._graph_vertices(graph, asTopologic=False) or []
            vertex_subjects = {}
            if includeVertices:
                for i, vertex in enumerate(vertex_records):
                    if Ontology.Class(vertex) is None:
                        Ontology.Annotate(vertex, ontologyClass="top:Node", silent=True)
                    v_subject = _record_identity(vertex, role="node", fallbackIndex=i, prefix=namespacePrefix)
                    vertex_subjects[i] = v_subject
                    if isinstance(vertex, dict):
                        vertex_subjects[vertex.get("index", i)] = v_subject
                    triples.append((graph_subject, "top:hasNode", v_subject))
                    triples.extend(_triples(vertex, subject=v_subject, includeDictionaries=includeDictionaries,
                                            includeBOT=includeBOT, namespacePrefix=namespacePrefix, silent=True))
                    if isinstance(vertex, dict):
                        vd = Ontology._record_dictionary(vertex) or {}
                        # Add explicit coordinate triples only when they were not already
                        # exported from the vertex dictionary. This prevents duplicate
                        # integer/double variants such as top:x 0 and top:x 0.0.
                        missing_coords = not all(k in vd for k in ["x", "y", "z"])
                        if missing_coords:
                            x, y, z = Ontology._record_coordinates(vertex)
                            triples.append((v_subject, "top:x", _literal(x)))
                            triples.append((v_subject, "top:y", _literal(y)))
                            triples.append((v_subject, "top:z", _literal(z)))
            if includeEdges:
                edges = Ontology._graph_edges(graph, asTopologic=False) or []
                if not vertex_subjects:
                    for i, vertex in enumerate(vertex_records):
                        v_subject = _record_identity(vertex, role="node", fallbackIndex=i, prefix=namespacePrefix)
                        vertex_subjects[i] = v_subject
                        if isinstance(vertex, dict):
                            vertex_subjects[vertex.get("index", i)] = v_subject
                for i, edge in enumerate(edges):
                    if Ontology.Class(edge) is None:
                        Ontology.Annotate(edge, ontologyClass="top:Relationship", silent=True)
                    e_subject = _record_identity(edge, role="edge", fallbackIndex=i, prefix=namespacePrefix)
                    triples.append((graph_subject, "top:hasRelationship", e_subject))
                    triples.extend(_triples(edge, subject=e_subject, includeDictionaries=includeDictionaries,
                                            includeBOT=includeBOT, namespacePrefix=namespacePrefix, silent=True))
                    sv, ev = Ontology._graph_edge_endpoints(graph, edge, vertex_records=vertex_records)
                    sv_subject = _record_identity(sv, role="node", fallbackIndex=None, prefix=namespacePrefix) if sv is not None else None
                    ev_subject = _record_identity(ev, role="node", fallbackIndex=None, prefix=namespacePrefix) if ev is not None else None
                    if sv_subject is not None:
                        triples.append((e_subject, "top:startsAt", sv_subject))
                    if ev_subject is not None:
                        triples.append((e_subject, "top:endsAt", ev_subject))
                    if sv_subject is not None and ev_subject is not None:
                        triples.append((sv_subject, "top:connectsTo", ev_subject))
                        d = Ontology._record_dictionary(edge) or {}
                        predicate = d.get("ontology_predicate") or d.get("ontologyPredicate")
                        inverse = d.get("inverse_predicate") or d.get("inversePredicate")
                        if predicate:
                            pred_q = _property_qname(predicate)
                            if pred_q and not pred_q.startswith("dict:"):
                                triples.append((sv_subject, pred_q, ev_subject))
                        if inverse:
                            inv_q = _property_qname(inverse)
                            if inv_q and not inv_q.startswith("dict:"):
                                triples.append((ev_subject, inv_q, sv_subject))
            seen = set(); result = []
            for t in triples:
                if t not in seen:
                    seen.add(t); result.append(t)
            return result
        except Exception as e:
            if not silent:
                print("Ontology.GraphTriples - Error: Could not create graph triples. Returning partial list.")
                print("Error:", e)
            return triples

    def _turtle_from_triples(triples, namespaces=None, instanceNamespace="http://w3id.org/topologicpy/instance#", includeHeader=True):
        namespaces = dict(namespaces or Ontology.NAMESPACES)
        namespaces.setdefault("inst", instanceNamespace)
        namespaces.setdefault("dict", "http://w3id.org/topologicpy/dictionary#")
        def is_literal(term):
            return isinstance(term, str) and term.startswith('"')
        def is_qname(term):
            if not isinstance(term, str) or ":" not in term:
                return False
            prefix, local = term.split(":", 1)
            return bool(local) and prefix in namespaces
        def format_term(term, predicate=False):
            if term is None:
                return None
            if is_literal(term):
                return term
            if not isinstance(term, str):
                return _literal(term)
            text = term.strip()
            if not text:
                return None
            if is_qname(text):
                return text
            if text.startswith("_:"):
                return text
            if text.startswith("<") and text.endswith(">"):
                return text
            if text.startswith("http://") or text.startswith("https://"):
                return "<" + text + ">"
            if predicate:
                return "dict:" + _safe_local_name(text)
            return _literal(text)
        lines = []
        if includeHeader:
            for prefix, uri in namespaces.items():
                lines.append("@prefix " + prefix + ": <" + uri + "> .")
            lines.append("")
        seen = set()
        for triple in triples or []:
            if triple is None or len(triple) != 3:
                continue
            fs, fp, fo = format_term(triple[0]), format_term(triple[1], predicate=True), format_term(triple[2])
            if fs is None or fp is None or fo is None:
                continue
            line = fs + " " + fp + " " + fo + " ."
            if line not in seen:
                seen.add(line); lines.append(line)
        return "\n".join(lines) + "\n"

    def _ontology_triples(includeBOT=True):
        return list(Ontology.ONTOLOGY_TRIPLES_005)

    def _ontology_ttl_string(includeBOT=True, instanceNamespace="http://w3id.org/topologicpy/instance#"):
        return _turtle_from_triples(_ontology_triples(includeBOT=includeBOT), namespaces=Ontology.NAMESPACES,
                                    instanceNamespace=instanceNamespace, includeHeader=True)

    def _validate(topology, requireClass=False, silent=False):
        report = {"valid": True, "errors": [], "warnings": []}
        d = Ontology._dictionary(topology)
        if d is None:
            report["warnings"].append("No dictionary was found on the input object.")
            if requireClass:
                report["valid"] = False
                report["errors"].append("No dictionary was found on the input object.")
            return report
        cls = d.get(Ontology.ONTOLOGY_CLASS_KEY)
        if requireClass and cls in [None, ""]:
            report["valid"] = False
            report["errors"].append("Missing ontology_class.")
        if cls not in [None, ""]:
            ccls = _canonical_class(cls, defaultValue=cls)
            if ccls != cls:
                report["warnings"].append("Non-canonical ontology class " + str(cls) + " should be " + str(ccls) + ".")
            if isinstance(ccls, str) and ccls.startswith("top:") and ccls not in Ontology.KNOWN_TOP_CLASSES:
                report["warnings"].append("Unknown top: ontology class: " + str(ccls) + ".")
        for key in d.keys() if isinstance(d, dict) else []:
            pred = _property_qname(key)
            if pred and pred.startswith("top:") and pred not in Ontology.KNOWN_TOP_PROPERTIES:
                report["warnings"].append("Dictionary key " + str(key) + " maps to unknown top: predicate " + str(pred) + ".")
        if not silent and not report["valid"]:
            print("Ontology.Validate - Error: Ontology metadata is invalid.")
            for error in report["errors"]:
                print(" -", error)
        return report

    Ontology._safe_local_name = staticmethod(_safe_local_name)
    Ontology._rdf_literal = staticmethod(_literal)
    Ontology.ExpandQName = staticmethod(_expand_qname)
    Ontology.CanonicalClass = staticmethod(_canonical_class)
    Ontology.IsResourceString = staticmethod(_is_uri_or_qname)
    Ontology.PropertyQName = staticmethod(_property_qname)
    Ontology._triple_object = staticmethod(_triple_object)
    Ontology._uri_for_topology = staticmethod(_uri_for_topology)
    Ontology.Triples = staticmethod(_triples)
    Ontology.GraphTriples = staticmethod(_graph_triples)
    Ontology.TurtleFromTriples = staticmethod(_turtle_from_triples)
    Ontology.OntologyTriples = staticmethod(_ontology_triples)
    Ontology.OntologyTTLString = staticmethod(_ontology_ttl_string)
    Ontology.Validate = staticmethod(_validate)

_Ontology005_Apply()


# -----------------------------------------------------------------------------
# Final compatibility and robustness fixes
# -----------------------------------------------------------------------------

def _Ontology_FinalFix_Apply():
    """Applies final robustness patches without changing the public API."""

    def _dictionary_value(dictionary, key, defaultValue=None):
        if dictionary is None or key is None:
            return defaultValue
        if isinstance(dictionary, dict):
            return dictionary.get(key, defaultValue)
        try:
            from topologicpy.Dictionary import Dictionary
            try:
                return Dictionary.ValueAtKey(dictionary, key, defaultValue)
            except TypeError:
                value = Dictionary.ValueAtKey(dictionary, key)
                return defaultValue if value is None else value
        except Exception:
            try:
                return dictionary.get(key, defaultValue)
            except Exception:
                return defaultValue

    def _dictionary_to_python(dictionary):
        if dictionary is None:
            return {}
        if isinstance(dictionary, dict):
            return dict(dictionary)
        try:
            from topologicpy.Dictionary import Dictionary
            return dict(Dictionary.PythonDictionary(dictionary) or {})
        except Exception:
            try:
                keys = Dictionary.Keys(dictionary)  # type: ignore[name-defined]
                return {k: _dictionary_value(dictionary, k, None) for k in keys}
            except Exception:
                return {}

    def _set_value(topology, key, value, silent=False):
        """Sets a dictionary key/value pair and preserves returned topology objects."""
        if topology is None:
            if not silent:
                print("Ontology._set_value - Error: The input topology is None. Returning None.")
            return None
        if key is None:
            if not silent:
                print("Ontology._set_value - Error: The input key is None. Returning None.")
            return None

        if isinstance(topology, dict):
            d = topology.get("dictionary", topology)
            if not isinstance(d, dict):
                d = {}
            d[key] = value
            if "dictionary" in topology:
                topology["dictionary"] = d
            return topology

        if Ontology._is_tgraph(topology):
            try:
                from topologicpy.TGraph import TGraph
                d = dict(TGraph.Dictionary(topology) or {})
                d[key] = value
                try:
                    result = topology.SetDictionary(d)
                    return result if result is not None else topology
                except Exception:
                    result = TGraph.SetDictionary(topology, d)
                    return result if result is not None else topology
            except Exception as e:
                if not silent:
                    print("Ontology._set_value - Error: Could not set the TGraph dictionary value. Returning None.")
                    print("Error:", e)
                return None

        try:
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary
            d = Topology.Dictionary(topology)
            if d is None:
                d = Dictionary.ByPythonDictionary({})
            d = Dictionary.SetValueAtKey(d, key, value)
            try:
                result = Topology.SetDictionary(topology, d, silent=silent)
            except TypeError:
                result = Topology.SetDictionary(topology, d)
            return result if result is not None else topology
        except Exception as e:
            if not silent:
                print("Ontology._set_value - Error: Could not set the dictionary value. Returning None.")
                print("Error:", e)
            return None

    def _value(topology, key, defaultValue=None):
        return _dictionary_value(Ontology._dictionary(topology), key, defaultValue)

    def _normalize_dictionary(topology,
                              labelKeys=None,
                              categoryKeys=None,
                              ifcClassKeys=None,
                              ifcGUIDKeys=None,
                              silent=False):
        """Normalises common dictionary keys into canonical ontology keys."""
        if topology is None:
            if not silent:
                print("Ontology.NormalizeDictionary - Error: The input topology is None. Returning None.")
            return None

        labelKeys = labelKeys or ["name", "Name", "LongName", "ifc_name", "IFC_name", "label"]
        categoryKeys = categoryKeys or ["category", "type", "ObjectType"]
        ifcClassKeys = ifcClassKeys or ["ifc_class", "IfcClass", "IFC_class", "class", "type"]
        ifcGUIDKeys = ifcGUIDKeys or ["ifc_guid", "IFC_guid", "IFC_GUID", "GlobalId", "global_id", "guid"]

        d = Ontology._dictionary(topology)
        if d is None:
            return topology

        def first_value(keys):
            for key in keys:
                value = _dictionary_value(d, key, None)
                if value is not None and value != "":
                    return value
            return None

        label = first_value(labelKeys)
        category = first_value(categoryKeys)
        ifcClass = first_value(ifcClassKeys)
        ifcGUID = first_value(ifcGUIDKeys)

        if label is not None:
            topology = Ontology.SetLabel(topology, label, silent=silent)
        if category is not None:
            topology = Ontology.SetCategory(topology, str(category).lower(), silent=silent)
        if ifcClass is not None:
            topology = Ontology.AnnotateIFC(topology, ifcClass=ifcClass, silent=silent)
        if ifcGUID is not None:
            topology = Ontology._set_value(topology, Ontology.IFC_GUID_KEY, ifcGUID, silent=silent)

        return topology

    def _validate(topology,
                  requireClass=True,
                  requireCategory=False,
                  requireLabel=False,
                  requireURI=False,
                  checkClassKnown=True,
                  checkCategory=True,
                  checkQName=True,
                  silent=False):
        """Validates ontology metadata using the documented report schema."""
        report = {
            "ok": False,
            "valid": False,
            "errors": [],
            "warnings": [],
            "dictionary": {},
        }

        if topology is None:
            report["errors"].append("The input topology is None.")
            return report

        d_raw = Ontology._dictionary(topology)
        d = _dictionary_to_python(d_raw)
        report["dictionary"] = dict(d)

        if d_raw is None:
            if requireClass or requireCategory or requireLabel or requireURI:
                report["errors"].append("No dictionary was found on the input object.")
            else:
                report["warnings"].append("No dictionary was found on the input object.")

        cls = d.get(Ontology.ONTOLOGY_CLASS_KEY)
        category = d.get(Ontology.CATEGORY_KEY)
        label = d.get(Ontology.LABEL_KEY)
        uri = d.get(Ontology.URI_KEY) or d.get(Ontology.ONTOLOGY_URI_KEY)

        if requireClass and cls in (None, ""):
            report["errors"].append("Missing ontology_class.")
        if requireCategory and category in (None, ""):
            report["errors"].append("Missing category.")
        if requireLabel and label in (None, ""):
            report["errors"].append("Missing label.")
        if requireURI and uri in (None, ""):
            report["errors"].append("Missing uri or ontology_uri.")

        canonical_cls = None
        if cls not in (None, ""):
            canonical_cls = Ontology.CanonicalClass(cls, defaultValue=cls)
            if canonical_cls != cls:
                report["warnings"].append(
                    "Non-canonical ontology class " + str(cls) + " should be " + str(canonical_cls) + "."
                )
            if checkClassKnown and isinstance(canonical_cls, str) and canonical_cls.startswith("top:"):
                if canonical_cls not in getattr(Ontology, "KNOWN_TOP_CLASSES", set(Ontology.TOP_SUPERCLASSES.keys())):
                    report["warnings"].append("Unknown top: ontology class: " + str(canonical_cls) + ".")
            if checkQName and isinstance(canonical_cls, str) and ":" in canonical_cls:
                prefix, local = canonical_cls.split(":", 1)
                if not prefix or not local or prefix not in Ontology.NAMESPACES:
                    report["warnings"].append("Unknown or invalid QName prefix in ontology_class: " + str(canonical_cls) + ".")

        if checkCategory and canonical_cls not in (None, "") and category not in (None, ""):
            expected = Ontology.CategoryByClass(canonical_cls, defaultValue=None)
            if expected is not None and str(category) != str(expected):
                report["warnings"].append(
                    "Category " + str(category) + " does not match expected category " + str(expected) + " for " + str(canonical_cls) + "."
                )

        if checkQName:
            for key in (Ontology.ONTOLOGY_URI_KEY, Ontology.URI_KEY):
                value = d.get(key)
                if isinstance(value, str) and ":" in value and not value.startswith(("http://", "https://", "<")):
                    prefix, local = value.split(":", 1)
                    if not prefix or not local or prefix not in Ontology.NAMESPACES:
                        report["warnings"].append("Unknown or invalid QName prefix in " + str(key) + ": " + str(value) + ".")

        known_properties = getattr(Ontology, "KNOWN_TOP_PROPERTIES", set(Ontology.OBJECT_PROPERTIES.keys()) | set(Ontology.DATA_PROPERTIES.keys()))
        for key in d.keys():
            pred = Ontology.PropertyQName(key)
            if pred and pred.startswith("top:") and pred not in known_properties:
                report["warnings"].append(
                    "Dictionary key " + str(key) + " maps to unknown top: predicate " + str(pred) + "."
                )

        report["ok"] = len(report["errors"]) == 0
        report["valid"] = report["ok"]

        if not silent and report["errors"]:
            print("Ontology.Validate - Error: Ontology metadata is invalid.")
            for error in report["errors"]:
                print(" -", error)
        return report

    Ontology._set_value = staticmethod(_set_value)
    Ontology._value = staticmethod(_value)
    Ontology.NormalizeDictionary = staticmethod(_normalize_dictionary)
    Ontology.Validate = staticmethod(_validate)

_Ontology_FinalFix_Apply()


# -----------------------------------------------------------------------------
# Final identity/export fix
# -----------------------------------------------------------------------------

def _Ontology_FinalIdentityFix_Apply():
    """Restores dictionary-aware instance identities for non-record objects."""

    def _safe_local_name(value):
        try:
            return Ontology._safe_local_name(value)
        except Exception:
            import re
            s = "unnamed" if value is None else str(value).strip()
            s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_") or "unnamed"
            return "id_" + s if s[0].isdigit() else s

    def _is_resource(value):
        try:
            return Ontology.IsResourceString(value)
        except Exception:
            return isinstance(value, str) and (value.startswith("http://") or value.startswith("https://") or ":" in value)

    def _dict_for_identity(obj):
        d = Ontology._record_dictionary(obj)
        if isinstance(d, dict):
            return d
        d = Ontology._dictionary(obj)
        if isinstance(d, dict):
            return d
        try:
            from topologicpy.Dictionary import Dictionary
            return dict(Dictionary.PythonDictionary(d) or {})
        except Exception:
            return {}

    def _identity(obj, role="item", fallbackIndex=None, prefix="inst"):
        d = _dict_for_identity(obj)
        for key in [Ontology.URI_KEY, "uri", "URI"]:
            value = d.get(key)
            if isinstance(value, str) and value.strip():
                value = value.strip()
                if _is_resource(value):
                    return value
                return prefix + ":" + _safe_local_name(value)
        for key in ["uuid", "UUID", "id", "ID", Ontology.IFC_GUID_KEY, "ifc_guid", "ifcGUID", "IFC_global_id", "GlobalId", Ontology.LABEL_KEY]:
            value = d.get(key)
            if value not in (None, ""):
                return prefix + ":" + _safe_local_name(str(value))
        if isinstance(obj, dict):
            for key in ["uuid", "id", "index"]:
                value = obj.get(key)
                if value not in (None, ""):
                    return prefix + ":" + role + "_" + _safe_local_name(str(value))
        if fallbackIndex is not None:
            return prefix + ":" + role + "_" + _safe_local_name(str(fallbackIndex))
        try:
            from topologicpy.Topology import Topology
            uid = Topology.UUID(obj, silent=True)
            if uid not in (None, ""):
                return prefix + ":" + role + "_" + _safe_local_name(uid)
        except Exception:
            pass
        return prefix + ":" + role + "_" + _safe_local_name(id(obj))

    def _uri_for_topology(topology, prefix="inst"):
        return _identity(topology, role="item", fallbackIndex=None, prefix=prefix)

    def _graph_triples(graph,
                       includeVertices=True,
                       includeEdges=True,
                       includeDictionaries=True,
                       includeBOT=True,
                       namespacePrefix="inst",
                       silent=False):
        triples = []
        if graph is None:
            if not silent:
                print("Ontology.GraphTriples - Error: The input graph is None. Returning an empty list.")
            return triples
        try:
            graph_subject = _identity(graph, role="graph", fallbackIndex="graph", prefix=namespacePrefix)
            triples.extend(Ontology.Triples(graph,
                                            subject=graph_subject,
                                            includeDictionaries=includeDictionaries,
                                            includeBOT=includeBOT,
                                            namespacePrefix=namespacePrefix,
                                            silent=True))
            vertex_records = Ontology._graph_vertices(graph, asTopologic=False) or []
            vertex_subjects = {}
            if includeVertices:
                for i, vertex in enumerate(vertex_records):
                    if Ontology.Class(vertex) is None:
                        Ontology.Annotate(vertex, ontologyClass="top:Node", silent=True)
                    v_subject = _identity(vertex, role="node", fallbackIndex=i, prefix=namespacePrefix)
                    vertex_subjects[i] = v_subject
                    if isinstance(vertex, dict):
                        vertex_subjects[vertex.get("index", i)] = v_subject
                    triples.append((graph_subject, "top:hasNode", v_subject))
                    triples.extend(Ontology.Triples(vertex,
                                                    subject=v_subject,
                                                    includeDictionaries=includeDictionaries,
                                                    includeBOT=includeBOT,
                                                    namespacePrefix=namespacePrefix,
                                                    silent=True))
                    if isinstance(vertex, dict):
                        vd = Ontology._record_dictionary(vertex) or {}
                        if not all(k in vd for k in ["x", "y", "z"]):
                            x, y, z = Ontology._record_coordinates(vertex)
                            triples.append((v_subject, "top:x", Ontology._rdf_literal(x)))
                            triples.append((v_subject, "top:y", Ontology._rdf_literal(y)))
                            triples.append((v_subject, "top:z", Ontology._rdf_literal(z)))
            if includeEdges:
                edges = Ontology._graph_edges(graph, asTopologic=False) or []
                if not vertex_subjects:
                    for i, vertex in enumerate(vertex_records):
                        v_subject = _identity(vertex, role="node", fallbackIndex=i, prefix=namespacePrefix)
                        vertex_subjects[i] = v_subject
                        if isinstance(vertex, dict):
                            vertex_subjects[vertex.get("index", i)] = v_subject
                for i, edge in enumerate(edges):
                    if Ontology.Class(edge) is None:
                        Ontology.Annotate(edge, ontologyClass="top:Relationship", silent=True)
                    e_subject = _identity(edge, role="edge", fallbackIndex=i, prefix=namespacePrefix)
                    triples.append((graph_subject, "top:hasRelationship", e_subject))
                    triples.extend(Ontology.Triples(edge,
                                                    subject=e_subject,
                                                    includeDictionaries=includeDictionaries,
                                                    includeBOT=includeBOT,
                                                    namespacePrefix=namespacePrefix,
                                                    silent=True))
                    sv, ev = Ontology._graph_edge_endpoints(graph, edge, vertex_records=vertex_records)
                    sv_subject = _identity(sv, role="node", fallbackIndex=None, prefix=namespacePrefix) if sv is not None else None
                    ev_subject = _identity(ev, role="node", fallbackIndex=None, prefix=namespacePrefix) if ev is not None else None
                    if sv_subject is not None:
                        triples.append((e_subject, "top:startsAt", sv_subject))
                    if ev_subject is not None:
                        triples.append((e_subject, "top:endsAt", ev_subject))
                    if sv_subject is not None and ev_subject is not None:
                        triples.append((sv_subject, "top:connectsTo", ev_subject))
                        d = Ontology._record_dictionary(edge) or {}
                        predicate = d.get("ontology_predicate") or d.get("ontologyPredicate")
                        inverse = d.get("inverse_predicate") or d.get("inversePredicate")
                        if predicate:
                            pred_q = Ontology.PropertyQName(predicate)
                            if pred_q and not pred_q.startswith("dict:"):
                                triples.append((sv_subject, pred_q, ev_subject))
                        if inverse:
                            inv_q = Ontology.PropertyQName(inverse)
                            if inv_q and not inv_q.startswith("dict:"):
                                triples.append((ev_subject, inv_q, sv_subject))
            seen = set()
            result = []
            for triple in triples:
                if triple not in seen:
                    seen.add(triple)
                    result.append(triple)
            return result
        except Exception as e:
            if not silent:
                print("Ontology.GraphTriples - Error: Could not create graph triples. Returning partial list.")
                print("Error:", e)
            return triples

    Ontology._uri_for_topology = staticmethod(_uri_for_topology)
    Ontology.GraphTriples = staticmethod(_graph_triples)

_Ontology_FinalIdentityFix_Apply()


# -----------------------------------------------------------------------------
# Final triple de-duplication fix
# -----------------------------------------------------------------------------

def _Ontology_FinalTripleFix_Apply():
    _previous_triples = Ontology.Triples

    def _triples_deduped(*args, **kwargs):
        triples = _previous_triples(*args, **kwargs)
        seen = set()
        result = []
        for triple in triples or []:
            try:
                key = tuple(triple)
            except Exception:
                key = repr(triple)
            if key not in seen:
                seen.add(key)
                result.append(triple)
        return result

    Ontology.Triples = staticmethod(_triples_deduped)

_Ontology_FinalTripleFix_Apply()
