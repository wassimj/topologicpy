# Depthmap reference fixture

`barnsbury_hh.json` contains original stored outputs from SpaceGroupUCL/depthmapX
`testdata/barnsbury_axial.graph`, retrieved at commit
`6ea9755efe30d51603b99b842898b62a2591b850`.
The source file identifies its producing version as `depthmapX v0.27.b`.
The JSON includes its URL and SHA-256, 61 vertex identifiers, 116 undirected
connections, and saved Integration [HH], Mean Depth and Node Count values.

The values were read from the binary attribute table, not computed by
TopologicPy. Connections were read from the connector records, with degrees
checked against saved Connectivity and adjacency checked for symmetry.

This validates global HH integration against an actual saved Depthmap result,
using float32-aware tolerances. It does not certify every current DepthmapX mode
or local-radius output. Current source HH definitions and small hand-calculated
radius cases are covered separately. Tests require no network access, Depthmap
executable or NetworkX installation.


`barnsbury_segment_connections.json` contains 178 original segments and 746
oriented transitions from `testdata/barnsbury_segment.graph` at the same pinned
commit. The geometry comes from its binary SalaShape records; outgoing endpoint
lists, arrival direction and float32 angular weights come from Connector records.
Every transition was checked to join the source exit to the target entry within
1e-5 coordinate units. Saved Connectivity and Angular Connectivity independently
check the count and sum of transition costs per segment. Tests compare the
TopologicPy endpoint-state graph against every original transition and weight.

This segment file has no saved angular choice or integration columns. The fixture
validates physical connectivity, orientation and quarter-turn units, **not**
complete angular choice/integration parity. Those analyses also have independent
small-network route-enumeration tests. TopologicPy's AngularIntegration is
closeness with a reachable-fraction correction; it is not Depthmap angular
integration or NAIN. Choice uses exact angles with a documented minimum-hop rule
for zero-cost ties; it is not the discretized Tulip algorithm or NACH.

AngularConnectivity(method="depthmap") sums these turn costs and is checked
against all 178 saved Angular Connectivity values. Its default method="degree"
preserves the previous unweighted neighbour-count behaviour.


`barnsbury_extended_tulip_reference.json` contains 3,434 original segments,
explicit oriented connections, and saved DepthmapX 0.9.1 T1024 outputs for
length-weighted metric R100, unweighted metric R250, and unweighted topological
R4. Executable, input graph, and exported CSV hashes are recorded in the file.
Tests import this committed fixture directly and compare Choice, Integration,
and Node Count without requiring local outputs, network access, or DepthmapX.

`tulip_float32_regression.json` contains frozen pre-optimisation engine outputs
for a synthetic diamond with integral, fractional, and large lengths around the
float32 shortcut boundary. These preserve queue tie order and weighted arithmetic;
they are regression outcomes, not independent DepthmapX reference outputs.
