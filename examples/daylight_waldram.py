"""Run a preliminary before/after daylight comparison in model units of metres.

Usage: python examples/daylight_waldram.py --output <directory>
Uses only TopologicPy and Python standard-library file handling.
"""
import argparse
import json
from pathlib import Path
from topologicpy.Daylight import Daylight
from topologicpy.Cell import Cell
from topologicpy.Face import Face
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


def main(output):
    cell = Cell.Prism(Vertex.ByCoordinates(0, 0, 0), width=4, length=4,
                      height=3, placement='lowerleft')
    window = Face.ByVertices([Vertex.ByCoordinates(*p) for p in
                              [(1, 0, 1), (3, 0, 1), (3, 0, 2.5), (1, 0, 2.5)]])
    cell = Topology.AddApertures(cell, [window], subTopologyType='face')
    room = Daylight.PrepareRoom(cell)
    grid = Daylight.SensorGrid(Daylight.WorkingPlane(cell), spacing=.3)
    obstruction = Face.ByVertices([Vertex.ByCoordinates(*p) for p in
                                   [(-5, -1, 0), (9, -1, 0), (9, -1, 9), (-5, -1, 9)]])
    before = Daylight.Waldram(room, Daylight.PrepareScene([]), grid=grid)
    after = Daylight.Waldram(room, Daylight.PrepareScene([obstruction]), grid=grid)
    comparison = Daylight.Compare(before, after)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, result in [('before', before), ('after', after)]:
        (output / (name + '.json')).write_text(
            json.dumps(Daylight.ResultData(result, includeGeometry=True), indent=2,
                       allow_nan=False), encoding='utf-8')
    (output / 'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'sensors': len(grid['vertices']), 'comparison': comparison}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='daylight-results')
    main(parser.parse_args().output)
