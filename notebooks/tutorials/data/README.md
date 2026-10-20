# Tutorial data

`one_room.ifc` is a synthetic IFC4 STEP fixture authored for these tutorials. It contains a project/site/building/storey hierarchy and one 4 × 3 × 3 m space in metre units. It contains no external project data. Reading it in tutorial 12 uses TopologicPy’s own IFC importer and does not require IfcOpenShell.

`energy/office.osm`, `energy/london.epw`, and `energy/london.ddy` are copies of the existing repository assets in `assets/EnergyModel`: `OSMTemplate-OfficeBuilding-3.10.0.osm` and the `GBR_London.Gatwick.037760_IWEC` files. Their existing provenance and terms apply. The template is an educational starting point; its assumptions are not a validated specification for a real building.
