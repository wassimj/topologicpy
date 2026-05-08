import uuid
from topologicpy.Dictionary import Dictionary
from topologicpy.Topology import Topology

class BIMElement:
    """
    Base BIM utilities for TopologicPy.
    All BIM entities are Topologic topologies carrying a dictionary with BIM keys.
    """

    @staticmethod
    def SchemaVersion() -> str:
        return "0.1.0"

    @staticmethod
    def DefaultSchema() -> dict:
        # This is the canonical schema (defaults). Treat as read-only.
        return {
            "bim_schema_version": BIMElement.SchemaVersion(),
            "bim_type": "",
            "bim_guid": "",
            "bim_name": "",
            "bim_category": "",
            "bim_level": "",
            "bim_tags": [],
            "bim_properties": {},     # type-specific + free-form properties
            "bim_quantities": {},     # computed quantities (area/vol/len)
            "bim_materials": {},      # slots or layersets
            "bim_relationships": {    # stable relational references
                "host_guid": None,
                "opening_guids": [],
                "space_guids": [],
                "adjacent_space_guids": []
            }
        }

    @staticmethod
    def NewGUID() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _asDict(d):
        # accept Topologic Dictionary or python dict
        if d is None:
            return {}
        if isinstance(d, dict):
            return d
        # best-effort conversion if user provides a Topologic Dictionary
        try:
            return Dictionary.ToPythonDictionary(d)
        except Exception:
            return {}

    @staticmethod
    def Ensure(topology, defaults: dict = None, overwrite: bool = False, silent: bool = False):
        """
        Ensure topology has the BIM schema keys in its dictionary.
        If overwrite is True, defaults overwrite existing values.
        """
        if topology is None:
            if silent:
                return None
            raise ValueError("BIMElement.Ensure: topology is None.")

        base = BIMElement.DefaultSchema()
        if defaults:
            base.update(BIMElement._asDict(defaults))

        d = Topology.Dictionary(topology)
        pd = {}
        try:
            pd = Dictionary.ToPythonDictionary(d) if d is not None else {}
        except Exception:
            pd = {}

        for k, v in base.items():
            if overwrite or (k not in pd):
                pd[k] = v

        # enforce schema version
        pd["bim_schema_version"] = BIMElement.SchemaVersion()

        td = Dictionary.ByPythonDictionary(pd)
        return Topology.SetDictionary(topology, td)

    @staticmethod
    def SetType(topology, bim_type: str):
        topology = BIMElement.Ensure(topology, silent=True)
        d = Dictionary.ToPythonDictionary(Topology.Dictionary(topology))
        d["bim_type"] = str(bim_type)
        return Topology.SetDictionary(topology, Dictionary.ByPythonDictionary(d))

    @staticmethod
    def Type(topology) -> str:
        d = Topology.Dictionary(topology)
        if d is None:
            return ""
        v = Dictionary.ValueAtKey(d, "bim_type")
        return "" if v is None else str(v)

    @staticmethod
    def SetGUID(topology, guid: str = None):
        topology = BIMElement.Ensure(topology, silent=True)
        d = Dictionary.ToPythonDictionary(Topology.Dictionary(topology))
        d["bim_guid"] = str(guid) if guid else BIMElement.NewGUID()
        return Topology.SetDictionary(topology, Dictionary.ByPythonDictionary(d))

    @staticmethod
    def GUID(topology) -> str:
        d = Topology.Dictionary(topology)
        if d is None:
            return ""
        v = Dictionary.ValueAtKey(d, "bim_guid")
        return "" if v is None else str(v)

    @staticmethod
    def SetName(topology, name: str):
        topology = BIMElement.Ensure(topology, silent=True)
        d = Dictionary.ToPythonDictionary(Topology.Dictionary(topology))
        d["bim_name"] = str(name)
        return Topology.SetDictionary(topology, Dictionary.ByPythonDictionary(d))

    @staticmethod
    def SetProperty(topology, key: str, value, overwrite: bool = True):
        topology = BIMElement.Ensure(topology, silent=True)
        pd = Dictionary.ToPythonDictionary(Topology.Dictionary(topology))
        props = pd.get("bim_properties", {})
        if overwrite or (key not in props):
            props[str(key)] = value
        pd["bim_properties"] = props
        return Topology.SetDictionary(topology, Dictionary.ByPythonDictionary(pd))

    @staticmethod
    def GetProperty(topology, key: str, default=None):
        d = Topology.Dictionary(topology)
        if d is None:
            return default
        pd = Dictionary.ToPythonDictionary(d)
        props = pd.get("bim_properties", {})
        return props.get(str(key), default)

    @staticmethod
    def BoundingBox(topology):
        # For future reference: correct method is Topology.BoundingBox
        return Topology.BoundingBox(topology)
