# Copyright (C) 2024
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

import os
import warnings

try:
    import honeybee.facetype
    from honeybee.face import Face as HBFace
    from honeybee.model import Model as HBModel
    from honeybee.room import Room as HBRoom
    from honeybee.shade import Shade as HBShade
    from honeybee.aperture import Aperture as HBAperture
    from honeybee.door import Door as HBDoor
except:
    print("Honeybee - Installing required honeybee library.")
    try:
        os.system("pip install honeybee")
    except:
        os.system("pip install honeybee --user")
    try:
        import honeybee.facetype
        from honeybee.face import Face as HBFace
        from honeybee.model import Model as HBModel
        from honeybee.room import Room as HBRoom
        from honeybee.shade import Shade as HBShade
        from honeybee.aperture import Aperture as HBAperture
        from honeybee.door import Door as HBDoor
    except:
        warnings.warn("Honeybee - ERROR: Could not import honeybee")

try:
    import honeybee_energy.lib.constructionsets as constr_set_lib
    import honeybee_energy.lib.programtypes as prog_type_lib
    import honeybee_energy.lib.scheduletypelimits as schedule_types
    from honeybee_energy.schedule.ruleset import ScheduleRuleset
    from honeybee_energy.schedule.day import ScheduleDay
    from honeybee_energy.load.setpoint import Setpoint
    from honeybee_energy.load.hotwater import  ServiceHotWater
except:
    print("Honeybee - Installing required honeybee-energy library.")
    try:
        os.system("pip install -U honeybee-energy[standards]")
    except:
        os.system("pip install -U honeybee-energy[standards] --user")
    try:
        import honeybee_energy.lib.constructionsets as constr_set_lib
        import honeybee_energy.lib.programtypes as prog_type_lib
        import honeybee_energy.lib.scheduletypelimits as schedule_types
        from honeybee_energy.schedule.ruleset import ScheduleRuleset
        from honeybee_energy.schedule.day import ScheduleDay
        from honeybee_energy.load.setpoint import Setpoint
        from honeybee_energy.load.hotwater import  ServiceHotWater
    except:
        warnings.warn("Honeybee - Error: Could not import honeybee-energy")

try:
    from honeybee_radiance.sensorgrid import SensorGrid
except:
    print("Honeybee - Installing required honeybee-radiance library.")
    try:
        os.system("pip install -U honeybee-radiance")
    except:
        os.system("pip install -U honeybee-radiance --user")
    try:
        from honeybee_radiance.sensorgrid import SensorGrid
    except:
        warnings.warn("Honeybee - Error: Could not import honeybee-radiance")

try:
    from ladybug.dt import Time
except:
    print("Honeybee - Installing required ladybug library.")
    try:
        os.system("pip install -U ladybug")
    except:
        os.system("pip install -U ladybug --user")
    try:
        from ladybug.dt import Time
    except:
        warnings.warn("Honeybee - Error: Could not import ladybug")

try:
    from ladybug_geometry.geometry3d.face import Face3D
    from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
except:
    print("Honeybee - Installing required ladybug-geometry library.")
    try:
        os.system("pip install -U ladybug-geometry")
    except:
        os.system("pip install -U ladybug-geometry --user")
    try:
        from ladybug_geometry.geometry3d.face import Face3D
        from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
    except:
        warnings.warn("Honeybee - Error: Could not import ladybug-geometry")

import json
import topologic_core as topologic

class Honeybee:
    @staticmethod
    def ConstructionSetByIdentifier(id):
        """
        Returns the built-in construction set by the input identifying string.

        Parameters
        ----------
        id : str
            The construction set identifier.

        Returns
        -------
        HBConstructionSet
            The found built-in construction set.

        """
        return constr_set_lib.construction_set_by_identifier(id)
    
    @staticmethod
    def ConstructionSets():
        """
        Returns the list of built-in construction sets

        Returns
        -------
        list
            The list of built-in construction sets.

        """
        constrSets = []
        constrIdentifiers = list(constr_set_lib.CONSTRUCTION_SETS)
        for constrIdentifier in constrIdentifiers: 
            constrSets.append(constr_set_lib.construction_set_by_identifier(constrIdentifier))
        return [constrSets, constrIdentifiers]
    
    @staticmethod
    def ExportToHBJSON(model, path, overwrite=False):
        """
        Exports the input HB Model to a file.

        Parameters
        ----------
        model : HBModel
            The input HB Model.
        path : str
            The location of the output file.
        overwrite : bool , optional
            If set to True this method overwrites any existing file. Otherwise, it won't. The default is False.

        Returns
        -------
        bool
            Returns True if the operation is successful. Returns False otherwise.

        """
        from os.path import exists

        # Make sure the file extension is .hbjson
        ext = path[len(path)-7:len(path)]
        if ext.lower() != ".hbjson":
            path = path+".hbjson"
        
        if not overwrite and exists(path):
            print("DGL.ExportToHBJSON - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        f = None
        try:
            if overwrite == True:
                f = open(path, "w")
            else:
                f = open(path, "x") # Try to create a new File
        except:
            print("DGL.ExportToHBJSON - Error: Could not create a new file at the following location: "+path+". Returning None.")
            return None
        if (f):
            json.dump(model.to_dict(), f, indent=4)
            f.close()    
            return True
        return False
    
    @staticmethod
    def ModelByTopology(tpBuilding,
                tpShadingFacesCluster = None,
                buildingName: str = "Generic_Building",
                defaultProgramIdentifier: str = "Generic Office Program",
                defaultConstructionSetIdentifier: str = "Default Generic Construction Set",
                coolingSetpoint: float = 25.0,
                heatingSetpoint: float = 20.0,
                humidifyingSetpoint: float = 30.0,
                dehumidifyingSetpoint: float = 55.0,
                roomNameKey: str = "TOPOLOGIC_name",
                roomTypeKey: str = "TOPOLOGIC_type",
                apertureTypeKey: str = "TOPOLOGIC_type",
                addSensorGrid: bool = False,
                mantissa: int = 6):
        """
        Creates an HB Model from the input Topology.

        Parameters
        ----------
        tpBuilding : topologic_core.CellComplex or topologic_core.Cell
            The input building topology.
        tpShadingFaceCluster : topologic_core.Cluster , optional
            The input cluster for shading faces. The default is None.
        buildingName : str , optional
            The desired name of the building. The default is "Generic_Building".
        defaultProgramIdentifier: str , optional
            The desired default program identifier. The default is "Generic Office Program".
        defaultConstructionSetIdentifier: str , optional
            The desired default construction set identifier. The default is "Default Generic Construction Set".
        coolingSetpoint : float , optional
            The desired HVAC cooling set point in degrees Celsius. The default is 25.
        heatingSetpoint : float , optional
            The desired HVAC heating set point in degrees Celsius. The default is 20.
        humidifyingSetpoint : float , optional
            The desired HVAC humidifying set point in percentage. The default is 55.
        roomNameKey : str , optional
            The dictionary key under which the room name is stored. The default is "TOPOLOGIC_name".
        roomTypeKey : str , optional
            The dictionary key under which the room type is stored. The default is "TOPOLOGIC_type".
        apertureTypeKey : str , optional
            The dictionary key under which the aperture type is stored. The default is "TOPOLOGIC_type".
        addSensorGrid : bool , optional
            If set to True a sensor grid is add to horizontal surfaces. The default is False.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        
        Returns
        -------
        HBModel
            The created HB Model

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Aperture import Aperture
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        def cellFloor(cell):
            faces = Topology.Faces(cell)
            c = [Vertex.Z(Topology.CenterOfMass(x)) for x in faces]
            return round(min(c),2)

        def floorLevels(cells, min_difference):
            floors = [cellFloor(x) for x in cells]
            floors = list(set(floors)) #create a unique list
            floors.sort()
            returnList = []
            for aCell in cells:
                for floorNumber, aFloor in enumerate(floors):
                    if abs(cellFloor(aCell) - aFloor) > min_difference:
                        continue
                    returnList.append("Floor"+str(floorNumber).zfill(2))
                    break
            return returnList

        def getKeyName(d, keyName):
            if not d == None:
                keys = Dictionary.Keys(d)
            else:
                keys = []
            for key in keys:
                if key.lower() == keyName.lower():
                    return key
            return None

        def createUniqueName(name, nameList, number):
            if not (name in nameList):
                return name
            elif not ((name+"_"+str(number)) in nameList):
                return name+"_"+str(number)
            else:
                return createUniqueName(name,nameList, number+1)
        
        if not Topology.IsInstance(tpBuilding, "Topology"):
            return None
        rooms = []
        tpCells = Topology.Cells(tpBuilding)
        # Sort cells by Z Levels
        tpCells.sort(key=lambda c: cellFloor(c), reverse=False)
        fl = floorLevels(tpCells, 2)
        spaceNames = []
        sensorGrids = []
        for spaceNumber, tpCell in enumerate(tpCells):
            tpDictionary = Topology.Dictionary(tpCell)
            tpCellName = "Untitled Space"
            tpCellStory = fl[0]
            tpCellProgramIdentifier = None
            tpCellConstructionSetIdentifier = None
            tpCellConditioned = True
            if tpDictionary:
                keyName = getKeyName(tpDictionary, 'Story')
                tpCellStory = Dictionary.ValueAtKey(tpDictionary, keyName, silent=True) or fl[spaceNumber]
                if tpCellStory:
                    tpCellStory = tpCellStory.replace(" ","_")
                else:
                    tpCellStory = "Untitled_Floor"
                if roomNameKey:
                    keyName = getKeyName(tpDictionary, roomNameKey)
                else:
                    keyName = getKeyName(tpDictionary, 'Name')
                tpCellName = Dictionary.ValueAtKey(tpDictionary,keyName, silent=True) or createUniqueName(tpCellStory+"_SPACE_"+(str(spaceNumber+1)), spaceNames, 1)
                if roomTypeKey:
                    keyName = getKeyName(tpDictionary, roomTypeKey)
                try:
                    tpCellProgramIdentifier = Dictionary.ValueAtKey(tpDictionary, keyName, silent=True)
                    if tpCellProgramIdentifier:
                        program = prog_type_lib.program_type_by_identifier(tpCellProgramIdentifier)
                    elif defaultProgramIdentifier:
                        program = prog_type_lib.program_type_by_identifier(defaultProgramIdentifier)
                except:
                    program = prog_type_lib.office_program #Default Office Program as a last resort
                keyName = getKeyName(tpDictionary, 'construction_set')
                try:
                    tpCellConstructionSetIdentifier = Dictionary.ValueAtKey(tpDictionary, keyName, silent=True)
                    if tpCellConstructionSetIdentifier:
                        constr_set = constr_set_lib.construction_set_by_identifier(tpCellConstructionSetIdentifier)
                    elif defaultConstructionSetIdentifier:
                        constr_set = constr_set_lib.construction_set_by_identifier(defaultConstructionSetIdentifier)
                except:
                    constr_set = constr_set_lib.construction_set_by_identifier("Default Generic Construction Set")
            else:
                tpCellStory = fl[spaceNumber]
                tpCellName = str(tpCellStory)+"_SPACE_"+(str(spaceNumber+1))
                program = prog_type_lib.office_program
                constr_set = constr_set_lib.construction_set_by_identifier("Default Generic Construction Set")
            spaceNames.append(tpCellName)

            tpCellFaces = Topology.Faces(tpCell)
            if tpCellFaces:
                hbRoomFaces = []
                for tpFaceNumber, tpCellFace in enumerate(tpCellFaces):
                    tpCellFaceNormal = Face.Normal(tpCellFace, mantissa=mantissa)
                    hbRoomFacePoints = []
                    tpFaceVertices = Topology.Vertices(Face.ExternalBoundary(tpCellFace))
                    for tpVertex in tpFaceVertices:
                        hbRoomFacePoints.append(Point3D(Vertex.X(tpVertex, mantissa=mantissa), Vertex.Y(tpVertex, mantissa=mantissa), Vertex.Z(tpVertex, mantissa=mantissa)))
                    hbRoomFace = HBFace(tpCellName+'_Face_'+str(tpFaceNumber+1), Face3D(hbRoomFacePoints))
                    tpFaceApertures = []
                    tpFaceApertures = Topology.Apertures(tpCellFace)
                    if tpFaceApertures:
                        for tpFaceApertureNumber, apertureTopology in enumerate(tpFaceApertures):
                            tpFaceApertureDictionary = Topology.Dictionary(apertureTopology)
                            if tpFaceApertureDictionary:
                                apertureKeyName = getKeyName(tpFaceApertureDictionary, apertureTypeKey)
                                tpFaceApertureType = Dictionary.ValueAtKey(tpFaceApertureDictionary,apertureKeyName, silent=True)
                            hbFaceAperturePoints = []
                            tpFaceApertureVertices = []
                            tpFaceApertureVertices = Topology.Vertices(Face.ExternalBoundary(apertureTopology))
                            for tpFaceApertureVertex in tpFaceApertureVertices:
                                hbFaceAperturePoints.append(Point3D(Vertex.X(tpFaceApertureVertex, mantissa=mantissa), Vertex.Y(tpFaceApertureVertex, mantissa=mantissa), Vertex.Z(tpFaceApertureVertex, mantissa=mantissa)))
                            if(tpFaceApertureType):
                                if ("door" in tpFaceApertureType.lower()):
                                    hbFaceAperture = HBDoor(tpCellName+'_Face_'+str(tpFaceNumber+1)+'_Door_'+str(tpFaceApertureNumber), Face3D(hbFaceAperturePoints))
                                else:
                                    hbFaceAperture = HBAperture(tpCellName+'_Face_'+str(tpFaceNumber+1)+'_Window_'+str(tpFaceApertureNumber), Face3D(hbFaceAperturePoints))
                            else:
                                hbFaceAperture = HBAperture(tpCellName+'_Face_'+str(tpFaceNumber+1)+'_Window_'+str(tpFaceApertureNumber), Face3D(hbFaceAperturePoints))
                            hbRoomFace.add_aperture(hbFaceAperture)
                    else:
                        tpFaceDictionary = Topology.Dictionary(tpCellFace)
                        if (abs(tpCellFaceNormal[2]) < 1e-6) and tpFaceDictionary: #It is a mostly vertical wall and has a dictionary
                            apertureRatio = Dictionary.ValueAtKey(tpFaceDictionary,'apertureRatio', silent=True)
                            if apertureRatio:
                                hbRoomFace.apertures_by_ratio(apertureRatio, tolerance=0.01)
                    fType = honeybee.facetype.get_type_from_normal(Vector3D(tpCellFaceNormal[0],tpCellFaceNormal[1],tpCellFaceNormal[2]), roof_angle=30, floor_angle=150)
                    hbRoomFace.type = fType
                    hbRoomFaces.append(hbRoomFace)
                room = HBRoom(tpCellName, hbRoomFaces, 0.01, 1)
                if addSensorGrid:
                    floor_mesh = room.generate_grid(0.5, 0.5, 1)
                    sensorGrids.append(SensorGrid.from_mesh3d(tpCellName+"_SG", floor_mesh))
                heat_setpt = ScheduleRuleset.from_constant_value('Room Heating', heatingSetpoint, schedule_types.temperature)
                cool_setpt = ScheduleRuleset.from_constant_value('Room Cooling', coolingSetpoint, schedule_types.temperature)
                humidify_setpt = ScheduleRuleset.from_constant_value('Room Humidifying', humidifyingSetpoint, schedule_types.humidity)
                dehumidify_setpt = ScheduleRuleset.from_constant_value('Room Dehumidifying', dehumidifyingSetpoint, schedule_types.humidity)
                setpoint = Setpoint('Room Setpoint', heat_setpt, cool_setpt, humidify_setpt, dehumidify_setpt)
                simple_office = ScheduleDay('Simple Weekday', [0, 1, 0], [Time(0, 0), Time(9, 0), Time(17, 0)]) #Todo: Remove hardwired scheduleday
                schedule = ScheduleRuleset('Office Water Use', simple_office, None, schedule_types.fractional) #Todo: Remove hardwired schedule
                shw = ServiceHotWater('Office Hot Water', 0.1, schedule) #Todo: Remove hardwired schedule hot water
                room.properties.energy.program_type = program
                room.properties.energy.construction_set = constr_set
                room.properties.energy.add_default_ideal_air() #Ideal Air Exchange
                room.properties.energy.setpoint = setpoint #Heating/Cooling/Humidifying/Dehumidifying
                room.properties.energy.service_hot_water = shw #Service Hot Water
                if tpCellStory:
                    room.story = tpCellStory
                rooms.append(room)
        HBRoom.solve_adjacency(rooms, 0.01)

        hbShades = []
        if(tpShadingFacesCluster):
            hbShades = []
            tpShadingFaces = Topology.SubTopologies(tpShadingFacesCluster, subTopologyType="face")
            for faceIndex, tpShadingFace in enumerate(tpShadingFaces):
                faceVertices = []
                faceVertices = Topology.Vertices(Face.ExternalBoundary(tpShadingFace))
                facePoints = []
                for aVertex in faceVertices:
                    facePoints.append(Point3D(Vertex.X(aVertex, mantissa=mantissa), Vertex.Y(aVertex, mantissa=mantissa), Vertex.Z(aVertex, mantissa=mantissa)))
                hbShadingFace = Face3D(facePoints, None, [])
                hbShade = HBShade("SHADINGSURFACE_" + str(faceIndex+1), hbShadingFace)
                hbShades.append(hbShade)
        model = HBModel(buildingName, rooms, orphaned_shades=hbShades)
        if addSensorGrid:
            model.properties.radiance.sensor_grids = []
            model.properties.radiance.add_sensor_grids(sensorGrids)
        return model
    
    @staticmethod
    def ProgramTypeByIdentifier(id):
        """
        Returns the program type by the input identifying string.

        Parameters
        ----------
        id : str
            The identifiying string.

        Returns
        -------
        HBProgram
            The found built-in program.

        """
        return prog_type_lib.program_type_by_identifier(id)
    
    @staticmethod
    def ProgramTypes():
        """
        Returns the list of available built-in program types.

        Returns
        -------
        list
            The list of available built-in program types.

        """
        progTypes = []
        progIdentifiers = list(prog_type_lib.PROGRAM_TYPES)
        for progIdentifier in progIdentifiers: 
            progTypes.append(prog_type_lib.program_type_by_identifier(progIdentifier))
        return [progTypes, progIdentifiers]
    
    @staticmethod
    def String(model):
        """
        Returns the string representation of the input model.

        Parameters
        ----------
        model : HBModel
            The input HB Model.

        Returns
        -------
        dict
            A dictionary representing the input HB Model.

        """
        return model.to_dict()