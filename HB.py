import topologic
import math
import honeybee_energy.lib.constructionsets as constr_set_lib
import honeybee.facetype
from honeybee.face import Face
from ladybug_geometry.geometry3d.face import Face3D
from honeybee.model import Model
from honeybee.room import Room
from honeybee.shade import Shade
from honeybee.aperture import Aperture
from honeybee.door import Door
from honeybee_energy.schedule.ruleset import ScheduleRuleset
from honeybee_energy.schedule.day import ScheduleDay
from honeybee_energy.load.setpoint import Setpoint
from honeybee_energy.load.hotwater import  ServiceHotWater
import honeybee_energy.lib.programtypes as prog_type_lib
import honeybee_energy.lib.scheduletypelimits as schedule_types
from ladybug.dt import Time
from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
import json
import Dictionary

class HB:
    @staticmethod
    def HBConstructionSetByIdentifier(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return constr_set_lib.construction_set_by_identifier(item)
    
    @staticmethod
    def HBConstructionSets():
        """
        Returns
        -------
        list
            DESCRIPTION.

        """
        constrSets = []
        constrIdentifiers = list(constr_set_lib.CONSTRUCTION_SETS)
        for constrIdentifier in constrIdentifiers: 
            constrSets.append(constr_set_lib.construction_set_by_identifier(constrIdentifier))
        return [constrSets, constrIdentifiers]
    
    @staticmethod
    def HBJSONByTopology(osModel, weatherFilePath, designDayFilePath, tpBuilding,
                         tpShadingSurfacesCluster, floorLevels, buildingName,
                         buildingType, defaultSpaceType, northAxis, glazingRatio, coolingTemp, heatingTemp):
        """
        Parameters
        ----------
        osModel : TYPE
            DESCRIPTION.
        weatherFilePath : TYPE
            DESCRIPTION.
        designDayFilePath : TYPE
            DESCRIPTION.
        tpBuilding : TYPE
            DESCRIPTION.
        tpShadingSurfacesCluster : TYPE
            DESCRIPTION.
        floorLevels : TYPE
            DESCRIPTION.
        buildingName : TYPE
            DESCRIPTION.
        buildingType : TYPE
            DESCRIPTION.
        defaultSpaceType : TYPE
            DESCRIPTION.
        northAxis : TYPE
            DESCRIPTION.
        glazingRatio : TYPE
            DESCRIPTION.
        coolingTemp : TYPE
            DESCRIPTION.
        heatingTemp : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # osModel = item[0]
        # weatherFilePath = item[1]
        # designDayFilePath = item[2]
        # tpBuilding = item[3]
        # tpShadingSurfacesCluster = item[4]
        # floorLevels = item[5]
        # buildingName = item[6]
        # buildingType = item[7]
        # defaultSpaceType = item[8]
        # northAxis = item[9]
        # glazingRatio = item[10]
        # coolingTemp = item[11]
        # heatingTemp = item[12]
        
        def listAttributeValues(listAttribute):
            listAttributes = listAttribute.ListValue()
            returnList = []
            for attr in listAttributes:
                if isinstance(attr, topologic.IntAttribute):
                    returnList.append(attr.IntValue())
                elif isinstance(attr, topologic.DoubleAttribute):
                    returnList.append(attr.DoubleValue())
                elif isinstance(attr, topologic.StringAttribute):
                    returnList.append(attr.StringValue())
            return returnList

        def valueAtKey(item, key):
            try:
                attr = item.ValueAtKey(key)
            except:
                raise Exception("Dictionary.ValueAtKey - Error: Could not retrieve a Value at the specified key ("+key+")")
            if isinstance(attr, topologic.IntAttribute):
                return (attr.IntValue())
            elif isinstance(attr, topologic.DoubleAttribute):
                return (attr.DoubleValue())
            elif isinstance(attr, topologic.StringAttribute):
                return (attr.StringValue())
            elif isinstance(attr, topologic.ListAttribute):
                return (listAttributeValues(attr))
            else:
                return None

        rooms = []
        tpCells = []
        _ = tpBuilding.Cells(None, tpCells)
        # Sort cells by Z Levels
        tpCells.sort(key=lambda c: c.CenterOfMass().Z(), reverse=False)
        for spaceNumber, tpCell in enumerate(tpCells):
            tpDictionary = tpCell.GetDictionary()
            tpCellName = None
            tpCellStory = None
            if tpDictionary:
                tpCellName = valueAtKey(tpDictionary,'name')
                tpCellStory = valueAtKey(tpDictionary,'story')            
            tpCellFaces = []
            _ = tpCell.Faces(None, tpCellFaces)
            if tpCellFaces:
                hbRoomFaces = []
                for tpFaceNumber, tpCellFace in enumerate(tpCellFaces):
                    hbRoomFacePoints = []
                    tpFaceVertices = []
                    _ = tpCellFace.ExternalBoundary().Vertices(None, tpFaceVertices)
                    for tpVertex in tpFaceVertices:
                        hbRoomFacePoints.append(Point3D(tpVertex.X(), tpVertex.Y(), tpVertex.Z()))
                    hbRoomFace = Face(tpCellName+'_Face_'+str(tpFaceNumber+1), Face3D(hbRoomFacePoints))
                    faceNormal = topologic.FaceUtility.NormalAtParameters(tpFace, 0.5, 0.5)
                    ang = math.degrees(math.acos(faceNormal.dot([0, 0, 1])))
                    print("HBJSONByTopology: Angle between face normal and UP",ang)
                    if ang > 175:
                        hbRoomFace.type = "floor"
                    tpFaceApertures = []
                    _ = tpCellFace.Apertures(tpFaceApertures)
                    if tpFaceApertures:
                        for tpFaceApertureNumber, tpFaceAperture in enumerate(tpFaceApertures):
                            apertureTopology = topologic.Aperture.Topology(tpFaceAperture)
                            tpFaceApertureDictionary = apertureTopology.GetDictionary()
                            if tpFaceApertureDictionary:
                                tpFaceApertureType = valueAtKey(tpFaceApertureDictionary,'type')
                            hbFaceAperturePoints = []
                            tpFaceApertureVertices = []
                            _ = apertureTopology.ExternalBoundary().Vertices(None, tpFaceApertureVertices)
                            for tpFaceApertureVertex in tpFaceApertureVertices:
                                hbFaceAperturePoints.append(Point3D(tpFaceApertureVertex.X(), tpFaceApertureVertex.Y(), tpFaceApertureVertex.Z()))
                            if(tpFaceApertureType):
                                if ("door" in tpFaceApertureType.lower()):
                                    hbFaceAperture = Door(tpCellName+'_Face_'+str(tpFaceNumber+1)+'_Door_'+str(tpFaceApertureNumber), Face3D(hbFaceAperturePoints))
                                else:
                                    hbFaceAperture = Aperture(tpCellName+'_Face_'+str(tpFaceNumber+1)+'_Window_'+str(tpFaceApertureNumber), Face3D(hbFaceAperturePoints))
                            else:
                                hbFaceAperture = Aperture(tpCellName+'_Face_'+str(tpFaceNumber+1)+'_Window_'+str(tpFaceApertureNumber), Face3D(hbFaceAperturePoints))
                            hbRoomFace.add_aperture(hbFaceAperture)
                    hbRoomFaces.append(hbRoomFace)
                if tpCellName == None:
                    tpCellName = "GENERICROOM_"+(str(spaceNumber+1))
                room = Room(tpCellName, hbRoomFaces, 0.01, 1) #ToDo: Figure out how to add Story number
                heat_setpt = ScheduleRuleset.from_constant_value('Room Heating', heatingTemp, schedule_types.temperature)
                cool_setpt = ScheduleRuleset.from_constant_value('Room Cooling', coolingTemp, schedule_types.temperature)
                humidify_setpt = ScheduleRuleset.from_constant_value('Room Humidifying', 30, schedule_types.humidity) #Todo: Remove hardwired number
                dehumidify_setpt = ScheduleRuleset.from_constant_value('Room Dehumidifying', 55, schedule_types.humidity) #Todo: Remove hardwired number
                setpoint = Setpoint('Room Setpoint', heat_setpt, cool_setpt, humidify_setpt, dehumidify_setpt)
                simple_office = ScheduleDay('Simple Weekday', [0, 1, 0], [Time(0, 0), Time(9, 0), Time(17, 0)])
                schedule = ScheduleRuleset('Office Water Use', simple_office, None, schedule_types.fractional)
                shw = ServiceHotWater('Office Hot Water', 0.1, schedule)
                room.properties.energy.program_type = prog_type_lib.office_program #Default Office Program
                room.properties.energy.add_default_ideal_air() #Ideal Air Exchange
                room.properties.energy.setpoint = setpoint #Heating/Cooling/Humidifying/Dehumidifying
                room.properties.energy.service_hot_water = shw #Service Hot Water
                if tpCellStory:
                    room.story = tpCellStory
                rooms.append(room)
        Room.solve_adjacency(rooms, 0.01)
        Room.stories_by_floor_height(rooms, min_difference=2.0)

        hbShades = []
        shadingFaces = []
        _ = tpShadingSurfacesCluster.Faces(None, shadingFaces)
        for faceIndex, shadingFace in enumerate(shadingFaces):
            faceVertices = []
            _ = shadingFace.ExternalBoundary().Vertices(None, faceVertices)
            facePoints = []
            for aVertex in faceVertices:
                facePoints.append(Point3D(aVertex.X(), aVertex.Y(), aVertex.Z()))
            hbShadingFace = Face3D(facePoints, None, [])
            hbShade = Shade("SHADINGSURFACE_" + str(faceIndex), hbShadingFace)
            hbShades.append(hbShade)
        model = Model('TopologicModel', rooms, orphaned_shades=hbShades)
        return model.to_dict()
    
    @staticmethod
    def HBModelByTopology(tpBuilding, tpShadingFacesCluster,
                          buildingName, defaultProgramIdentifier, defaultConstructionSetIdentifier,
                          coolingSetpoint, heatingSetpoint, humidifyingSetpoint, dehumidifyingSetpoint,
                          roomNameKey, roomTypeKey):
        """
        Parameters
        ----------
        tpBuilding : TYPE
            DESCRIPTION.
        tpShadingFacesCluster : TYPE
            DESCRIPTION.
        buildingName : TYPE
            DESCRIPTION.
        defaultProgramIdentifier : TYPE
            DESCRIPTION.
        defaultConstructionSetIdentifier : TYPE
            DESCRIPTION.
        coolingSetpoint : TYPE
            DESCRIPTION.
        heatingSetpoint : TYPE
            DESCRIPTION.
        humidifyingSetpoint : TYPE
            DESCRIPTION.
        dehumidifyingSetpoint : TYPE
            DESCRIPTION.
        roomNameKey : TYPE
            DESCRIPTION.
        roomTypeKey : TYPE
            DESCRIPTION.

        Returns
        -------
        model : TYPE
            DESCRIPTION.

        """
        # tpBuilding = item[0]
        # tpShadingFacesCluster = item[1]
        # buildingName = item[2]
        # defaultProgramIdentifier = item[3]
        # defaultConstructionSetIdentifier = item[4]
        # coolingSetpoint = item[5]
        # heatingSetpoint = item[6]
        # humidifyingSetpoint = item[7]
        # dehumidifyingSetpoint = item[8]
        # roomNameKey = item[9]
        # roomTypeKey = item[10]
        
        def cellFloor(cell):
            faces = []
            _ = cell.Faces(None, faces)
            c = [x.CenterOfMass().Z() for x in faces]
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
            keys = d.Keys()
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
        
        if buildingName:
            buildingName = buildingName.replace(" ","_")
        else:
            buildingName = "GENERICBUILDING"
        rooms = []
        tpCells = []
        _ = tpBuilding.Cells(None, tpCells)
        # Sort cells by Z Levels
        tpCells.sort(key=lambda c: cellFloor(c), reverse=False)
        fl = floorLevels(tpCells, 2)
        spaceNames = []
        for spaceNumber, tpCell in enumerate(tpCells):
            tpDictionary = tpCell.GetDictionary()
            tpCellName = None
            tpCellStory = None
            tpCellProgramIdentifier = None
            tpCellConstructionSetIdentifier = None
            tpCellConditioned = True
            if tpDictionary:
                keyName = getKeyName(tpDictionary, 'Story')
                tpCellStory = Dictionary.DictionaryValueAtKey(tpDictionary, keyName)
                if tpCellStory:
                    tpCellStory = tpCellStory.replace(" ","_")
                else:
                    tpCellStory = fl[spaceNumber]
                if roomNameKey:
                    keyName = getKeyName(tpDictionary, roomNameKey)
                else:
                    keyName = getKeyName(tpDictionary, 'Name')
                tpCellName = Dictionary.DictionaryValueAtKey(tpDictionary,keyName)
                if tpCellName:
                    tpCellName = createUniqueName(tpCellName.replace(" ","_"), spaceNames, 1)
                else:
                    tpCellName = tpCellStory+"_SPACE_"+(str(spaceNumber+1))
                if roomTypeKey:
                    keyName = getKeyName(tpDictionary, roomTypeKey)
                else:
                    keyName = getKeyName(tpDictionary, 'Program')
                tpCellProgramIdentifier = Dictionary.DictionaryValueAtKey(tpDictionary, keyName)
                if tpCellProgramIdentifier:
                    program = prog_type_lib.program_type_by_identifier(tpCellProgramIdentifier)
                elif defaultProgramIdentifier:
                    program = prog_type_lib.program_type_by_identifier(defaultProgramIdentifier)
                else:
                    program = prog_type_lib.office_program #Default Office Program as a last resort
                keyName = getKeyName(tpDictionary, 'construction_set')
                tpCellConstructionSetIdentifier = Dictionary.DictionaryValueAtKey(tpDictionary, keyName)
                if tpCellConstructionSetIdentifier:
                    constr_set = constr_set_lib.construction_set_by_identifier(tpCellConstructionSetIdentifier)
                elif defaultConstructionSetIdentifier:
                    constr_set = constr_set_lib.construction_set_by_identifier(defaultConstructionSetIdentifier)
                else:
                    constr_set = constr_set_lib.construction_set_by_identifier("Default Generic Construction Set")
            else:
                tpCellStory = fl[spaceNumber]
                tpCellName = tpCellStory+"_SPACE_"+(str(spaceNumber+1))
                program = prog_type_lib.office_program
                constr_set = constr_set_lib.construction_set_by_identifier("Default Generic Construction Set")
            spaceNames.append(tpCellName)

            tpCellFaces = []
            _ = tpCell.Faces(None, tpCellFaces)
            if tpCellFaces:
                hbRoomFaces = []
                for tpFaceNumber, tpCellFace in enumerate(tpCellFaces):
                    tpCellFaceNormal = topologic.FaceUtility.NormalAtParameters(tpCellFace, 0.5, 0.5)
                    hbRoomFacePoints = []
                    tpFaceVertices = []
                    _ = tpCellFace.ExternalBoundary().Vertices(None, tpFaceVertices)
                    for tpVertex in tpFaceVertices:
                        hbRoomFacePoints.append(Point3D(tpVertex.X(), tpVertex.Y(), tpVertex.Z()))
                    hbRoomFace = Face(tpCellName+'_Face_'+str(tpFaceNumber+1), Face3D(hbRoomFacePoints))
                    tpFaceApertures = []
                    _ = tpCellFace.Apertures(tpFaceApertures)
                    if tpFaceApertures:
                        for tpFaceApertureNumber, tpFaceAperture in enumerate(tpFaceApertures):
                            apertureTopology = topologic.Aperture.Topology(tpFaceAperture)
                            tpFaceApertureDictionary = apertureTopology.GetDictionary()
                            if tpFaceApertureDictionary:
                                tpFaceApertureType = Dictionary.DictionaryValueAtKey(tpFaceApertureDictionary,'type')
                            hbFaceAperturePoints = []
                            tpFaceApertureVertices = []
                            _ = apertureTopology.ExternalBoundary().Vertices(None, tpFaceApertureVertices)
                            for tpFaceApertureVertex in tpFaceApertureVertices:
                                hbFaceAperturePoints.append(Point3D(tpFaceApertureVertex.X(), tpFaceApertureVertex.Y(), tpFaceApertureVertex.Z()))
                            if(tpFaceApertureType):
                                if ("door" in tpFaceApertureType.lower()):
                                    hbFaceAperture = Door(tpCellName+'_Face_'+str(tpFaceNumber+1)+'_Door_'+str(tpFaceApertureNumber), Face3D(hbFaceAperturePoints))
                                else:
                                    hbFaceAperture = Aperture(tpCellName+'_Face_'+str(tpFaceNumber+1)+'_Window_'+str(tpFaceApertureNumber), Face3D(hbFaceAperturePoints))
                            else:
                                hbFaceAperture = Aperture(tpCellName+'_Face_'+str(tpFaceNumber+1)+'_Window_'+str(tpFaceApertureNumber), Face3D(hbFaceAperturePoints))
                            hbRoomFace.add_aperture(hbFaceAperture)
                    else:
                        tpFaceDictionary = tpCellFace.GetDictionary()
                        if (abs(tpCellFaceNormal[2]) < 1e-6) and tpFaceDictionary: #It is a mostly vertical wall and has a dictionary
                            apertureRatio = Dictionary.DictionaryValueAtKey(tpFaceDictionary,'apertureRatio')
                            if apertureRatio:
                                hbRoomFace.apertures_by_ratio(apertureRatio, tolerance=0.01)
                    fType = honeybee.facetype.get_type_from_normal(Vector3D(tpCellFaceNormal[0],tpCellFaceNormal[1],tpCellFaceNormal[2]), roof_angle=30, floor_angle=150)
                    hbRoomFace.type = fType
                    hbRoomFaces.append(hbRoomFace)
                room = Room(tpCellName, hbRoomFaces, 0.01, 1)
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
        Room.solve_adjacency(rooms, 0.01)
        #for room in rooms:
            #room.properties.energy.construction_set = constr_set
        #Room.stories_by_floor_height(rooms, min_difference=2.0)

        if(tpShadingFacesCluster):
            hbShades = []
            tpShadingFaces = []
            _ = tpShadingFacesCluster.Faces(None, tpShadingFaces)
            for faceIndex, tpShadingFace in enumerate(tpShadingFaces):
                faceVertices = []
                _ = tpShadingFace.ExternalBoundary().Vertices(None, faceVertices)
                facePoints = []
                for aVertex in faceVertices:
                    facePoints.append(Point3D(aVertex.X(), aVertex.Y(), aVertex.Z()))
                hbShadingFace = Face3D(facePoints, None, [])
                hbShade = Shade("SHADINGSURFACE_" + str(faceIndex+1), hbShadingFace)
                hbShades.append(hbShade)
        model = Model(buildingName, rooms, orphaned_shades=hbShades)
        return model

    @staticmethod
    def HBModelExportToHBJSON(hbModel, filepath, overwrite):
        """
        Parameters
        ----------
        hbModel : TYPE
            DESCRIPTION.
        filepath : TYPE
            DESCRIPTION.
        overwrite : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.

        """
        # hbModel, filepath = item
        # Make sure the file extension is .hbjson
        ext = filepath[len(filepath)-7:len(filepath)]
        if ext.lower() != ".hbjson":
            filepath = filepath+".hbjson"
        f = None
        try:
            if overwrite == True:
                f = open(filepath, "w")
            else:
                f = open(filepath, "x") # Try to create a new File
        except:
            raise Exception("Error: Could not create a new file at the following location: "+filepath)
        if (f):
            json.dump(hbModel.to_dict(), f, indent=4)
            f.close()    
            return True
        return False
    
    @staticmethod
    def HBModelString(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return item.to_dict()
    
    @staticmethod
    def HBProgramTypeByIdentifier(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return prog_type_lib.program_type_by_identifier(item)
    
    @staticmethod
    def HBProgramTypes():
        """
        Returns
        -------
        list
            DESCRIPTION.

        """
        progTypes = []
        progIdentifiers = list(prog_type_lib.PROGRAM_TYPES)
        for progIdentifier in progIdentifiers: 
            progTypes.append(prog_type_lib.program_type_by_identifier(progIdentifier))
        return [progTypes, progIdentifiers]