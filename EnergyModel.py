import topologicpy
import topologic
from topologicpy.Topology import Topology
from topologicpy.Dictionary import Dictionary
import openstudio
openstudio.Logger.instance().standardOutLogger().setLogLevel(openstudio.Fatal)
import math
from collections import OrderedDict
import os
from os.path import exists
import json
from datetime import datetime
try:
    from tqdm.auto import tqdm
except:
    import sys, subprocess
    call = [sys.executable, '-m', 'pip', 'install', 'tqdm', '-t', sys.path[0]]
    subprocess.run(call)
    from tqdm.auto import tqdm

class EnergyModel:
    @staticmethod
    def ByImportedOSM(path: str):
        """
            Creates an EnergyModel from the input OSM file path.
        
        Parameters
        ----------
        path : string
            The path to the input .OSM file.

        Returns
        -------
        openstudio.openstudiomodelcore.Model
            The OSM model.

        """
        if not path:
            return None
        translator = openstudio.osversion.VersionTranslator()
        osmPath = openstudio.openstudioutilitiescore.toPath(path)
        osModel = translator.loadModel(osmPath)
        if osModel.isNull():
            return None
        else:
            osModel = osModel.get()
        return osModel
    
    @staticmethod
    def ByTopology(building : topologic.CellComplex,
                   shadingSurfaces : topologic.Topology = None,
                   osModelPath : str = None,
                   weatherFilePath : str = None,
                   designDayFilePath  : str = None,
                   floorLevels : list = None,
                   buildingName : str = "TopologicBuilding",
                   buildingType : str = "Commercial",
                   northAxis : float = 0.0,
                   glazingRatio : float = 0.0,
                   coolingTemp : float = 25.0,
                   heatingTemp : float = 20.0,
                   defaultSpaceType : str = "189.1-2009 - Office - WholeBuilding - Lg Office - CZ4-8",
                   spaceNameKey : str = "TOPOLOGIC_name",
                   spaceTypeKey : str = "TOPOLOGIC_type"):
        """
            Creates an EnergyModel from the input topology and parameters.

        Parameters
        ----------
        building : topologic.CellComplex
            The input building topology.
        shadingSurfaces : topologic.Topology , optional
            The input topology for shading surfaces. The default is None.
        osModelPath : str , optional
            The path to the template OSM file. The default is "./assets/EnergyModel/OSMTemplate-OfficeBuilding-3.5.0.osm".
        weatherFilePath : str , optional
            The input energy plus weather (epw) file. The default is "./assets/EnergyModel/GBR_London.Gatwick.037760_IWEC.epw".
        designDayFilePath : str , optional
            The input design day (ddy) file path. The default is "./assets/EnergyModel/GBR_London.Gatwick.037760_IWEC.ddy",
        floorLevels : list , optional
            The list of floor level Z heights including the lowest most and the highest most levels. If set to None, this method will attempt to
            find the floor levels from the horizontal faces of the input topology
        buildingName : str , optional
            The desired name of the building. The default is "TopologicBuilding".
        buildingType : str , optional
            The building type. The default is "Commercial".
        defaultSpaceType : str , optional
            The default space type to apply to spaces that do not have a type assigned in their dictionary. The default is "189.1-2009 - Office - WholeBuilding - Lg Office - CZ4-8".
        northAxis : float , optional
            The counter-clockwise angle in degrees from the positive Y-axis representing the direction of the north axis. The default is 0.0.
        glazingRatio : float , optional
            The glazing ratio (ratio of windows to wall) to use for exterior vertical walls that do not have apertures. If you do not wish to use a glazing ratio, set it to 0. The default is 0.
        coolingTemp : float , optional
            The desired temperature in degrees at which the cooling system should activate. The default is 25.0.
        heatingTemp : float , optional
            The desired temperature in degrees at which the heating system should activate. The default is 25.0..
        spaceNameKey : str , optional
            The dictionary key to use to find the space name value. The default is "Name".
        spaceTypeKey : str , optional
            The dictionary key to use to find the space type value. The default is "Type".

        Returns
        -------
        openstudio.openstudiomodelcore.Model
            The created OSM model.

        """
        
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
        
        def getFloorLevels(building):
            from topologicpy.Vertex import Vertex
            from topologicpy.CellComplex import CellComplex
            from topologicpy.Dictionary import Dictionary

            d = CellComplex.Decompose(building)
            bhf = d['bottomHorizontalFaces']
            ihf = d['internalHorizontalFaces']
            thf = d ['topHorizontalFaces']
            hf = bhf+ihf+thf
            floorLevels = [Vertex.Z(Topology.Centroid(f)) for f in hf]
            floorLevels = list(set(floorLevels))
            floorLevels.sort()
            return floorLevels
        
        if not osModelPath:
            import os
            osModelPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "EnergyModel", "OSMTemplate-OfficeBuilding-3.5.0.osm")
        if not weatherFilePath or not designDayFilePath:
            import os
            weatherFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "EnergyModel", "GBR_London.Gatwick.037760_IWEC.epw")
            designDayFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "EnergyModel", "GBR_London.Gatwick.037760_IWEC.ddy")
        translator = openstudio.osversion.VersionTranslator()
        osmFile = openstudio.openstudioutilitiescore.toPath(osModelPath)
        osModel = translator.loadModel(osmFile)
        if osModel.isNull():
            return None
        else:
            osModel = osModel.get()
        osEPWFile = openstudio.openstudioutilitiesfiletypes.EpwFile.load(openstudio.toPath(weatherFilePath))
        if osEPWFile.is_initialized():
            osEPWFile = osEPWFile.get()
            openstudio.model.WeatherFile.setWeatherFile(osModel, osEPWFile)
        ddyModel = openstudio.openstudioenergyplus.loadAndTranslateIdf(openstudio.toPath(designDayFilePath))
        if ddyModel.is_initialized():
            ddyModel = ddyModel.get()
            for ddy in ddyModel.getObjectsByType(openstudio.IddObjectType("OS:SizingPeriod:DesignDay")):
                osModel.addObject(ddy.clone())
        else:
            print("Could not load the DesignDay file")
            return None

        osBuilding = osModel.getBuilding()
        if not floorLevels:
            floorLevels = getFloorLevels(building)
        osBuilding.setStandardsNumberOfStories(len(floorLevels) - 1)
        osBuilding.setNominalFloortoFloorHeight(max(floorLevels) / osBuilding.standardsNumberOfStories().get())
        osBuilding.setDefaultConstructionSet(osModel.getDefaultConstructionSets()[0])
        osBuilding.setDefaultScheduleSet(osModel.getDefaultScheduleSets()[0])
        osBuilding.setName(buildingName)
        osBuilding.setStandardsBuildingType(buildingType)
        osBuilding.setSpaceType(osModel.getSpaceTypeByName(defaultSpaceType).get())
        for storyNumber in range(osBuilding.standardsNumberOfStories().get()):
            osBuildingStory = openstudio.model.BuildingStory(osModel)
            osBuildingStory.setName("STORY_" + str(storyNumber))
            osBuildingStory.setNominalZCoordinate(floorLevels[storyNumber])
            osBuildingStory.setNominalFloortoFloorHeight(osBuilding.nominalFloortoFloorHeight().get())
        osBuilding.setNorthAxis(northAxis)

        heatingScheduleConstant = openstudio.model.ScheduleConstant(osModel)
        heatingScheduleConstant.setValue(heatingTemp)
        coolingScheduleConstant = openstudio.model.ScheduleConstant(osModel)
        coolingScheduleConstant.setValue(coolingTemp)
        osThermostat = openstudio.model.ThermostatSetpointDualSetpoint(osModel)
        osThermostat.setHeatingSetpointTemperatureSchedule(heatingScheduleConstant)
        osThermostat.setCoolingSetpointTemperatureSchedule(coolingScheduleConstant)

        osBuildingStorys = list(osModel.getBuildingStorys())
        osBuildingStorys.sort(key=lambda x: x.nominalZCoordinate().get())
        osSpaces = []
        spaceNames = []
        for spaceNumber, buildingCell in enumerate(Topology.SubTopologies(building, "Cell")):
            osSpace = openstudio.model.Space(osModel)
            osSpaceZ = buildingCell.CenterOfMass().Z()
            osBuildingStory = osBuildingStorys[0]
            for x in osBuildingStorys:
                osBuildingStoryZ = x.nominalZCoordinate().get()
                if osBuildingStoryZ + x.nominalFloortoFloorHeight().get() < osSpaceZ:
                    continue
                if osBuildingStoryZ < osSpaceZ:
                    osBuildingStory = x
                break
            osSpace.setBuildingStory(osBuildingStory)
            cellDictionary = buildingCell.GetDictionary()
            if cellDictionary:
                if spaceTypeKey:
                    keyType = getKeyName(cellDictionary, spaceTypeKey)
                else:
                    keyType = getKeyName(cellDictionary, 'type')
                if keyType:
                    osSpaceTypeName = Dictionary.ValueAtKey(cellDictionary,keyType)
                else:
                    osSpaceTypeName = defaultSpaceType
                if osSpaceTypeName:
                    sp_ = osModel.getSpaceTypeByName(osSpaceTypeName)
                    if sp_.is_initialized():
                        osSpace.setSpaceType(sp_.get())
                if spaceNameKey:
                    keyName = getKeyName(cellDictionary, spaceNameKey)

                else:
                    keyName = getKeyName(cellDictionary, 'name')
                osSpaceName = None
                if keyName:
                    osSpaceName = createUniqueName(Dictionary.ValueAtKey(cellDictionary,keyName),spaceNames, 1)
                if osSpaceName:
                    osSpace.setName(osSpaceName)
            else:
                osSpaceName = osBuildingStory.name().get() + "_SPACE_" + str(spaceNumber)
                osSpace.setName(osSpaceName)
                sp_ = osModel.getSpaceTypeByName(defaultSpaceType)
                if sp_.is_initialized():
                    osSpace.setSpaceType(sp_.get())
            spaceNames.append(osSpaceName)
            cellFaces = Topology.SubTopologies(buildingCell, "Face")
            if cellFaces:
                for faceNumber, buildingFace in enumerate(cellFaces):
                    osFacePoints = []
                    for vertex in Topology.SubTopologies(buildingFace.ExternalBoundary(), "Vertex"):
                        osFacePoints.append(openstudio.Point3d(vertex.X(), vertex.Y(), vertex.Z()))
                    osSurface = openstudio.model.Surface(osFacePoints, osModel)
                    faceNormal = topologic.FaceUtility.NormalAtParameters(buildingFace, 0.5, 0.5)
                    osFaceNormal = openstudio.Vector3d(faceNormal[0], faceNormal[1], faceNormal[2])
                    osFaceNormal.normalize()
                    if osFaceNormal.dot(osSurface.outwardNormal()) < 1e-6:
                        osSurface.setVertices(list(reversed(osFacePoints)))
                    osSurface.setSpace(osSpace)
                    faceCells = []
                    _ = topologic.FaceUtility.AdjacentCells(buildingFace, building, faceCells)
                    if len(faceCells) == 1: #Exterior Surfaces
                        osSurface.setOutsideBoundaryCondition("Outdoors")
                        if (math.degrees(math.acos(osSurface.outwardNormal().dot(openstudio.Vector3d(0, 0, 1)))) > 135) or (math.degrees(math.acos(osSurface.outwardNormal().dot(openstudio.Vector3d(0, 0, 1)))) < 45):
                            osSurface.setSurfaceType("RoofCeiling")
                            osSurface.setOutsideBoundaryCondition("Outdoors")
                            osSurface.setName(osSpace.name().get() + "_TopHorizontalSlab_" + str(faceNumber))
                            if max(list(map(lambda vertex: vertex.Z(), Topology.SubTopologies(buildingFace, "Vertex")))) < 1e-6:
                                osSurface.setSurfaceType("Floor")
                                osSurface.setOutsideBoundaryCondition("Ground")
                                osSurface.setName(osSpace.name().get() + "_BottomHorizontalSlab_" + str(faceNumber))
                        else:
                            osSurface.setSurfaceType("Wall")
                            osSurface.setOutsideBoundaryCondition("Outdoors")
                            osSurface.setName(osSpace.name().get() + "_ExternalVerticalFace_" + str(faceNumber))
                            # Check for exterior apertures
                            faceDictionary = buildingFace.GetDictionary()
                            apertures = []
                            _ = buildingFace.Apertures(apertures)
                            if len(apertures) > 0:
                                for aperture in apertures:
                                    osSubSurfacePoints = []
                                    #apertureFace = TopologySubTopologies.processItem([aperture, topologic.Face])[0]
                                    apertureFace = topologic.Aperture.Topology(aperture)
                                    for vertex in Topology.SubTopologies(apertureFace.ExternalBoundary(), "Vertex"):
                                        osSubSurfacePoints.append(openstudio.Point3d(vertex.X(), vertex.Y(), vertex.Z()))
                                    osSubSurface = openstudio.model.SubSurface(osSubSurfacePoints, osModel)
                                    apertureFaceNormal = topologic.FaceUtility.NormalAtParameters(apertureFace, 0.5, 0.5)
                                    osSubSurfaceNormal = openstudio.Vector3d(apertureFaceNormal[0], apertureFaceNormal[1], apertureFaceNormal[2])
                                    osSubSurfaceNormal.normalize()
                                    if osSubSurfaceNormal.dot(osSubSurface.outwardNormal()) < 1e-6:
                                        osSubSurface.setVertices(list(reversed(osSubSurfacePoints)))
                                    osSubSurface.setSubSurfaceType("FixedWindow")
                                    osSubSurface.setSurface(osSurface)
                            else:
                                    # Get the dictionary keys
                                    keys = faceDictionary.Keys()
                                    if ('TOPOLOGIC_glazing_ratio' in keys):
                                        faceGlazingRatio = Dictionary.ValueAtKey(faceDictionary,'TOPOLOGIC_glazing_ratio')
                                        if faceGlazingRatio and faceGlazingRatio >= 0.01:
                                            osSurface.setWindowToWallRatio(faceGlazingRatio)
                                    else:
                                        if glazingRatio > 0.01: #Glazing ratio must be more than 1% to make any sense.
                                            osSurface.setWindowToWallRatio(glazingRatio)
                    else: #Interior Surfaces
                        if (math.degrees(math.acos(osSurface.outwardNormal().dot(openstudio.Vector3d(0, 0, 1)))) > 135):
                            osSurface.setSurfaceType("Floor")
                            osSurface.setName(osSpace.name().get() + "_InternalHorizontalFace_" + str(faceNumber))
                        elif (math.degrees(math.acos(osSurface.outwardNormal().dot(openstudio.Vector3d(0, 0, 1)))) < 40):
                            osSurface.setSurfaceType("RoofCeiling")
                            osSurface.setName(osSpace.name().get() + "_InternalHorizontalFace_" + str(faceNumber))
                        else:
                            osSurface.setSurfaceType("Wall")
                            osSurface.setName(osSpace.name().get() + "_InternalVerticalFace_" + str(faceNumber))
                        # Check for interior apertures
                        faceDictionary = buildingFace.GetDictionary()
                        apertures = []
                        _ = buildingFace.Apertures(apertures)
                        if len(apertures) > 0:
                            for aperture in apertures:
                                osSubSurfacePoints = []
                                #apertureFace = TopologySubTopologies.processItem([aperture, "Face"])[0]
                                apertureFace = topologic.Aperture.Topology(aperture)
                                for vertex in Topology.SubTopologies(apertureFace.ExternalBoundary(), "Vertex"):
                                    osSubSurfacePoints.append(openstudio.Point3d(vertex.X(), vertex.Y(), vertex.Z()))
                                osSubSurface = openstudio.model.SubSurface(osSubSurfacePoints, osModel)
                                apertureFaceNormal = topologic.FaceUtility.NormalAtParameters(apertureFace, 0.5, 0.5)
                                osSubSurfaceNormal = openstudio.Vector3d(apertureFaceNormal[0], apertureFaceNormal[1], apertureFaceNormal[2])
                                osSubSurfaceNormal.normalize()
                                if osSubSurfaceNormal.dot(osSubSurface.outwardNormal()) < 1e-6:
                                    osSubSurface.setVertices(list(reversed(osSubSurfacePoints)))
                                osSubSurface.setSubSurfaceType("Door") #We are assuming all interior apertures to be doors
                                osSubSurface.setSurface(osSurface)

            osThermalZone = openstudio.model.ThermalZone(osModel)
            osThermalZone.setVolume(topologic.CellUtility.Volume(buildingCell))
            osThermalZone.setName(osSpace.name().get() + "_THERMAL_ZONE")
            osThermalZone.setUseIdealAirLoads(True)
            osThermalZone.setVolume(topologic.CellUtility.Volume(buildingCell))
            osThermalZone.setThermostatSetpointDualSetpoint(osThermostat)
            osSpace.setThermalZone(osThermalZone)

            for x in osSpaces:
                if osSpace.boundingBox().intersects(x.boundingBox()):
                    osSpace.matchSurfaces(x)
            osSpaces.append(osSpace)

        
        if shadingSurfaces:
            osShadingGroup = openstudio.model.ShadingSurfaceGroup(osModel)
            for faceIndex, shadingFace in enumerate(Topology.SubTopologies(shadingSurfaces, "Face")):
                facePoints = []
                for aVertex in Topology.SubTopologies(shadingFace.ExternalBoundary(), "Vertex"):
                    facePoints.append(openstudio.Point3d(aVertex.X(), aVertex.Y(), aVertex.Z()))
                aShadingSurface = openstudio.model.ShadingSurface(facePoints, osModel)
                faceNormal = topologic.FaceUtility.NormalAtParameters(shadingFace, 0.5, 0.5)
                osFaceNormal = openstudio.Vector3d(faceNormal[0], faceNormal[1], faceNormal[2])
                osFaceNormal.normalize()
                if osFaceNormal.dot(aShadingSurface.outwardNormal()) < 0:
                    aShadingSurface.setVertices(list(reversed(facePoints)))
                aShadingSurface.setName("SHADINGSURFACE_" + str(faceIndex))
                aShadingSurface.setShadingSurfaceGroup(osShadingGroup)

        osModel.purgeUnusedResourceObjects()
        return osModel
    
    @staticmethod
    def ColumnNames(model, reportName, tableName):
        """
            Returns the list of column names given an OSM model, report name, and table name.

        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.
        reportName : str
            The input report name.
        tableName : str
            The input table name.

        Returns
        -------
        list
            the list of column names.

        """
        sql = model.sqlFile().get()
        query = "SELECT ColumnName FROM tabulardatawithstrings WHERE ReportName = '"+reportName+"' AND TableName = '"+tableName+"'"
        columnNames = sql.execAndReturnVectorOfString(query).get()
        return list(OrderedDict( (x,1) for x in columnNames ).keys()) #Making a unique list and keeping its order

    @staticmethod
    def DefaultConstructionSets(model):
        """
            Returns the default construction sets in the input OSM model.

        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.

        Returns
        -------
        list
            The default construction sets.

        """
        sets = model.getDefaultConstructionSets()
        names = []
        for aSet in sets:
            names.append(aSet.name().get())
        return [sets, names]
    
    @staticmethod
    def DefaultScheduleSets(model):
        """
            Returns the default schedule sets found in the input OSM model.

        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.

        Returns
        -------
        list
            The list of default schedule sets.

        """
        sets = model.getDefaultScheduleSets()
        names = []
        for aSet in sets:
            names.append(aSet.name().get())
        return [sets, names]
    
    @staticmethod
    def ExportToGbXML(model, path, overwrite=False):
        """
            Exports the input OSM model to a gbxml file.

        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.
        path : str
            The path for saving the file.
        overwrite : bool, optional
            If set to True any file with the same name is over-written. The default is False.

        Returns
        -------
        bool
            True if the file is written successfully. False otherwise.

        """
        ext = path[len(path)-4:len(path)]
        if ext.lower() != ".xml":
            path = path+".xml"
        if(exists(path) and (overwrite == False)):
            return None
        return openstudio.gbxml.GbXMLForwardTranslator().modelToGbXML(model, openstudio.openstudioutilitiescore.toPath(path))

    
    @staticmethod
    def ExportToOSM(model, path, overwrite=False):
        """
            Exports the input OSM model to an OSM file.
        
        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.
        path : str
            The path for saving the file.
        overwrite : bool, optional
            If set to True any file with the same name is over-written. The default is False.

        Returns
        -------
        bool
            True if the file is written successfully. False otherwise.

        """
        ext = path[len(path)-4:len(path)]
        if ext.lower() != ".osm":
            path = path+".osm"
        osCondition = False
        osPath = openstudio.openstudioutilitiescore.toPath(path)
        osCondition = model.save(osPath, overwrite)
        return osCondition

    @staticmethod
    def GbXMLString(model):
        """
            Returns the gbxml string of the input OSM model.

        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.

        Returns
        -------
        str
            The gbxml string.

        """
        return openstudio.gbxml.GbXMLForwardTranslator().modelToGbXMLString(model)
    
    @staticmethod
    def Query(model,
              reportName : str = "HVACSizingSummary",
              reportForString : str = "Entire Facility",
              tableName : str = "Zone Sensible Cooling",
              columnName : str = "Calculated Design Load",
              rowNames : list = [],
              units : str = "W"):
        """
            Queries the model for values.

        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.
        reportName : str , optional
            The input report name. The default is "HVACSizingSummary".
        reportForString : str, optional
            The input report for string. The default is "Entire Facility".
        tableName : str , optional
            The input table name. The default is "Zone Sensible Cooling".
        columnName : str , optional
            The input column name. The default is "Calculated Design Load".
        rowNames : list , optional
            The input list of row names. The default is [].
        units : str , optional
            The input units. The default is "W".

        Returns
        -------
        list
            The list of values.

        """
        
        def doubleValueFromQuery(sqlFile, reportName, reportForString,
                                 tableName, columnName, rowName,
                                 units):
            query = "SELECT Value FROM tabulardatawithstrings WHERE ReportName='" + reportName + "' AND ReportForString='" + reportForString + "' AND TableName = '" + tableName + "' AND RowName = '" + rowName + "' AND ColumnName= '" + columnName + "' AND Units='" + units + "'";
            osOptionalDoubleValue = sqlFile.execAndReturnFirstDouble(query)
            if (osOptionalDoubleValue.is_initialized()):
                return osOptionalDoubleValue.get()
            else:
                return None
        
        sqlFile = model.sqlFile().get()
        returnValues = []
        for rowName in rowNames:
            returnValues.append(doubleValueFromQuery(sqlFile, reportName, reportForString, tableName, columnName, rowName, units))
        return returnValues
    
    @staticmethod
    def ReportNames(model):
        """
            Returns the report names found in the input OSM model.

        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.

        Returns
        -------
        list
            The list of report names found in the input OSM model.

        """
        sql = model.sqlFile().get()
        reportNames = sql.execAndReturnVectorOfString("SELECT ReportName FROM tabulardatawithstrings").get()
        return list(OrderedDict( (x,1) for x in reportNames ).keys()) #Making a unique list and keeping its order

    @staticmethod
    def RowNames(model, reportName, tableName):
        """
            Returns the list of row names given an OSM model, report name, and table name.

        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.
        reportName : str
            The input name of the report.
        tableName : str
            The input name of the table.

        Returns
        -------
        list
            The list of row names.

        """
        sql = model.sqlFile().get()
        query = "SELECT RowName FROM tabulardatawithstrings WHERE ReportName = '"+reportName+"' AND TableName = '"+tableName+"'"
        columnNames = sql.execAndReturnVectorOfString(query).get()
        return list(OrderedDict( (x,1) for x in columnNames ).keys()) #Making a unique list and keeping its order

    @staticmethod
    def Run(model, weatherFilePath: str = None, osBinaryPath : str = None, outputFolder : str = None):
        """
            Runs an energy simulation.
        
        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.
        weatherFilePath : str
            The path to the epw weather file.
        osBinaryPath : str
            The path to the openstudio binary.
        outputFolder : str
            The path to the output folder.

        Returns
        -------
        model : openstudio.openstudiomodelcore.Model
            The simulated OSM model.

        """
        import os
        if not weatherFilePath:
            weatherFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "EnergyModel", "GBR_London.Gatwick.037760_IWEC.epw")
        pbar = tqdm(desc='Running Simulation', total=100, leave=False)
        utcnow = datetime.utcnow()
        timestamp = utcnow.strftime("UTC-%Y-%m-%d-%H-%M-%S")
        if not outputFolder:
            home = os.path.expanduser('~')
            outputFolder = os.path.join(home, "EnergyModels", timestamp)
        else:
            outputFolder = os.path.join(outputFolder, timestamp)
        os.mkdir(outputFolder)
        pbar.update(10)
        osmPath = os.path.join(outputFolder, model.getBuilding().name().get() + ".osm")
        model.save(openstudio.openstudioutilitiescore.toPath(osmPath), True)
        oswPath = os.path.join(outputFolder, model.getBuilding().name().get() + ".osw")
        pbar.update(20)
        #print("oswPath = "+oswPath)
        workflow = model.workflowJSON()
        workflow.setSeedFile(openstudio.openstudioutilitiescore.toPath(osmPath))
        pbar.update(30)
        #print("Seed File Set")
        workflow.setWeatherFile(openstudio.openstudioutilitiescore.toPath(weatherFilePath))
        pbar.update(40)
        #print("Weather File Set")
        workflow.saveAs(openstudio.openstudioutilitiescore.toPath(oswPath))
        pbar.update(50)
        #print("OSW File Saved")
        cmd = osBinaryPath+" run -w " + "\"" + oswPath + "\""
        pbar.update(60)
        os.system(cmd)
        #print("Simulation DONE")
        sqlPath = os.path.join(os.path.join(outputFolder,"run"), "eplusout.sql")
        pbar.update(100)
        #print("sqlPath = "+sqlPath)
        osSqlFile = openstudio.SqlFile(openstudio.openstudioutilitiescore.toPath(sqlPath))
        model.setSqlFile(osSqlFile)
        pbar.close()
        return model
    
    @staticmethod
    def SpaceDictionaries(model):
        """
            Return the space dictionaries found in the input OSM model.
        
        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.

        Returns
        -------
        dict
            The dictionary of space types, names, and colors found in the input OSM model. The dictionary has the following keys:
            - "types"
            - "names"
            - "colors"

        """
        types = model.getSpaceTypes()
        names = []
        colors = []
        for aType in types:
            names.append(aType.name().get())
            red = aType.renderingColor().get().renderingRedValue()
            green = aType.renderingColor().get().renderingGreenValue()
            blue = aType.renderingColor().get().renderingBlueValue()
            colors.append([red,green,blue])
        return {'types': types, 'names': names, 'colors': colors}
    
    @staticmethod
    def SpaceTypes(model):
        """
            Return the space types found in the input OSM model.
        
        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.

        Returns
        -------
        list
            The list of space types

        """
        return model.getSpaceTypes()
    
    @staticmethod
    def SpaceTypeNames(model):
        """
            Return the space type names found in the input OSM model.
        
        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.

        Returns
        -------
        list
            The list of space type names

        """
        types = model.getSpaceTypes()
        names = []
        colors = []
        for aType in types:
            names.append(aType.name().get())
        return names
    
    @staticmethod
    def SpaceColors(model):
        """
            Return the space colors found in the input OSM model.
        
        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.

        Returns
        -------
        list
            The list of space colors. Each item is a three-item list representing the red, green, and blue values of the color.

        """
        types = model.getSpaceTypes()
        colors = []
        for aType in types:
            red = aType.renderingColor().get().renderingRedValue()
            green = aType.renderingColor().get().renderingGreenValue()
            blue = aType.renderingColor().get().renderingBlueValue()
            colors.append([red,green,blue])
        return colors
    
    @staticmethod
    def SqlFile(model):
        """
            Returns the SQL file found in the input OSM model.
        
        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.

        Returns
        -------
        SQL file
            The SQL file found in the input OSM model.

        """
        return model.sqlFile().get()
    
    @staticmethod
    def TableNames(model, reportName):
        """
            Returns the table names found in the input OSM model and report name.
        
        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.
        reportName : str
            The input report name.

        Returns
        -------
        list
            The list of table names found in the input OSM model and report name.

        """
        sql = model.sqlFile().get()
        tableNames = sql.execAndReturnVectorOfString("SELECT TableName FROM tabulardatawithstrings WHERE ReportName='"+reportName+"'").get()
        return list(OrderedDict( (x,1) for x in tableNames ).keys()) #Making a unique list and keeping its order

    @staticmethod
    def Topologies(model):
        """
        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.

        Returns
        -------
        dict
            The dictionary of topologies found in the input OSM model. The keys of the dictionary are:
            - "cells"
            - "apertures"
            - "shadingFaces"

        """
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        def surfaceToFace(surface):
            surfaceEdges = []
            surfaceVertices = surface.vertices()
            for i in range(len(surfaceVertices)-1):
                sv = topologic.Vertex.ByCoordinates(surfaceVertices[i].x(), surfaceVertices[i].y(), surfaceVertices[i].z())
                ev = topologic.Vertex.ByCoordinates(surfaceVertices[i+1].x(), surfaceVertices[i+1].y(), surfaceVertices[i+1].z())
                edge = Edge.ByStartVertexEndVertex(sv, ev)
                if not edge:
                    continue
                surfaceEdges.append(edge)
            sv = topologic.Vertex.ByCoordinates(surfaceVertices[len(surfaceVertices)-1].x(), surfaceVertices[len(surfaceVertices)-1].y(), surfaceVertices[len(surfaceVertices)-1].z())
            ev = topologic.Vertex.ByCoordinates(surfaceVertices[0].x(), surfaceVertices[0].y(), surfaceVertices[0].z())
            edge = Edge.ByStartVertexEndVertex(sv, ev)
            surfaceEdges.append(edge)
            surfaceWire = Wire.ByEdges(surfaceEdges)
            internalBoundaries = []
            surfaceFace = Face.ByWires(surfaceWire, internalBoundaries)
            return surfaceFace
        
        def addApertures(face, apertures):
            usedFaces = []
            for aperture in apertures:
                cen = aperture.CenterOfMass()
                try:
                    params = face.ParametersAtVertex(cen)
                    u = params[0]
                    v = params[1]
                    w = 0.5
                except:
                    u = 0.5
                    v = 0.5
                    w = 0.5
                context = topologic.Context.ByTopologyParameters(face, u, v, w)
                _ = topologic.Aperture.ByTopologyContext(aperture, context)
            return face
        spaces = list(model.getSpaces())
        
        vertexIndex = 0
        cells = []
        apertures = []
        shadingFaces = []
        shadingSurfaces = list(model.getShadingSurfaces())
        
        for aShadingSurface in shadingSurfaces:
            shadingFace = surfaceToFace(aShadingSurface)
            if aShadingSurface.shadingSurfaceGroup().is_initialized():
                shadingGroup = aShadingSurface.shadingSurfaceGroup().get()
                if shadingGroup.space().is_initialized():
                    space = shadingGroup.space().get()
                    osTransformation = space.transformation()
                    osTranslation = osTransformation.translation()
                    osMatrix = osTransformation.rotationMatrix()
                    rotation11 = osMatrix[0, 0]
                    rotation12 = osMatrix[0, 1]
                    rotation13 = osMatrix[0, 2]
                    rotation21 = osMatrix[1, 0]
                    rotation22 = osMatrix[1, 1]
                    rotation23 = osMatrix[1, 2]
                    rotation31 = osMatrix[2, 0]
                    rotation32 = osMatrix[2, 1]
                    rotation33 = osMatrix[2, 2]
                    shadingFace = topologic.TopologyUtility.Transform(shadingFace, osTranslation.x(), osTranslation.y(), osTranslation.z(), rotation11, rotation12, rotation13, rotation21, rotation22, rotation23, rotation31, rotation32, rotation33)
            shadingFaces.append(shadingFace)
        
        for count, aSpace in enumerate(spaces):
            osTransformation = aSpace.transformation()
            osTranslation = osTransformation.translation()
            osMatrix = osTransformation.rotationMatrix()
            rotation11 = osMatrix[0, 0]
            rotation12 = osMatrix[0, 1]
            rotation13 = osMatrix[0, 2]
            rotation21 = osMatrix[1, 0]
            rotation22 = osMatrix[1, 1]
            rotation23 = osMatrix[1, 2]
            rotation31 = osMatrix[2, 0]
            rotation32 = osMatrix[2, 1]
            rotation33 = osMatrix[2, 2]
            spaceFaces = []
            surfaces = aSpace.surfaces()

            for aSurface in surfaces:
                aFace = surfaceToFace(aSurface)
                aFace = topologic.TopologyUtility.Transform(aFace, osTranslation.x(), osTranslation.y(), osTranslation.z(), rotation11, rotation12, rotation13, rotation21, rotation22, rotation23, rotation31, rotation32, rotation33)
                #aFace.__class__ = topologic.Face
                subSurfaces = aSurface.subSurfaces()
                for aSubSurface in subSurfaces:
                    aperture = surfaceToFace(aSubSurface)
                    aperture = topologic.TopologyUtility.Transform(aperture, osTranslation.x(), osTranslation.y(), osTranslation.z(), rotation11, rotation12, rotation13, rotation21, rotation22, rotation23, rotation31, rotation32, rotation33)
                    # aperture.__class__ = topologic.Face
                    apertures.append(aperture)
                addApertures(aFace, apertures)
                spaceFaces.append(aFace)
            spaceFaces = [x for x in spaceFaces if isinstance(x, topologic.Face)]
            spaceCell = Cell.ByFaces(spaceFaces)
            if not spaceCell:
                spaceCell = Shell.ByFaces(spaceFaces)
            if not spaceCell:
                spaceCell = Cluster.ByTopologies(spaceFaces)
            if spaceCell: #debugging
                # Set Dictionary for Cell
                keys = []
                values = []

                keys.append("TOPOLOGIC_id")
                keys.append("TOPOLOGIC_name")
                keys.append("TOPOLOGIC_type")
                keys.append("TOPOLOGIC_color")
                spaceID = str(aSpace.handle()).replace('{','').replace('}','')
                values.append(spaceID)
                values.append(aSpace.name().get())
                spaceTypeName = "Unknown"
                red = 255
                green = 255
                blue = 255
                
                if (aSpace.spaceType().is_initialized()):
                    if(aSpace.spaceType().get().name().is_initialized()):
                        spaceTypeName = aSpace.spaceType().get().name().get()
                    if(aSpace.spaceType().get().renderingColor().is_initialized()):
                        red = aSpace.spaceType().get().renderingColor().get().renderingRedValue()
                        green = aSpace.spaceType().get().renderingColor().get().renderingGreenValue()
                        blue = aSpace.spaceType().get().renderingColor().get().renderingBlueValue()
                values.append(spaceTypeName)
                values.append([red, green, blue])
                d = Dictionary.ByKeysValues(keys, values)
                spaceCell = Topology.SetDictionary(spaceCell, d)
                cells.append(spaceCell)
        return {'cells':cells, 'apertures':apertures, 'shadingFaces': shadingFaces}

    @staticmethod
    def Units(model, reportName, tableName, columnName):
        """
        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.
        reportName : str
            The input report name.
        tableName : str
            The input table name.
        columnName : str
            The input column name.

        Returns
        -------
        str
            The units string found in the input OSM model, report name, table name, and column name.

        """
        # model = item[0]
        # reportName = item[1]
        # tableName = item[2]
        # columnName = item[3]
        sql = model.sqlFile().get()
        query = "SELECT Units FROM tabulardatawithstrings WHERE ReportName = '"+reportName+"' AND TableName = '"+tableName+"' AND ColumnName = '"+columnName+"'"
        units = sql.execAndReturnFirstString(query)
        if (units.is_initialized()):
            units = units.get()
        else:
            return None
        return units
    
