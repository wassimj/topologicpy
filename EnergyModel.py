import topologicpy
import topologic
from topologicpy.Topology import Topology
from topologicpy.Dictionary import Dictionary
import math
from collections import OrderedDict
import os
from os.path import exists
import json
from datetime import datetime

class EnergyModel:
    @staticmethod
    def EnergyModelByImportedOSM(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        model : TYPE
            DESCRIPTION.

        """
        translator = openstudio.osversion.VersionTranslator()
        osmFile = openstudio.openstudioutilitiescore.toPath(item)
        model = translator.loadModel(osmFile)
        if model.isNull():
            raise Exception("File Path is not a valid path to an OpenStudio Model")
            return None
        else:
            model = model.get()
        return model
    
    @staticmethod
    def EnergyModelByTopology(osModelPath, weatherFilePath, designDayFilePath,
                              buildingTopology, shadingSurfaces, floorLevels,
                              buildingName, buildingType, defaultSpaceType,
                              northAxis, glazingRatio, coolingTemp,
                              heatingTemp, roomNameKey, roomTypeKey):
        """
        Parameters
        ----------
        osModelPath : TYPE
            DESCRIPTION.
        weatherFilePath : TYPE
            DESCRIPTION.
        designDayFilePath : TYPE
            DESCRIPTION.
        buildingTopology : TYPE
            DESCRIPTION.
        shadingSurfaces : TYPE
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
        roomNameKey : TYPE
            DESCRIPTION.
        roomTypeKey : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        osModel : TYPE
            DESCRIPTION.

        """
        
        # osModelPath = item[0]
        # weatherFilePath = item[1]
        # designDayFilePath = item[2]
        # buildingTopology = item[3]
        # shadingSurfaces = item[4]
        # floorLevels = item[5]
        # buildingName = item[6]
        # buildingType = item[7]
        # defaultSpaceType = item[8]
        # northAxis = item[9]
        # glazingRatio = item[10]
        # coolingTemp = item[11]
        # heatingTemp = item[12]
        # roomNameKey = item[13]
        # roomTypeKey = item[14]
        
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
        
        translator = openstudio.osversion.VersionTranslator()
        osmFile = openstudio.openstudioutilitiescore.toPath(osModelPath)
        osModel = translator.loadModel(osmFile)
        if osModel.isNull():
            raise Exception("File Path is not a valid path to an OpenStudio Model")
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

        osBuilding = osModel.getBuilding()
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
        for spaceNumber, buildingCell in enumerate(Topology.SubTopologies(buildingTopology, "Cell")):
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
                if roomTypeKey:
                    keyType = getKeyName(cellDictionary, roomTypeKey)
                else:
                    keyType = getKeyName(cellDictionary, 'type')
                osSpaceTypeName = Dictionary.ValueAtKey(cellDictionary,keyType)
                if osSpaceTypeName:
                    sp_ = osModel.getSpaceTypeByName(osSpaceTypeName)
                    if sp_.is_initialized():
                        osSpace.setSpaceType(sp_.get())
                if roomNameKey:
                    keyName = getKeyName(cellDictionary, roomNameKey)

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
                    _ = topologic.FaceUtility.AdjacentCells(buildingFace, buildingTopology, faceCells)
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

        osShadingGroup = openstudio.model.ShadingSurfaceGroup(osModel)
        if not isinstance(shadingSurfaces,int):
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
    def EnergyModelColumnNames(model, reportName, tableName):
        """
        Parameters
        ----------
        model : TYPE
            DESCRIPTION.
        reportName : TYPE
            DESCRIPTION.
        tableName : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # model = item[0]
        # reportName = item[1]
        # tableName = item[2]
        sql = model.sqlFile().get()
        query = "SELECT ColumnName FROM tabulardatawithstrings WHERE ReportName = '"+reportName+"' AND TableName = '"+tableName+"'"
        columnNames = sql.execAndReturnVectorOfString(query).get()
        return list(OrderedDict( (x,1) for x in columnNames ).keys()) #Making a unique list and keeping its order

    @staticmethod
    def EnergyModelDefaultConstructionSets(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        sets = item.getDefaultConstructionSets()
        names = []
        for aSet in sets:
            names.append(aSet.name().get())
        return [sets, names]
    
    @staticmethod
    def EnergyModelDefaultScheduleSets(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        sets = item.getDefaultScheduleSets()
        names = []
        for aSet in sets:
            names.append(aSet.name().get())
        return [sets, names]
    
    @staticmethod
    def EnergyModelExportToGbXML(osModel, filepath, overwrite):
        """
        Parameters
        ----------
        osModel : TYPE
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
        TYPE
            DESCRIPTION.

        """
        # osModel = item[0]
        # filepath = item[1]
        # Make sure the file extension is .xml
        ext = filepath[len(filepath)-4:len(filepath)]
        if ext.lower() != ".xml":
            filepath = filepath+".xml"
        if(exists(filepath) and (overwrite == False)):
            raise Exception("Error: Could not create a new file at the following location: "+filepath)
        return openstudio.gbxml.GbXMLForwardTranslator().modelToGbXML(osModel, openstudio.openstudioutilitiescore.toPath(filepath))

    @staticmethod
    def EnergyModelExportToHBJSON(hbjson, filepath, overwrite):
        """
        Parameters
        ----------
        hbjson : TYPE
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
        # hbjson = item[0]
        # filepath = item[1]
        # Make sure the file extension is .BREP
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
            json.dump(hbjson[0], f, indent=4)
            f.close()    
            return True
        return False
    
    @staticmethod
    def EnergyModelExportToOSM(model, filePath, overwrite):
        """
        Parameters
        ----------
        model : TYPE
            DESCRIPTION.
        filePath : TYPE
            DESCRIPTION.
        overwrite : TYPE
            DESCRIPTION.

        Returns
        -------
        osCondition : TYPE
            DESCRIPTION.

        """
        # model = item[0]
        # filePath = item[1]
        # Make sure the file extension is .OSM
        ext = filePath[len(filePath)-4:len(filePath)]
        if ext.lower() != ".osm":
            filePath = filePath+".osm"
        osCondition = False
        osPath = openstudio.openstudioutilitiescore.toPath(filePath)
        osCondition = model.save(osPath, overwrite)
        return osCondition

    @staticmethod
    def EnergyModelGbXMLString(item):
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
        return openstudio.gbxml.GbXMLForwardTranslator().modelToGbXMLString(item)
    
    @staticmethod
    def EnergyModelIFCToOSM(model, filePath):
        """
        Parameters
        ----------
        model : TYPE
            DESCRIPTION.
        filepath : TYPE
            DESCRIPTION.

        Returns
        -------
        osCondition : TYPE
            DESCRIPTION.

        """
        # model = item[0]
        # filePath = item[1]
        # Make sure the file extension is .OSM
        ext = filePath[len(filePath)-4:len(filePath)]
        if ext.lower() != ".osm":
            filePath = filePath+".osm"
        osCondition = False
        osPath = openstudio.openstudioutilitiescore.toPath(filePath)
        osCondition = model.save(osPath, True)
        return osCondition
    
    @staticmethod
    def EnergyModelQuery(model, EPReportName, EPReportForString, EPTableName,
                         EPColumnName, EPRowName, EPUnits):
        """
        Parameters
        ----------
        model : TYPE
            DESCRIPTION.
        EPReportName : TYPE
            DESCRIPTION.
        EPReportForString : TYPE
            DESCRIPTION.
        EPTableName : TYPE
            DESCRIPTION.
        EPColumnName : TYPE
            DESCRIPTION.
        EPRowName : TYPE
            DESCRIPTION.
        EpUnits : TYPE
            DESCRIPTION.

        Returns
        -------
        doubleValue : TYPE
            DESCRIPTION.

        """
        # model = item[0]
        # EPReportName = item[1]
        # EPReportForString = item[2]
        # EPTableName = item[3]
        # EPColumnName = item[4]
        # EPRowName = item[5]
        # EPUnits = item[6]
        
        def doubleValueFromQuery(sqlFile, EPReportName, EPReportForString,
                                 EPTableName, EPColumnName, EPRowName,
                                 EPUnits):
            doubleValue = 0.0
            query = "SELECT Value FROM tabulardatawithstrings WHERE ReportName='" + EPReportName + "' AND ReportForString='" + EPReportForString + "' AND TableName = '" + EPTableName + "' AND RowName = '" + EPRowName + "' AND ColumnName= '" + EPColumnName + "' AND Units='" + EPUnits + "'";
            osOptionalDoubleValue = sqlFile.execAndReturnFirstDouble(query)
            if (osOptionalDoubleValue.is_initialized()):
                doubleValue = osOptionalDoubleValue.get()
            else:
                raise Exception("Failed to get a double value from the SQL file.")
            return doubleValue
        
        sqlFile = model.sqlFile().get()
        doubleValue = doubleValueFromQuery(sqlFile, EPReportName,
                                           EPReportForString, EPTableName,
                                           EPColumnName, EPRowName, EPUnits)
        return doubleValue
    
    @staticmethod
    def EnergyModelReportNames(model):
        """
        Parameters
        ----------
        model : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # model = item[0]
        sql = model.sqlFile().get()
        reportNames = sql.execAndReturnVectorOfString("SELECT ReportName FROM tabulardatawithstrings").get()
        return list(OrderedDict( (x,1) for x in reportNames ).keys()) #Making a unique list and keeping its order

    @staticmethod
    def EnergyModelRowNames(model, reportName, tableName):
        """
        Parameters
        ----------
        model : TYPE
            DESCRIPTION.
        reportName : TYPE
            DESCRIPTION.
        tableName : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # model = item[0]
        # reportName = item[1]
        # tableName = item[2]
        sql = model.sqlFile().get()
        query = "SELECT RowName FROM tabulardatawithstrings WHERE ReportName = '"+reportName+"' AND TableName = '"+tableName+"'"
        columnNames = sql.execAndReturnVectorOfString(query).get()
        return list(OrderedDict( (x,1) for x in columnNames ).keys()) #Making a unique list and keeping its order

    @staticmethod
    def EnergyModelRunSimulation(model, weatherFile, osBinaryPath,
                                 outputFolder, run):
        """
        Parameters
        ----------
        model : TYPE
            DESCRIPTION.
        weatherFile : TYPE
            DESCRIPTION.
        osBinaryPath : TYPE
            DESCRIPTION.
        outputFolder : TYPE
            DESCRIPTION.
        run : TYPE
            DESCRIPTION.

        Returns
        -------
        model : TYPE
            DESCRIPTION.

        """
        # model = item[0]
        # weatherFile = item[1]
        # osBinaryPath = item[2]
        # outputFolder = item[3]
        # run = item[4]
        if not run:
            return None
        utcnow = datetime.utcnow()
        timestamp = utcnow.strftime("UTC-%Y-%m-%d-%H-%M-%S")
        outputFolder = os.path.join(outputFolder, timestamp)
        os.mkdir(outputFolder)
        osmPath = outputFolder + "/" + model.getBuilding().name().get() + ".osm"
        model.save(openstudio.openstudioutilitiescore.toPath(osmPath), True)
        oswPath = os.path.join(outputFolder, model.getBuilding().name().get() + ".osw")
        print("oswPath = "+oswPath)
        workflow = model.workflowJSON()
        workflow.setSeedFile(openstudio.openstudioutilitiescore.toPath(osmPath))
        print("Seed File Set")
        workflow.setWeatherFile(openstudio.openstudioutilitiescore.toPath(weatherFile))
        print("Weather File Set")
        workflow.saveAs(openstudio.openstudioutilitiescore.toPath(oswPath))
        print("OSW File Saved")
        cmd = osBinaryPath+" run -w " + "\"" + oswPath + "\""
        os.system(cmd)
        print("Simulation DONE")
        sqlPath = os.path.join(os.path.join(outputFolder,"run"), "eplusout.sql")
        print("sqlPath = "+sqlPath)
        osSqlFile = openstudio.SqlFile(openstudio.openstudioutilitiescore.toPath(sqlPath))
        model.setSqlFile(osSqlFile)
        return model
    
    @staticmethod
    def EnergyModelSpaceTypes(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        types = item.getSpaceTypes()
        names = []
        colors = []
        for aType in types:
            names.append(aType.name().get())
            red = aType.renderingColor().get().renderingRedValue()
            green = aType.renderingColor().get().renderingGreenValue()
            blue = aType.renderingColor().get().renderingBlueValue()
            colors.append([red,green,blue])
        return [types, names, colors]
    
    @staticmethod
    def EnergyModelSqlFile(item):
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
        return item.sqlFile().get()
    
    @staticmethod
    def EnergyModelTableNames(model, reportName):
        """
        Parameters
        ----------
        model : TYPE
            DESCRIPTION.
        reportName : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # model = item[0]
        # reportName = item[1]
        sql = model.sqlFile().get()
        tableNames = sql.execAndReturnVectorOfString("SELECT TableName FROM tabulardatawithstrings WHERE ReportName='"+reportName+"'").get()
        return list(OrderedDict( (x,1) for x in tableNames ).keys()) #Making a unique list and keeping its order

    @staticmethod
    def EnergyModelTopologies(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        def surfaceToFace(surface):
            surfaceEdges = []
            surfaceVertices = surface.vertices()
            for i in range(len(surfaceVertices)-1):
                sv = topologic.Vertex.ByCoordinates(surfaceVertices[i].x(), surfaceVertices[i].y(), surfaceVertices[i].z())
                ev = topologic.Vertex.ByCoordinates(surfaceVertices[i+1].x(), surfaceVertices[i+1].y(), surfaceVertices[i+1].z())
                edge = topologic.Edge.ByStartVertexEndVertex(sv, ev)
                surfaceEdges.append(edge)
            sv = topologic.Vertex.ByCoordinates(surfaceVertices[len(surfaceVertices)-1].x(), surfaceVertices[len(surfaceVertices)-1].y(), surfaceVertices[len(surfaceVertices)-1].z())
            ev = topologic.Vertex.ByCoordinates(surfaceVertices[0].x(), surfaceVertices[0].y(), surfaceVertices[0].z())
            edge = topologic.Edge.ByStartVertexEndVertex(sv, ev)
            surfaceEdges.append(edge)
            surfaceWire = topologic.Wire.ByEdges(surfaceEdges)
            internalBoundaries = []
            surfaceFace = topologic.Face.ByExternalInternalBoundaries(surfaceWire, internalBoundaries)
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
        
        spaces = item.getSpaces()
        vertexIndex = 0
        cells = []
        apertures = []
        shadingFaces = []
        shadingSurfaces = item.getShadingSurfaces()
        for aShadingSurface in shadingSurfaces:
            shadingFaces.append(surfaceToFace(aShadingSurface))
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
            spaceCell = topologic.Cell.ByFaces(spaceFaces)
            print(count, spaceCell)
            if not spaceCell:
                spaceCell = topologic.Shell.ByFaces(spaceFaces)
            if not spaceCell:
                spaceCell = topologic.Cluster.ByTopologies(spaceFaces)
            if spaceCell:
                # Set Dictionary for Cell
                stl_keys = []
                stl_keys.append("TOPOLOGIC_id")
                stl_keys.append("TOPOLOGIC_name")
                stl_keys.append("TOPOLOGIC_type")
                stl_keys.append("TOPOLOGIC_color")
                stl_values = []
                spaceID = str(aSpace.handle()).replace('{','').replace('}','')
                stl_values.append(topologic.StringAttribute(spaceID))
                stl_values.append(topologic.StringAttribute(aSpace.name().get()))
                spaceTypeName = "Unknown"
                red = 255
                green = 255
                blue = 255
                if (aSpace.spaceType().is_initialized()):
                    if(aSpace.spaceType().get().name().is_initialized()):
                        spaceTypeName = aSpace.spaceType().get().name().get()
                    if(aSpace.spaceType().get().renderingColor()):
                        red = aSpace.spaceType().get().renderingColor().get().renderingRedValue()
                        green = aSpace.spaceType().get().renderingColor().get().renderingGreenValue()
                        blue = aSpace.spaceType().get().renderingColor().get().renderingBlueValue()
                stl_values.append(topologic.StringAttribute(spaceTypeName))
                l = []
                l.append(topologic.IntAttribute(red))
                l.append(topologic.IntAttribute(green))
                l.append(topologic.IntAttribute(blue))
                stl_values.append(topologic.ListAttribute(l))
                dict = topologic.Dictionary.ByKeysValues(stl_keys, stl_values)
                _ = spaceCell.SetDictionary(dict)
                cells.append(spaceCell)
        return [cells, apertures, shadingFaces]
    
    @staticmethod
    def EnergyModelUnits(model, reportName, tableName, columnName):
        """
        Parameters
        ----------
        model : TYPE
            DESCRIPTION.
        reportName : TYPE
            DESCRIPTION.
        tableName : TYPE
            DESCRIPTION.
        columnName : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        units : TYPE
            DESCRIPTION.

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
            raise Exception("Failed to get units from the SQL file.")
        return units
    
