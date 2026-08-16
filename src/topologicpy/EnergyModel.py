# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3.0 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from topologicpy.Core import Core
import shutil
import math
from collections import OrderedDict
import os
from os.path import exists
from datetime import datetime, timezone
import warnings

class EnergyModel:
    '''
    @staticmethod
    def ByOSMFile(file):
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
        if not file:
            print("EnergyModel.ByOSMFile - Error: The input path is not valid. Returning None.")
            return None
        osModel = file.read()
        if osModel.isNull():
            print("EnergyModel.ByOSMFile - Error: The openstudio model is null. Returning None.")
            return None
        else:
            osModel = osModel.get()
        return osModel

    '''

    @staticmethod
    def _ImportOpenStudio(methodName: str = "EnergyModel", silent: bool = False):
        """
        Imports OpenStudio without attempting to install it. Returns None if unavailable.
        """
        try:
            import openstudio
            try:
                openstudio.Logger.instance().standardOutLogger().setLogLevel(openstudio.Fatal)
            except Exception:
                pass
            return openstudio
        except Exception:
            if not silent:
                warnings.warn(f"{methodName} - Error: Could not import openstudio. Please install openstudio manually. Returning None.")
            return None

    @staticmethod
    def _OptionalGet(opt, default=None):
        """
        Safely unwraps OpenStudio optional values and option-like objects.
        """
        try:
            if opt is None:
                return default
            if hasattr(opt, "is_initialized"):
                return opt.get() if opt.is_initialized() else default
            if hasattr(opt, "isNull"):
                return default if opt.isNull() else opt.get()
            if hasattr(opt, "get") and callable(getattr(opt, "get", None)):
                return opt.get()
            return opt
        except Exception:
            return default

    @staticmethod
    def _OSPath(path_str, openstudio):
        """
        Converts a string path to an OpenStudio path object when the binding exposes one.
        """
        if path_str is None:
            return None
        for candidate in (
            lambda p: openstudio.openstudioutilitiescore.toPath(p),
            lambda p: openstudio.toPath(p),
            lambda p: openstudio.path(p),
        ):
            try:
                return candidate(path_str)
            except Exception:
                pass
        return path_str

    @staticmethod
    def _SQLFile(model):
        """
        Returns the initialized OpenStudio SQL file attached to the model, or None.
        """
        try:
            sql_opt = model.sqlFile()
        except Exception:
            return None
        return EnergyModel._OptionalGet(sql_opt, None)

    @staticmethod
    def _VectorOfString(optional_vector):
        """
        Safely converts an OpenStudio OptionalVectorString-like result to a list.
        """
        value = EnergyModel._OptionalGet(optional_vector, [])
        try:
            return list(value)
        except Exception:
            return []

    @staticmethod
    def _ObjectName(model_object, fallback=""):
        """
        Safely returns an OpenStudio object's name.
        """
        try:
            return EnergyModel._OptionalGet(model_object.name(), fallback)
        except Exception:
            return fallback

    @staticmethod
    def _RenderingColorRGB(model_object, default=None):
        """
        Safely returns [red, green, blue] rendering values for an OpenStudio object.
        """
        if default is None:
            default = [255, 255, 255]
        try:
            color = EnergyModel._OptionalGet(model_object.renderingColor(), None)
            if color is None:
                return list(default)
            return [
                color.renderingRedValue(),
                color.renderingGreenValue(),
                color.renderingBlueValue(),
            ]
        except Exception:
            return list(default)

    @staticmethod
    def ByOSMPath(path: str):
        """
        Creates an EnergyModel from the input OSM file path.
        """
        openstudio = EnergyModel._ImportOpenStudio("EnergyModel.ByOSMPath")
        if openstudio is None:
            return None

        if not isinstance(path, str) or not path:
            print("EnergyModel.ByOSMPath - Error: The input path is not valid. Returning None.")
            return None
        if not os.path.exists(path):
            print("EnergyModel.ByOSMPath - Error: The input path does not exist. Returning None.")
            return None

        try:
            translator = openstudio.osversion.VersionTranslator()
            osmPath = EnergyModel._OSPath(path, openstudio)
            osModel_opt = translator.loadModel(osmPath)
            osModel = EnergyModel._OptionalGet(osModel_opt, None)
        except Exception:
            osModel = None

        if osModel is None:
            print("EnergyModel.ByOSMPath - Error: The OpenStudio model is null or could not be loaded. Returning None.")
            return None
        return osModel

    @staticmethod
    def ByTopology(building,
                shadingSurfaces  = None,
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
                spaceTypeKey : str = "TOPOLOGIC_type",
                mantissa : int = 6,
                tolerance : float = 0.0001):
        """
            Creates an EnergyModel from the input topology and parameters.

        Parameters
        ----------
        building : topologic_core.CellComplex or topologic_core.Cell
            The input building topology.
        shadingSurfaces : topologic_core.Topology , optional
            The input topology for shading surfaces. Default is None.
        osModelPath : str , optional
            The path to the template OSM file. Default is "./assets/EnergyModel/OSMTemplate-OfficeBuilding-3.10.0.osm".
        weatherFilePath : str , optional
            The input energy plus weather (epw) file. Default is "./assets/EnergyModel/GBR_London.Gatwick.037760_IWEC.epw".
        designDayFilePath : str , optional
            The input design day (ddy) file path. Default is "./assets/EnergyModel/GBR_London.Gatwick.037760_IWEC.ddy".
        floorLevels : list , optional
            The list of floor level Z heights including the lowest most and the highest most levels. If set to None, this method will attempt to
            find the floor levels from the horizontal faces of the input topology.
        buildingName : str , optional
            The desired name of the building. Default is "TopologicBuilding".
        buildingType : str , optional
            The building type. Default is "Commercial".
        defaultSpaceType : str , optional
            The default space type to apply to spaces that do not have a type assigned in their dictionary.
        northAxis : float , optional
            The counter-clockwise angle in degrees from the positive Y-axis representing the direction of the north axis. Default is 0.0.
        glazingRatio : float , optional
            The glazing ratio (ratio of windows to wall) to use for exterior vertical walls that do not have apertures. If you do not wish to use a glazing ratio, set it to 0. Default is 0.
        coolingTemp : float , optional
            The desired temperature in degrees at which the cooling system should activate. Default is 25.0.
        heatingTemp : float , optional
            The desired temperature in degrees at which the heating system should activate. Default is 20.0.
        spaceNameKey : str , optional
            The dictionary key to use to find the space name value. Default is "TOPOLOGIC_name".
        spaceTypeKey : str , optional
            The dictionary key to use to find the space type value. Default is "TOPOLOGIC_type".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        openstudio.openstudiomodelcore.Model
            The created OSM model.

        """
        import math
        import os
        import warnings

        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        def getKeyName(d, keyName):
            if d is None or keyName is None:
                return None
            try:
                keys = Dictionary.Keys(d)
            except Exception:
                return None
            for key in keys:
                if str(key).lower() == str(keyName).lower():
                    return key
            return None

        def createUniqueName(name, nameList, number):
            if name is None:
                return None
            name = str(name)
            if number > 9999:
                return name + "_9999"
            if name not in nameList:
                return name
            candidate = name + "_" + "{:04d}".format(number)
            if candidate not in nameList:
                return candidate
            return createUniqueName(name, nameList, number + 1)

        def getFloorLevels(building):
            from topologicpy.Vertex import Vertex
            from topologicpy.Cell import Cell
            from topologicpy.CellComplex import CellComplex

            if Topology.IsInstance(building, "CellComplex"):
                d = CellComplex.Decompose(building)
                hf = d['bottomHorizontalFaces'] + d['internalHorizontalFaces'] + d['topHorizontalFaces']
            elif Topology.IsInstance(building, "Cell"):
                d = Cell.Decompose(building)
                hf = d['bottomHorizontalFaces'] + d['topHorizontalFaces']
            else:
                return None
            levels = [Vertex.Z(Topology.Centroid(f), mantissa=mantissa) for f in hf]
            levels = sorted(list(set(levels)))
            return levels

        def safe_optional_get(opt):
            try:
                if opt and opt.is_initialized():
                    return opt.get()
            except Exception:
                pass
            return None

        def safe_name(model_object, fallback=""):
            try:
                name_opt = model_object.name()
                if name_opt.is_initialized():
                    return name_opt.get()
            except Exception:
                pass
            return fallback

        def surface_tilt_degrees(osSurface, openstudio):
            try:
                up = openstudio.Vector3d(0, 0, 1)
                dot = osSurface.outwardNormal().dot(up)
                dot = max(-1.0, min(1.0, dot))
                return math.degrees(math.acos(dot))
            except Exception:
                return 90.0

        def vertices_to_point3d_list(vertices, openstudio):
            pts = []
            for v in vertices:
                pts.append(
                    openstudio.Point3d(
                        Vertex.X(v, mantissa=mantissa),
                        Vertex.Y(v, mantissa=mantissa),
                        Vertex.Z(v, mantissa=mantissa)
                    )
                )
            return pts

        def orient_surface_vertices(points, face_normal, surface_obj, openstudio):
            try:
                osFaceNormal = openstudio.Vector3d(face_normal[0], face_normal[1], face_normal[2])
                osFaceNormal.normalize()
                if osFaceNormal.dot(surface_obj.outwardNormal()) < 1e-6:
                    surface_obj.setVertices(list(reversed(points)))
            except Exception:
                pass

        def os_path(path_str, openstudio):
            try:
                return openstudio.openstudioutilitiescore.toPath(path_str)
            except Exception:
                pass
            try:
                return openstudio.toPath(path_str)
            except Exception:
                pass
            try:
                return openstudio.path(path_str)
            except Exception:
                pass
            raise RuntimeError(f"EnergyModel.ByTopology - Could not convert path to OpenStudio path: {path_str}")

        def first_or_none(seq):
            try:
                return seq[0] if len(seq) > 0 else None
            except Exception:
                return None

        openstudio = EnergyModel._ImportOpenStudio("EnergyModel.ByTopology")
        if openstudio is None:
            return None

        if not osModelPath:
            osModelPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "EnergyModel", "OSMTemplate-OfficeBuilding-3.10.0.osm")
        if not weatherFilePath:
            weatherFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "EnergyModel", "GBR_London.Gatwick.037760_IWEC.epw")
        if not designDayFilePath:
            designDayFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "EnergyModel", "GBR_London.Gatwick.037760_IWEC.ddy")

        if not os.path.exists(osModelPath):
            raise FileNotFoundError(f"EnergyModel.ByTopology - OSM template not found: {osModelPath}")
        if not os.path.exists(weatherFilePath):
            raise FileNotFoundError(f"EnergyModel.ByTopology - Weather file not found: {weatherFilePath}")
        if not os.path.exists(designDayFilePath):
            raise FileNotFoundError(f"EnergyModel.ByTopology - Design day file not found: {designDayFilePath}")

        if not Topology.IsInstance(building, "CellComplex") and not Topology.IsInstance(building, "Cell"):
            warnings.warn("EnergyModel.ByTopology - Error: The input building is not a valid Cell or CellComplex. Returning None.")
            return None

        # Load OSM safely
        osmFile = os_path(osModelPath, openstudio)
        translator = openstudio.osversion.VersionTranslator()
        model_opt = translator.loadModel(osmFile)
        osModel = EnergyModel._OptionalGet(model_opt, None)
        if osModel is None:
            raise RuntimeError(f"EnergyModel.ByTopology - Could not load OSM template: {osModelPath}")

        # Load EPW safely
        epwFile = os_path(weatherFilePath, openstudio)
        epw_opt = openstudio.openstudioutilitiesfiletypes.EpwFile.load(epwFile)
        osEPWFile = EnergyModel._OptionalGet(epw_opt, None)
        if osEPWFile is not None:
            openstudio.model.WeatherFile.setWeatherFile(osModel, osEPWFile)
        else:
            raise RuntimeError(f"EnergyModel.ByTopology - Could not load EPW weather file: {weatherFilePath}")

        # Load DDY safely
        ddyFile = os_path(designDayFilePath, openstudio)
        ddy_opt = openstudio.openstudioenergyplus.loadAndTranslateIdf(ddyFile)
        ddyModel = EnergyModel._OptionalGet(ddy_opt, None)
        if ddyModel is not None:
            for ddy in ddyModel.getObjectsByType(openstudio.IddObjectType("OS:SizingPeriod:DesignDay")):
                osModel.addObject(ddy.clone())
        else:
            raise RuntimeError(f"EnergyModel.ByTopology - Could not load DDY design day file: {designDayFilePath}")
        
        # Try to assign a default space type
        space_type_names = EnergyModel.SpaceTypeNames(osModel)
        if defaultSpaceType == None:
            for space_type_name in space_type_names:
                if "office" in space_type_name.lower() or "room" in space_type_name.lower():
                    defaultSpaceType = space_type_name
                    break
        if not defaultSpaceType in space_type_names:
            raise RuntimeError(f"EnergyModel.ByTopology - Default Space Type {defaultSpaceType} not found in OSM template.")
        osBuilding = osModel.getBuilding()
        if osBuilding is None:
            raise RuntimeError("EnergyModel.ByTopology - Could not retrieve Building object from OSM template.")

        if not floorLevels:
            floorLevels = getFloorLevels(building)
        if not floorLevels or len(floorLevels) < 2:
            raise RuntimeError("EnergyModel.ByTopology - Could not derive valid floor levels from the input topology.")

        floorLevels = sorted(list(set(floorLevels)))
        numberOfStories = len(floorLevels) - 1
        if numberOfStories < 1:
            raise RuntimeError("EnergyModel.ByTopology - The derived number of stories is less than 1.")

        osBuilding.setStandardsNumberOfStories(numberOfStories)

        floor_to_floor_height = (max(floorLevels) - min(floorLevels)) / numberOfStories if numberOfStories > 0 else 3.0
        if floor_to_floor_height <= tolerance:
            floor_to_floor_height = 3.0
        osBuilding.setNominalFloortoFloorHeight(floor_to_floor_height)

        defaultConstructionSets = list(osModel.getDefaultConstructionSets())
        if len(defaultConstructionSets) < 1:
            raise RuntimeError("EnergyModel.ByTopology - No DefaultConstructionSet found in OSM template.")
        osBuilding.setDefaultConstructionSet(defaultConstructionSets[0])

        defaultScheduleSets = list(osModel.getDefaultScheduleSets())
        if len(defaultScheduleSets) < 1:
            raise RuntimeError("EnergyModel.ByTopology - No DefaultScheduleSet found in OSM template.")
        osBuilding.setDefaultScheduleSet(defaultScheduleSets[0])

        osBuilding.setName(buildingName)
        try:
            osBuilding.setStandardsBuildingType(buildingType)
        except Exception:
            pass

        defaultSpaceTypeOpt = osModel.getSpaceTypeByName(defaultSpaceType)
        if not defaultSpaceTypeOpt.is_initialized():
            availableSpaceTypes = []
            for st in osModel.getSpaceTypes():
                try:
                    n = st.name()
                    if n.is_initialized():
                        availableSpaceTypes.append(n.get())
                except Exception:
                    pass
            raise RuntimeError(
                "EnergyModel.ByTopology - Could not find the requested SpaceType "
                f"'{defaultSpaceType}' in the OSM template. Available SpaceTypes: {availableSpaceTypes}"
            )
        defaultSpaceTypeObj = defaultSpaceTypeOpt.get()
        osBuilding.setSpaceType(defaultSpaceTypeObj)

        for storyNumber in range(numberOfStories):
            osBuildingStory = openstudio.model.BuildingStory(osModel)
            osBuildingStory.setName("STORY_" + str(storyNumber))
            osBuildingStory.setNominalZCoordinate(floorLevels[storyNumber])
            osBuildingStory.setNominalFloortoFloorHeight(floor_to_floor_height)

        try:
            osBuilding.setNorthAxis(northAxis)
        except Exception:
            pass

        heatingScheduleConstant = openstudio.model.ScheduleConstant(osModel)
        heatingScheduleConstant.setValue(heatingTemp)
        coolingScheduleConstant = openstudio.model.ScheduleConstant(osModel)
        coolingScheduleConstant.setValue(coolingTemp)

        osThermostat = openstudio.model.ThermostatSetpointDualSetpoint(osModel)
        osThermostat.setHeatingSetpointTemperatureSchedule(heatingScheduleConstant)
        osThermostat.setCoolingSetpointTemperatureSchedule(coolingScheduleConstant)

        osBuildingStorys = list(osModel.getBuildingStorys())
        osBuildingStorys.sort(key=lambda x: safe_optional_get(x.nominalZCoordinate()) if safe_optional_get(x.nominalZCoordinate()) is not None else -1e12)

        if len(osBuildingStorys) < 1:
            raise RuntimeError("EnergyModel.ByTopology - No BuildingStory objects were created.")

        interiorHorizontalConstruction = None
        try:
            dcs = defaultConstructionSets[0]
            interior_surface_cons_opt = dcs.defaultInteriorSurfaceConstructions()
            if interior_surface_cons_opt.is_initialized():
                isc = interior_surface_cons_opt.get()
                floor_con_opt = isc.floorConstruction()
                roof_con_opt = isc.roofCeilingConstruction()
                if floor_con_opt.is_initialized():
                    interiorHorizontalConstruction = floor_con_opt.get()
                elif roof_con_opt.is_initialized():
                    interiorHorizontalConstruction = roof_con_opt.get()
        except Exception:
            interiorHorizontalConstruction = None

        osSpaces = []
        spaceNames = []

        if Topology.IsInstance(building, "CellComplex"):
            building_cells = Topology.SubTopologies(building, "Cell")
        else:
            building_cells = [building]

        for spaceNumber, buildingCell in enumerate(building_cells):
            osSpace = openstudio.model.Space(osModel)
            osSpaceZ = Vertex.Z(Topology.CenterOfMass(buildingCell), mantissa=mantissa)

            selectedStory = osBuildingStorys[0]
            for story in osBuildingStorys:
                storyZ = safe_optional_get(story.nominalZCoordinate())
                storyH = safe_optional_get(story.nominalFloortoFloorHeight())
                if storyZ is None:
                    continue
                if storyH is None:
                    storyH = floor_to_floor_height
                if storyZ + storyH < osSpaceZ:
                    continue
                if storyZ <= osSpaceZ:
                    selectedStory = story
                break

            osSpace.setBuildingStory(selectedStory)

            cellDictionary = Topology.Dictionary(buildingCell)
            keys = Dictionary.Keys(cellDictionary) if cellDictionary is not None else []

            osSpaceName = None
            chosenSpaceTypeObj = defaultSpaceTypeObj

            if len(keys) > 0:
                keyType = getKeyName(cellDictionary, spaceTypeKey) if spaceTypeKey else getKeyName(cellDictionary, "type")
                osSpaceTypeName = Dictionary.ValueAtKey(cellDictionary, keyType) if keyType else defaultSpaceType
                if osSpaceTypeName:
                    sp_opt = osModel.getSpaceTypeByName(osSpaceTypeName)
                    if sp_opt.is_initialized():
                        chosenSpaceTypeObj = sp_opt.get()

                keyName = getKeyName(cellDictionary, spaceNameKey) if spaceNameKey else getKeyName(cellDictionary, "name")
                if keyName:
                    raw_name = Dictionary.ValueAtKey(cellDictionary, keyName)
                    osSpaceName = createUniqueName(raw_name, spaceNames, 1)

            if not osSpaceName:
                osSpaceName = "SPACE_" + "{:04d}".format(spaceNumber)

            osSpace.setName(osSpaceName)
            spaceNames.append(osSpaceName)

            if chosenSpaceTypeObj is not None:
                osSpace.setSpaceType(chosenSpaceTypeObj)

            cellFaces = Topology.SubTopologies(buildingCell, "Face")
            if cellFaces:
                for faceNumber, buildingFace in enumerate(cellFaces):
                    boundary = Face.ExternalBoundary(buildingFace)
                    if boundary is None:
                        continue

                    faceVertices = Topology.SubTopologies(boundary, "Vertex")
                    if not faceVertices or len(faceVertices) < 3:
                        continue

                    osFacePoints = vertices_to_point3d_list(faceVertices, openstudio)
                    osSurface = openstudio.model.Surface(osFacePoints, osModel)

                    faceNormal = Face.Normal(buildingFace, mantissa=mantissa)
                    orient_surface_vertices(osFacePoints, faceNormal, osSurface, openstudio)
                    osSurface.setSpace(osSpace)

                    faceCells = Topology.AdjacentTopologies(buildingFace, building, topologyType="cell")
                    if not isinstance(faceCells, list):
                        faceCells = []
                    tilt = surface_tilt_degrees(osSurface, openstudio)
                    space_name = safe_name(osSpace, f"SPACE_{spaceNumber:04d}")

                    if len(faceCells) == 1:  # Exterior surfaces
                        osSurface.setOutsideBoundaryCondition("Outdoors")
                        if tilt > 135 or tilt < 45:
                            osSurface.setSurfaceType("RoofCeiling")
                            osSurface.setOutsideBoundaryCondition("Outdoors")
                            osSurface.setName(space_name + "_TopHorizontalSlab_" + str(faceNumber))

                            try:
                                face_zs = [Vertex.Z(v) for v in Topology.SubTopologies(buildingFace, "Vertex")]
                                if len(face_zs) > 0 and max(face_zs) < 1e-6:
                                    osSurface.setSurfaceType("Floor")
                                    osSurface.setOutsideBoundaryCondition("Ground")
                                    osSurface.setName(space_name + "_BottomHorizontalSlab_" + str(faceNumber))
                            except Exception:
                                pass
                        else:
                            osSurface.setSurfaceType("Wall")
                            osSurface.setOutsideBoundaryCondition("Outdoors")
                            osSurface.setName(space_name + "_ExternalVerticalFace_" + str(faceNumber))

                            faceDictionary = Topology.Dictionary(buildingFace)
                            apertures = Topology.Apertures(buildingFace)
                            if apertures and len(apertures) > 0:
                                for apertureFace in apertures:
                                    ap_boundary = Face.ExternalBoundary(apertureFace)
                                    if ap_boundary is None:
                                        continue
                                    ap_vertices = Topology.SubTopologies(ap_boundary, "Vertex")
                                    if not ap_vertices or len(ap_vertices) < 3:
                                        continue

                                    osSubSurfacePoints = vertices_to_point3d_list(ap_vertices, openstudio)
                                    osSubSurface = openstudio.model.SubSurface(osSubSurfacePoints, osModel)

                                    apertureFaceNormal = Face.Normal(apertureFace, mantissa=mantissa)
                                    orient_surface_vertices(osSubSurfacePoints, apertureFaceNormal, osSubSurface, openstudio)

                                    osSubSurface.setSubSurfaceType("FixedWindow")
                                    osSubSurface.setSurface(osSurface)
                            else:
                                faceGlazingRatio = None
                                if faceDictionary is not None:
                                    try:
                                        keys = Dictionary.Keys(faceDictionary)
                                        if 'TOPOLOGIC_glazing_ratio' in keys:
                                            faceGlazingRatio = Dictionary.ValueAtKey(faceDictionary, 'TOPOLOGIC_glazing_ratio')
                                    except Exception:
                                        faceGlazingRatio = None
                                try:
                                    faceGlazingRatio = float(faceGlazingRatio) if faceGlazingRatio is not None else None
                                except Exception:
                                    faceGlazingRatio = None
                                if faceGlazingRatio is not None and faceGlazingRatio >= 0.01:
                                    try:
                                        osSurface.setWindowToWallRatio(faceGlazingRatio)
                                    except Exception:
                                        pass
                                elif glazingRatio > 0.01:
                                    try:
                                        osSurface.setWindowToWallRatio(glazingRatio)
                                    except Exception:
                                        pass

                    else:  # Interior surfaces
                        if tilt > 135:
                            osSurface.setSurfaceType("Floor")
                            osSurface.setName(space_name + "_InternalHorizontalFace_" + str(faceNumber))
                            if interiorHorizontalConstruction is not None:
                                try:
                                    osSurface.setConstruction(interiorHorizontalConstruction)
                                except Exception:
                                    pass
                        elif tilt < 40:
                            osSurface.setSurfaceType("RoofCeiling")
                            osSurface.setName(space_name + "_InternalHorizontalFace_" + str(faceNumber))
                            if interiorHorizontalConstruction is not None:
                                try:
                                    osSurface.setConstruction(interiorHorizontalConstruction)
                                except Exception:
                                    pass
                        else:
                            osSurface.setSurfaceType("Wall")
                            osSurface.setName(space_name + "_InternalVerticalFace_" + str(faceNumber))

                        apertures = Topology.Apertures(buildingFace)
                        if apertures and len(apertures) > 0:
                            for apertureFace in apertures:
                                ap_boundary = Face.ExternalBoundary(apertureFace)
                                if ap_boundary is None:
                                    continue
                                ap_vertices = Topology.SubTopologies(ap_boundary, "Vertex")
                                if not ap_vertices or len(ap_vertices) < 3:
                                    continue

                                osSubSurfacePoints = vertices_to_point3d_list(ap_vertices, openstudio)
                                osSubSurface = openstudio.model.SubSurface(osSubSurfacePoints, osModel)

                                apertureFaceNormal = Face.Normal(apertureFace, mantissa=mantissa)
                                orient_surface_vertices(osSubSurfacePoints, apertureFaceNormal, osSubSurface, openstudio)

                                osSubSurface.setSubSurfaceType("Door")
                                osSubSurface.setSurface(osSurface)

            osThermalZone = openstudio.model.ThermalZone(osModel)
            cellVolume = Cell.Volume(buildingCell, mantissa=mantissa)
            if cellVolume is not None:
                try:
                    osThermalZone.setVolume(cellVolume)
                except Exception:
                    pass
            osThermalZone.setName(osSpaceName + "_THERMAL_ZONE")
            osThermalZone.setUseIdealAirLoads(True)
            if cellVolume is not None:
                try:
                    osThermalZone.setVolume(cellVolume)
                except Exception:
                    pass
            osThermalZone.setThermostatSetpointDualSetpoint(osThermostat)
            osSpace.setThermalZone(osThermalZone)

            for x in osSpaces:
                try:
                    if osSpace.boundingBox().intersects(x.boundingBox()):
                        osSpace.matchSurfaces(x)
                except Exception:
                    pass
            osSpaces.append(osSpace)

        if shadingSurfaces:
            osShadingGroup = openstudio.model.ShadingSurfaceGroup(osModel)
            for faceIndex, shadingFace in enumerate(Topology.SubTopologies(shadingSurfaces, "Face")):
                boundary = Face.ExternalBoundary(shadingFace)
                if boundary is None:
                    continue
                shadingVertices = Topology.SubTopologies(boundary, "Vertex")
                if not shadingVertices or len(shadingVertices) < 3:
                    continue

                facePoints = vertices_to_point3d_list(shadingVertices, openstudio)
                aShadingSurface = openstudio.model.ShadingSurface(facePoints, osModel)

                faceNormal = Face.Normal(shadingFace, mantissa=mantissa)
                try:
                    osFaceNormal = openstudio.Vector3d(faceNormal[0], faceNormal[1], faceNormal[2])
                    osFaceNormal.normalize()
                    if osFaceNormal.dot(aShadingSurface.outwardNormal()) < 0:
                        aShadingSurface.setVertices(list(reversed(facePoints)))
                except Exception:
                    pass

                aShadingSurface.setName("SHADINGSURFACE_" + str(faceIndex))
                aShadingSurface.setShadingSurfaceGroup(osShadingGroup)

        osModel.purgeUnusedResourceObjects()
        return osModel

    @staticmethod
    def ColumnNames(model, reportName, tableName):
        """
        Returns the list of column names given an OSM model, report name, and table name.
        """
        sql = EnergyModel._SQLFile(model)
        if sql is None:
            return []
        query = "SELECT ColumnName FROM tabulardatawithstrings WHERE ReportName = '" + str(reportName) + "' AND TableName = '" + str(tableName) + "'"
        columnNames = EnergyModel._VectorOfString(sql.execAndReturnVectorOfString(query))
        return list(OrderedDict((x, 1) for x in columnNames).keys())

    @staticmethod
    def DefaultConstructionSets(model):
        """
        Returns the default construction sets in the input OSM model.
        """
        try:
            sets = list(model.getDefaultConstructionSets())
        except Exception:
            sets = []
        names = [EnergyModel._ObjectName(aSet, "") for aSet in sets]
        return [sets, names]
    
    @staticmethod
    def DefaultScheduleSets(model):
        """
        Returns the default schedule sets found in the input OSM model.
        """
        try:
            sets = list(model.getDefaultScheduleSets())
        except Exception:
            sets = []
        names = [EnergyModel._ObjectName(aSet, "") for aSet in sets]
        return [sets, names]
    
    @staticmethod
    def ExportToGBXML(model, path, overwrite=False):
        """
        Exports the input OSM model to a GBXML file.
        """
        openstudio = EnergyModel._ImportOpenStudio("EnergyModel.ExportToGBXML")
        if openstudio is None:
            return None
        if not isinstance(path, str) or not path:
            print("EnergyModel.ExportToGBXML - Error: The input path is not a valid string. Returning None.")
            return None
        if not path.lower().endswith(".xml"):
            path = path + ".xml"
        if not overwrite and exists(path):
            print("EnergyModel.ExportToGBXML - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        try:
            os_path = EnergyModel._OSPath(path, openstudio)
            try:
                return openstudio.gbxml.GbXMLForwardTranslator().modelToGbXML(model, os_path)
            except TypeError:
                return openstudio.gbxml.GbXMLForwardTranslator().modelToGbXML(model, path)
        except Exception as e:
            warnings.warn(f"EnergyModel.ExportToGBXML - Error: Could not export GBXML ({e}). Returning None.")
            return None

    @staticmethod
    def ExportToOSM(model, path, overwrite=False):
        """
        Exports the input OSM model to an OSM file.
        """
        openstudio = EnergyModel._ImportOpenStudio("EnergyModel.ExportToOSM")
        if openstudio is None:
            return None
        if not isinstance(path, str) or not path:
            print("EnergyModel.ExportToOSM - Error: The input path is not a valid string. Returning None.")
            return None
        if not path.lower().endswith(".osm"):
            path = path + ".osm"
        if not overwrite and exists(path):
            print("EnergyModel.ExportToOSM - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        try:
            os_path = EnergyModel._OSPath(path, openstudio)
            try:
                return model.save(os_path, overwrite)
            except TypeError:
                return model.save(path, overwrite)
        except Exception as e:
            warnings.warn(f"EnergyModel.ExportToOSM - Error: Could not export OSM ({e}). Returning None.")
            return None
    
    @staticmethod
    def GBXMLString(model):
        """
        Returns the GBXML string of the input OSM model.
        """
        openstudio = EnergyModel._ImportOpenStudio("EnergyModel.GBXMLString")
        if openstudio is None:
            return None
        try:
            return openstudio.gbxml.GbXMLForwardTranslator().modelToGbXMLString(model)
        except Exception as e:
            warnings.warn(f"EnergyModel.GBXMLString - Error: Could not create GBXML string ({e}). Returning None.")
            return None
    
    @staticmethod
    def Query(model,
              reportName: str = "HVACSizingSummary",
              reportForString: str = "Entire Facility",
              tableName: str = "Zone Sensible Cooling",
              columnName: str = "Calculated Design Load",
              rowNames: list = None,
              units: str = "W"):
        """
        Queries the model for values.
        """
        if rowNames is None:
            rowNames = []
        elif not isinstance(rowNames, list):
            rowNames = [rowNames]

        sqlFile = EnergyModel._SQLFile(model)
        if sqlFile is None:
            return []

        def doubleValueFromQuery(sqlFile, reportName, reportForString, tableName, columnName, rowName, units):
            query = (
                "SELECT Value FROM tabulardatawithstrings WHERE "
                "ReportName='" + str(reportName) + "' AND "
                "ReportForString='" + str(reportForString) + "' AND "
                "TableName = '" + str(tableName) + "' AND "
                "RowName = '" + str(rowName) + "' AND "
                "ColumnName= '" + str(columnName) + "' AND "
                "Units='" + str(units) + "'"
            )
            try:
                osOptionalDoubleValue = sqlFile.execAndReturnFirstDouble(query)
                if osOptionalDoubleValue.is_initialized():
                    return osOptionalDoubleValue.get()
            except Exception:
                pass
            return None

        return [doubleValueFromQuery(sqlFile, reportName, reportForString, tableName, columnName, rowName, units) for rowName in rowNames]
    
    @staticmethod
    def ReportNames(model):
        """
        Returns the report names found in the input OSM model.
        """
        sql = EnergyModel._SQLFile(model)
        if sql is None:
            return []
        reportNames = EnergyModel._VectorOfString(sql.execAndReturnVectorOfString("SELECT ReportName FROM tabulardatawithstrings"))
        return list(OrderedDict((x, 1) for x in reportNames).keys())

    @staticmethod
    def RowNames(model, reportName, tableName):
        """
        Returns the list of row names given an OSM model, report name, and table name.
        """
        sql = EnergyModel._SQLFile(model)
        if sql is None:
            return []
        query = "SELECT RowName FROM tabulardatawithstrings WHERE ReportName = '" + str(reportName) + "' AND TableName = '" + str(tableName) + "'"
        rowNames = EnergyModel._VectorOfString(sql.execAndReturnVectorOfString(query))
        return list(OrderedDict((x, 1) for x in rowNames).keys())

    @staticmethod
    def Run(model, weatherFilePath: str = None, osBinaryPath: str = None, outputFolder: str = None, removeFiles: bool = False):
        """
        Runs an energy simulation.
        """
        import subprocess
        import time

        openstudio = EnergyModel._ImportOpenStudio("EnergyModel.Run")
        if openstudio is None:
            return None
        if model is None:
            warnings.warn("EnergyModel.Run - Error: The input model is None. Returning None.")
            return None

        def deleteOldFiles(path):
            if not path or not os.path.isdir(path):
                return None
            onemonth = time.time() - 30 * 86400
            try:
                for filename in os.listdir(path):
                    full_path = os.path.join(path, filename)
                    if os.path.getmtime(full_path) < onemonth:
                        if os.path.isfile(full_path):
                            os.remove(full_path)
                        elif os.path.isdir(full_path):
                            shutil.rmtree(full_path)
            except Exception:
                pass

        if not weatherFilePath:
            weatherFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "EnergyModel", "GBR_London.Gatwick.037760_IWEC.epw")
        if not os.path.exists(weatherFilePath):
            warnings.warn(f"EnergyModel.Run - Error: Weather file not found: {weatherFilePath}. Returning None.")
            return None

        if not osBinaryPath:
            osBinaryPath = shutil.which("openstudio")
        if not osBinaryPath:
            warnings.warn("EnergyModel.Run - Error: Could not find the OpenStudio executable. Returning None.")
            return None

        utcnow = datetime.now(timezone.utc)
        timestamp = utcnow.strftime("UTC-%Y-%m-%d-%H-%M-%S")
        if not outputFolder:
            outputFolder = os.path.join(os.path.expanduser("~"), "EnergyModels")

        if removeFiles:
            deleteOldFiles(outputFolder)

        outputFolder = os.path.join(outputFolder, timestamp)
        os.makedirs(outputFolder, exist_ok=True)

        try:
            try:
                building_name = model.getBuilding().name().get()
            except Exception:
                building_name = "TopologicBuilding"

            osmPath = os.path.join(outputFolder, building_name + ".osm")
            oswPath = os.path.join(outputFolder, building_name + ".osw")

            try:
                model.save(EnergyModel._OSPath(osmPath, openstudio), True)
            except TypeError:
                model.save(osmPath, True)

            workflow = model.workflowJSON()
            try:
                workflow.setSeedFile(EnergyModel._OSPath(osmPath, openstudio))
            except TypeError:
                workflow.setSeedFile(osmPath)

            try:
                workflow.setWeatherFile(EnergyModel._OSPath(weatherFilePath, openstudio))
            except TypeError:
                workflow.setWeatherFile(weatherFilePath)

            try:
                workflow.saveAs(EnergyModel._OSPath(oswPath, openstudio))
            except TypeError:
                workflow.saveAs(oswPath)

            cmd = [osBinaryPath, "run", "-w", oswPath]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                warnings.warn("EnergyModel.Run - Error: OpenStudio simulation failed. Returning None.")
                return None

            sqlPath = os.path.join(outputFolder, "run", "eplusout.sql")
            if not os.path.exists(sqlPath):
                warnings.warn("EnergyModel.Run - Error: Simulation SQL file was not created. Returning None.")
                return None

            try:
                osSqlFile = openstudio.SqlFile(EnergyModel._OSPath(sqlPath, openstudio))
            except TypeError:
                osSqlFile = openstudio.SqlFile(sqlPath)
            model.setSqlFile(osSqlFile)
            return model
        finally:
            pass
    
    @staticmethod
    def SpaceDictionaries(model):
        """
        Return the space dictionaries found in the input OSM model.
        """
        try:
            types = list(model.getSpaceTypes())
        except Exception:
            types = []
        names = []
        colors = []
        for aType in types:
            names.append(EnergyModel._ObjectName(aType, ""))
            colors.append(EnergyModel._RenderingColorRGB(aType))
        return {"types": types, "names": names, "colors": colors}
    
    @staticmethod
    def SpaceTypes(model):
        """
        Return the space types found in the input OSM model.
        """
        try:
            return list(model.getSpaceTypes())
        except Exception:
            return []
    
    @staticmethod
    def SpaceTypeNames(model):
        """
        Return the space type names found in the input OSM model.
        """
        try:
            types = list(model.getSpaceTypes())
        except Exception:
            types = []
        return [EnergyModel._ObjectName(aType, "") for aType in types]
    
    @staticmethod
    def SpaceColors(model):
        """
        Return the space colors found in the input OSM model.
        """
        try:
            types = list(model.getSpaceTypes())
        except Exception:
            types = []
        return [EnergyModel._RenderingColorRGB(aType) for aType in types]
    
    @staticmethod
    def SqlFile(model):
        """
        Returns the SQL file found in the input OSM model.
        """
        return EnergyModel._SQLFile(model)
    
    @staticmethod
    def TableNames(model, reportName):
        """
        Returns the table names found in the input OSM model and report name.
        """
        sql = EnergyModel._SQLFile(model)
        if sql is None:
            return []
        query = "SELECT TableName FROM tabulardatawithstrings WHERE ReportName='" + str(reportName) + "'"
        tableNames = EnergyModel._VectorOfString(sql.execAndReturnVectorOfString(query))
        return list(OrderedDict((x, 1) for x in tableNames).keys())

    @staticmethod
    def Topologies(model, tolerance=0.0001):
        """
        Returns the topologies found in the input OSM model.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Aperture import Aperture
        from topologicpy.Context import Context
        from topologicpy.Topology import Topology

        def point_xyz(point):
            values = []
            for name in ("x", "y", "z"):
                attr = getattr(point, name, None)
                if callable(attr):
                    values.append(attr())
                elif attr is not None:
                    values.append(attr)
                else:
                    values.append(None)
            if all(v is not None for v in values):
                return values
            try:
                return [point[0], point[1], point[2]]
            except Exception:
                return [None, None, None]

        def transform_topology(topology, osTransformation):
            if topology is None:
                return None
            try:
                osTranslation = osTransformation.translation()
                osMatrix = osTransformation.rotationMatrix()
                return Core.TopologyUtility.Transform(
                    topology,
                    osTranslation.x(), osTranslation.y(), osTranslation.z(),
                    osMatrix[0, 0], osMatrix[0, 1], osMatrix[0, 2],
                    osMatrix[1, 0], osMatrix[1, 1], osMatrix[1, 2],
                    osMatrix[2, 0], osMatrix[2, 1], osMatrix[2, 2],
                )
            except Exception:
                return topology

        def surfaceToFace(surface):
            try:
                surfaceVertices = list(surface.vertices())
            except Exception:
                return None
            if not surfaceVertices or len(surfaceVertices) < 3:
                return None

            surfaceEdges = []
            n = len(surfaceVertices)
            for i in range(n):
                j = (i + 1) % n
                sx, sy, sz = point_xyz(surfaceVertices[i])
                ex, ey, ez = point_xyz(surfaceVertices[j])
                if None in [sx, sy, sz, ex, ey, ez]:
                    continue
                sv = Vertex.ByCoordinates(sx, sy, sz)
                ev = Vertex.ByCoordinates(ex, ey, ez)
                edge = Edge.ByStartVertexEndVertex(sv, ev, tolerance=tolerance, silent=True)
                if edge:
                    surfaceEdges.append(edge)

            if len(surfaceEdges) < 3:
                return None
            surfaceWire = Wire.ByEdges(surfaceEdges, tolerance=tolerance)
            if surfaceWire is None:
                return None
            return Face.ByWires(surfaceWire, [], tolerance=tolerance)

        def addApertures(face, face_apertures):
            if face is None or not isinstance(face_apertures, list):
                return face
            for aperture in face_apertures:
                if aperture is None:
                    continue
                cen = Topology.CenterOfMass(aperture)
                try:
                    params = Face.ParametersAtVertex(face, cen)
                    u = params[0]
                    v = params[1]
                    w = 0.5
                except Exception:
                    u = 0.5
                    v = 0.5
                    w = 0.5
                context = Context.ByTopologyParameters(face, u, v, w)
                if context is not None:
                    _ = Aperture.ByTopologyContext(aperture, context)
            return face

        if model is None:
            return {"cells": [], "apertures": [], "shadingFaces": []}

        try:
            spaces = list(model.getSpaces())
        except Exception:
            spaces = []
        try:
            shadingSurfaces = list(model.getShadingSurfaces())
        except Exception:
            shadingSurfaces = []

        cells = []
        apertures = []
        shadingFaces = []

        for aShadingSurface in shadingSurfaces:
            shadingFace = surfaceToFace(aShadingSurface)
            if shadingFace is None:
                continue
            try:
                group = EnergyModel._OptionalGet(aShadingSurface.shadingSurfaceGroup(), None)
                if group is not None:
                    space = EnergyModel._OptionalGet(group.space(), None)
                    if space is not None:
                        shadingFace = transform_topology(shadingFace, space.transformation())
            except Exception:
                pass
            if shadingFace is not None:
                shadingFaces.append(shadingFace)

        for count, aSpace in enumerate(spaces):
            try:
                osTransformation = aSpace.transformation()
            except Exception:
                osTransformation = None
            try:
                surfaces = list(aSpace.surfaces())
            except Exception:
                surfaces = []

            spaceFaces = []
            for aSurface in surfaces:
                aFace = surfaceToFace(aSurface)
                if aFace is None:
                    continue
                if osTransformation is not None:
                    aFace = transform_topology(aFace, osTransformation)
                if aFace is None:
                    continue

                surface_apertures = []
                try:
                    subSurfaces = list(aSurface.subSurfaces())
                except Exception:
                    subSurfaces = []

                for aSubSurface in subSurfaces:
                    aperture = surfaceToFace(aSubSurface)
                    if aperture is None:
                        continue
                    if osTransformation is not None:
                        aperture = transform_topology(aperture, osTransformation)
                    if aperture is None:
                        continue
                    apertures.append(aperture)
                    surface_apertures.append(aperture)

                addApertures(aFace, surface_apertures)
                spaceFaces.append(aFace)

            spaceFaces = [x for x in spaceFaces if Topology.IsInstance(x, "Face")]
            if len(spaceFaces) == 0:
                continue
            spaceCell = Cell.ByFaces(spaceFaces, tolerance=tolerance)
            if not spaceCell:
                spaceCell = Shell.ByFaces(spaceFaces, tolerance=tolerance)
            if not Topology.IsInstance(spaceCell, "Cell"):
                spaceCell = Cluster.ByTopologies(spaceFaces)

            if Topology.IsInstance(spaceCell, "Topology"):
                keys = ["TOPOLOGIC_id", "TOPOLOGIC_name", "TOPOLOGIC_type", "TOPOLOGIC_color"]
                try:
                    spaceID = str(aSpace.handle()).replace("{", "").replace("}", "")
                except Exception:
                    spaceID = str(count)
                spaceName = EnergyModel._ObjectName(aSpace, "SPACE_" + str(count))
                spaceTypeName = "Unknown"
                color = [255, 255, 255]

                try:
                    spaceType = EnergyModel._OptionalGet(aSpace.spaceType(), None)
                    if spaceType is not None:
                        spaceTypeName = EnergyModel._ObjectName(spaceType, "Unknown")
                        color = EnergyModel._RenderingColorRGB(spaceType)
                except Exception:
                    pass

                values = [spaceID, spaceName, spaceTypeName, color]
                d = Dictionary.ByKeysValues(keys, values)
                spaceCell = Topology.SetDictionary(spaceCell, d)
                cells.append(spaceCell)

        return {"cells": cells, "apertures": apertures, "shadingFaces": shadingFaces}

    @staticmethod
    def Units(model, reportName, tableName, columnName):
        """
        Returns the units string found in the input OSM model, report name, table name, and column name.
        """
        sql = EnergyModel._SQLFile(model)
        if sql is None:
            return None
        query = "SELECT Units FROM tabulardatawithstrings WHERE ReportName = '" + str(reportName) + "' AND TableName = '" + str(tableName) + "' AND ColumnName = '" + str(columnName) + "'"
        try:
            units = sql.execAndReturnFirstString(query)
            if units.is_initialized():
                return units.get()
        except Exception:
            pass
        print("EnergyModel.Units - Error: Could not retrieve the units. Returning None.")
        return None
    
    @staticmethod
    def Version(check: bool = True, silent: bool = False):
        """
        Returns the OpenStudio SDK version number.
        """
        from topologicpy.Helper import Helper
        openstudio = EnergyModel._ImportOpenStudio("EnergyModel.Version", silent=silent)
        if openstudio is None:
            return None

        result = getattr(openstudio, "openStudioVersion", None)
        if callable(result):
            result = result()
        else:
            if not silent:
                print("EnergyModel.Version - Error: Could not retrieve the OpenStudio SDK version number. Returning None.")
            return None
        if check is True:
            result = Helper.CheckVersion("openstudio", result, silent=silent)
        return result
        
        
