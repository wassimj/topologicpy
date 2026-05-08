# Copyright (C) 2026
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

from __future__ import annotations

from topologicpy.Core import Core
import shutil
import math
from collections import OrderedDict
import os
from os.path import exists
from datetime import datetime, timezone
import warnings

try:
    from tqdm.auto import tqdm
except:
    print("EnergyModel - Installing required tqdm library.")
    try:
        os.system("pip install tqdm")
    except:
        os.system("pip install tqdm --user")
    try:
        from tqdm.auto import tqdm
        print("EnergyModel - tqdm library installed correctly.")
    except:
        warnings.warn("EnergyModel - Error: Could not import tqdm.")

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
    def ByOSMPath(path: str):
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
        try:
            import openstudio
            openstudio.Logger.instance().standardOutLogger().setLogLevel(openstudio.Fatal)
        except:
            print("EnergyModel.ByOSMPath - Information: Installing required openstudio library.")
            try:
                os.system("pip install openstudio")
            except:
                os.system("pip install openstudio --user")
            try:
                import openstudio
                openstudio.Logger.instance().standardOutLogger().setLogLevel(openstudio.Fatal)
                print("EnergyModel.ByOSMPath - Information: openstudio library installed correctly.")
            except:
                warnings.warn("EnergyModel.ByOSMPath - Error: Could not import openstudio.Please try to install openstudio manually. Returning None.")
                return None
        
        if not path:
            print("EnergyModel.ByOSMPath - Error: The input path is not valid. Returning None.")
            return None
        translator = openstudio.osversion.VersionTranslator()
        osmPath = openstudio.openstudioutilitiescore.toPath(path)
        osModel = translator.loadModel(osmPath)
        if osModel.isNull():
            print("EnergyModel.ByImportedOSM - Error: The openstudio model is null. Returning None.")
            return None
        else:
            osModel = osModel.get()
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

        try:
            import openstudio
            openstudio.Logger.instance().standardOutLogger().setLogLevel(openstudio.Fatal)
        except Exception:
            print("EnergyModel.ByTopology - Information: Installing required openstudio library.")
            try:
                os.system("pip install openstudio")
            except Exception:
                os.system("pip install openstudio --user")
            try:
                import openstudio
                openstudio.Logger.instance().standardOutLogger().setLogLevel(openstudio.Fatal)
                print("EnergyModel.ByTopology - Information: openstudio library installed correctly.")
            except Exception:
                warnings.warn("EnergyModel.ByTopology - Error: Could not import openstudio. Please install openstudio manually. Returning None.")
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
        if (not model_opt) or (not model_opt.is_initialized()):
            raise RuntimeError(f"EnergyModel.ByTopology - Could not load OSM template: {osModelPath}")
        osModel = model_opt.get()

        # Load EPW safely
        epwFile = os_path(weatherFilePath, openstudio)
        epw_opt = openstudio.openstudioutilitiesfiletypes.EpwFile.load(epwFile)
        if epw_opt.is_initialized():
            osEPWFile = epw_opt.get()
            openstudio.model.WeatherFile.setWeatherFile(osModel, osEPWFile)
        else:
            raise RuntimeError(f"EnergyModel.ByTopology - Could not load EPW weather file: {weatherFilePath}")

        # Load DDY safely
        ddyFile = os_path(designDayFilePath, openstudio)
        ddy_opt = openstudio.openstudioenergyplus.loadAndTranslateIdf(ddyFile)
        if ddy_opt.is_initialized():
            ddyModel = ddy_opt.get()
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
    def ExportToGBXML(model, path, overwrite=False):
        """
            Exports the input OSM model to a GBXML file.

        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.
        path : str
            The path for saving the file.
        overwrite : bool, optional
            If set to True any file with the same name is over-written. Default is False.

        Returns
        -------
        bool
            True if the file is written successfully. False otherwise.

        """
        from os.path import exists
        try:
            import openstudio
            openstudio.Logger.instance().standardOutLogger().setLogLevel(openstudio.Fatal)
        except:
            print("EnergyModel.ExportToGBXML - Information: Installing required openstudio library.")
            try:
                os.system("pip install openstudio")
            except:
                os.system("pip install openstudio --user")
            try:
                import openstudio
                openstudio.Logger.instance().standardOutLogger().setLogLevel(openstudio.Fatal)
                print("EnergyModel.ExportToGBXML - Information: openstudio library installed correctly.")
            except:
                warnings.warn("EnergyModel.ExportToGBXML - Error: Could not import openstudio.Please try to install openstudio manually. Returning None.")
                return None
        
        # Make sure the file extension is .xml
        ext = path[len(path)-4:len(path)]
        if ext.lower() != ".xml":
            path = path+".xml"
        
        if not overwrite and exists(path):
            print("EnergyModel.ExportToGBXML - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        # DEBUGGING
        #return openstudio.gbxml.GbXMLForwardTranslator().modelToGbXML(model, openstudio.openstudioutilitiescore.toPath(path))
        return openstudio.gbxml.GbXMLForwardTranslator().modelToGbXML(model, path)

    
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
            If set to True any file with the same name is over-written. Default is False.

        Returns
        -------
        bool
            True if the file is written successfully. False otherwise.

        """
        from os.path import exists
        try:
            import openstudio
            openstudio.Logger.instance().standardOutLogger().setLogLevel(openstudio.Fatal)
        except:
            print("EnergyModel.ExportToOSM - Information: Installing required openstudio library.")
            try:
                os.system("pip install openstudio")
            except:
                os.system("pip install openstudio --user")
            try:
                import openstudio
                openstudio.Logger.instance().standardOutLogger().setLogLevel(openstudio.Fatal)
                print("EnergyModel.ExportToOSM - Information: openstudio library installed correctly.")
            except:
                warnings.warn("EnergyModel.ExportToOSM - Error: Could not import openstudio.Please try to install openstudio manually. Returning None.")
                return None
        
        # Make sure the file extension is .osm
        ext = path[len(path)-4:len(path)]
        if ext.lower() != ".osm":
            path = path+".osm"
        
        if not overwrite and exists(path):
            print("EnergyModel.ExportToOSM - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        osCondition = False
        # DEBUGGING
        #osPath = openstudio.openstudioutilitiescore.toPath(path)
        #osCondition = model.save(osPath, overwrite)
        osCondition = model.save(path, overwrite)
        return osCondition
    
    @staticmethod
    def GBXMLString(model):
        """
            Returns the GBXML string of the input OSM model.

        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.

        Returns
        -------
        str
            The gbxml string.

        """
        try:
            import openstudio
            openstudio.Logger.instance().standardOutLogger().setLogLevel(openstudio.Fatal)
        except:
            print("EnergyModel.GBXMLString - Information: Installing required openstudio library.")
            try:
                os.system("pip install openstudio")
            except:
                os.system("pip install openstudio --user")
            try:
                import openstudio
                openstudio.Logger.instance().standardOutLogger().setLogLevel(openstudio.Fatal)
                print("EnergyModel.GBXMLString - Information: openstudio library installed correctly.")
            except:
                warnings.warn("EnergyModel.GBXMLString - Error: Could not import openstudio.Please try to install openstudio manually. Returning None.")
                return None
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
            The input report name. Default is "HVACSizingSummary".
        reportForString : str, optional
            The input report for string. Default is "Entire Facility".
        tableName : str , optional
            The input table name. Default is "Zone Sensible Cooling".
        columnName : str , optional
            The input column name. Default is "Calculated Design Load".
        rowNames : list , optional
            The input list of row names. Default is [].
        units : str , optional
            The input units. Default is "W".

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
    def Run(model, weatherFilePath: str = None, osBinaryPath : str = None, outputFolder : str = None, removeFiles : bool = False):
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
        removeFiles : bool , optional
            If set to True, the working files are removed at the end of the process. Default is False.

        Returns
        -------
        model : openstudio.openstudiomodelcore.Model
            The simulated OSM model.

        """
        import os
        import time
        try:
            import openstudio
            openstudio.Logger.instance().standardOutLogger().setLogLevel(openstudio.Fatal)
        except:
            print("EnergyModel.Run - Information: Installing required openstudio library.")
            try:
                os.system("pip install openstudio")
            except:
                os.system("pip install openstudio --user")
            try:
                import openstudio
                openstudio.Logger.instance().standardOutLogger().setLogLevel(openstudio.Fatal)
                print("EnergyModel.Run - Information: openstudio library installed correctly.")
            except:
                warnings.warn("EnergyModel.Run - Error: Could not import openstudio.Please try to install openstudio manually. Returning None.")
                return None
        def deleteOldFiles(path):
            onemonth = (time.time()) - 30 * 86400
            try:
                for filename in os.listdir(path):
                    if os.path.getmtime(os.path.join(path, filename)) < onemonth:
                        if os.path.isfile(os.path.join(path, filename)):
                            os.remove(os.path.join(path, filename))
                        elif os.path.isdir(os.path.join(path, filename)):
                            shutil.rmtree((os.path.join(path, filename)))
            except:
                pass
        if not weatherFilePath:
            weatherFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "EnergyModel", "GBR_London.Gatwick.037760_IWEC.epw")
        if removeFiles:
            deleteOldFiles(outputFolder)
        pbar = tqdm(desc='Running Simulation', total=100, leave=False)
        utcnow = datetime.now(timezone.utc)
        timestamp = utcnow.strftime("UTC-%Y-%m-%d-%H-%M-%S")
        if not outputFolder:
            home = os.path.expanduser('~')
            outputFolder = os.path.join(home, "EnergyModels", timestamp)
        else:
            outputFolder = os.path.join(outputFolder, timestamp)
        os.mkdir(outputFolder)
        pbar.update(10)
        osmPath = os.path.join(outputFolder, model.getBuilding().name().get() + ".osm")
        # DEBUGGING
        #model.save(openstudio.openstudioutilitiescore.toPath(osmPath), True)
        model.save(osmPath, True)
        oswPath = os.path.join(outputFolder, model.getBuilding().name().get() + ".osw")
        pbar.update(20)
        workflow = model.workflowJSON()
        # DEBUGGING
        #workflow.setSeedFile(openstudio.openstudioutilitiescore.toPath(osmPath))
        workflow.setSeedFile(osmPath)
        pbar.update(30)
        # DEBUGGING
        #workflow.setWeatherFile(openstudio.openstudioutilitiescore.toPath(weatherFilePath))
        workflow.setWeatherFile(weatherFilePath)
        pbar.update(40)
        # DEBUGGING
        #workflow.saveAs(openstudio.openstudioutilitiescore.toPath(oswPath))
        workflow.saveAs(oswPath)
        pbar.update(50)
        cmd = osBinaryPath+" run -w " + "\"" + oswPath + "\""
        pbar.update(60)
        os.system(cmd)
        sqlPath = os.path.join(os.path.join(outputFolder,"run"), "eplusout.sql")
        pbar.update(100)
        # DEBUGGING
        #osSqlFile = openstudio.SqlFile(openstudio.openstudioutilitiescore.toPath(sqlPath))
        osSqlFile = openstudio.SqlFile(sqlPath)
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
    def Topologies(model, tolerance=0.0001):
        """
        Parameters
        ----------
        model : openstudio.openstudiomodelcore.Model
            The input OSM model.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        dict
            The dictionary of topologies found in the input OSM model. The keys of the dictionary are:
            - "cells"
            - "apertures"
            - "shadingFaces"

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

        def surfaceToFace(surface):
            surfaceEdges = []
            surfaceVertices = surface.vertices()
            for i in range(len(surfaceVertices)-1):
                sv = Vertex.ByCoordinates(Vertex.X(surfaceVertices[i]), Vertex.Y(surfaceVertices[i]), Vertex.Z(surfaceVertices[i]))
                ev = Vertex.ByCoordinates(Vertex.X(surfaceVertices[i+1]), Vertex.Y(surfaceVertices[i+1]), Vertex.Z(surfaceVertices[i+1]))
                edge = Edge.ByStartVertexEndVertex(sv, ev, tolerance=tolerance, silent=False)
                if not edge:
                    continue
                surfaceEdges.append(edge)
            sv = Vertex.ByCoordinates(Vertex.X(surfaceVertices[len(surfaceVertices)-1]), Vertex.Y(surfaceVertices[len(surfaceVertices)-1]), Vertex.Z(surfaceVertices[len(surfaceVertices)-1]))
            ev = Vertex.ByCoordinates(Vertex.X(surfaceVertices[0]), Vertex.Y(surfaceVertices[0]), Vertex.Z(surfaceVertices[0]))
            edge = Edge.ByStartVertexEndVertex(sv, ev, tolerance=tolerance, silent=False)
            surfaceEdges.append(edge)
            surfaceWire = Wire.ByEdges(surfaceEdges, tolerance=tolerance)
            internalBoundaries = []
            surfaceFace = Face.ByWires(surfaceWire, internalBoundaries, tolerance=tolerance)
            return surfaceFace
        
        def addApertures(face, apertures):
            usedFaces = []
            for aperture in apertures:
                cen = Topology.CenterOfMass(aperture)
                try:
                    params = Face.ParametersAtVertex(face, cen)
                    u = params[0]
                    v = params[1]
                    w = 0.5
                except:
                    u = 0.5
                    v = 0.5
                    w = 0.5
                context = Context.ByTopologyParameters(face, u, v, w)
                _ = Aperture.ByTopologyContext(aperture, context)
            return face
        spaces = list(model.getSpaces())
        
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
                    shadingFace = Core.TopologyUtility.Transform(shadingFace, osTranslation.x(), osTranslation.y(), osTranslation.z(), rotation11, rotation12, rotation13, rotation21, rotation22, rotation23, rotation31, rotation32, rotation33)
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
                aFace = Core.TopologyUtility.Transform(aFace, osTranslation.x(), osTranslation.y(), osTranslation.z(), rotation11, rotation12, rotation13, rotation21, rotation22, rotation23, rotation31, rotation32, rotation33)
                subSurfaces = aSurface.subSurfaces()
                for aSubSurface in subSurfaces:
                    aperture = surfaceToFace(aSubSurface)
                    aperture = Core.TopologyUtility.Transform(aperture, osTranslation.x(), osTranslation.y(), osTranslation.z(), rotation11, rotation12, rotation13, rotation21, rotation22, rotation23, rotation31, rotation32, rotation33)
                    apertures.append(aperture)
                addApertures(aFace, apertures)
                spaceFaces.append(aFace)
            spaceFaces = [x for x in spaceFaces if Topology.IsInstance(x, "Face")]
            spaceCell = Cell.ByFaces(spaceFaces, tolerance=tolerance)
            if not spaceCell:
                spaceCell = Shell.ByFaces(spaceFaces, tolerance=tolerance)
            if not Topology.IsInstance(spaceCell, "Cell"):
                spaceCell = Cluster.ByTopologies(spaceFaces)
            if Topology.IsInstance(spaceCell, "Topology"): #debugging
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
            print("EnergyModel.Units - Error: Could not retrieve the units. Returning None.")
            return None
        return units
    
    @staticmethod
    def Version(check: bool = True, silent: bool = False):
        """
        Returns the OpenStudio SDK version number.

        Parameters
        ----------
        check : bool , optional
            if set to True, the version number is checked with the latest version on PyPi. Default is True.
        
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        str
            The OpenStudio SDK version number.

        """
        from topologicpy.Helper import Helper
        try:
            import openstudio
            openstudio.Logger.instance().standardOutLogger().setLogLevel(openstudio.Fatal)
        except:
            if not silent:
                print("EnergyModel.Version - Information: Installing required openstudio library.")
            try:
                os.system("pip install openstudio")
            except:
                os.system("pip install openstudio --user")
            try:
                import openstudio
                openstudio.Logger.instance().standardOutLogger().setLogLevel(openstudio.Fatal)
                if not silent:
                    print("EnergyModel.Version - Information: openstudio library installed correctly.")
            except:
                if not silent:
                    print("EnergyModel.Version - Error: Could not import openstudio.Please try to install openstudio manually. Returning None.")
                return None
        import requests
        from packaging import version

        result = getattr(openstudio, "openStudioVersion", None)
        if callable(result):
            result = result()
        else:
            if not silent:
                print("EnergyModel.Version - Error: Could not retrieve the openstudio SDK version number. Returning None.")
            return None
        if check == True:
            result = Helper.CheckVersion("openstudio", result, silent=silent)
        return result
        
        
