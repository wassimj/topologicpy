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
import topologic_core as topologic
try:
    import ephem
except:
    print("Sun - Installing required ephem library.")
    try:
        os.system("pip install ephem")
    except:
        os.system("pip install ephem --user")
    try:
        import ephem
        print("Sun - ephem library installed successfully.")
    except:
        warnings.warn("Sun - Error: Could not import ephem.")

class Sun():
    @staticmethod
    def WinterSolstice(latitude, year=None):
        """
        Returns the winter solstice date for the input latitude and year. See https://en.wikipedia.org/wiki/Winter_solstice.

        Parameters
        ----------
        latitude : float
            The input latitude.
        year : integer , optional
            The input year. The default is the current year.

        Returns
        -------
        datetime
            The datetime of the winter solstice

        """
        import os
        import warnings
        try:
            import ephem
        except:
            print("Sun.WinterSolstice - Information: Installing required ephem library.")
            try:
                os.system("pip install ephem")
            except:
                os.system("pip install ephem --user")
            try:
                import ephem
                print("Sun.WinterSolstice - Infromation: ephem library installed correctly.")
            except:
                warnings.warn("Sun.WinterSolstice - Error: Could not import ephem. Please try to install ephem manually. Returning None.")
                return None
        from datetime import datetime
        if year == None:
            year = datetime.now().year
        if latitude >= 0:
            solstice = ephem.next_solstice(ephem.Date(f"{year}/12/1"))
        else:
            solstice = ephem.next_solstice(ephem.Date(f"{year}/6/1"))
        return solstice.datetime()
    
    @staticmethod
    def SummerSolstice(latitude, year):
        """
        Returns the winter solstice date for the input latitude and year. See https://en.wikipedia.org/wiki/Summer_solstice.

        Parameters
        ----------
        latitude : float
            The input latitude.
        year : integer , optional
            The input year. The default is the current year.

        Returns
        -------
        datetime
            The datetime of the summer solstice

        """
        import os
        import warnings
        try:
            import ephem
        except:
            print("Sun.SummerSolstice - Information: Installing required ephem library.")
            try:
                os.system("pip install ephem")
            except:
                os.system("pip install ephem --user")
            try:
                import ephem
                print("Sun.SummerSolstice - Infromation: ephem library installed correctly.")
            except:
                warnings.warn("Sun.SummerSolstice - Error: Could not import ephem. Please try to install ephem manually. Returning None.")
                return None
        
        if latitude >= 0:
            solstice = ephem.next_solstice(ephem.Date(f"{year}/6/1"))
        else:
            solstice = ephem.next_solstice(ephem.Date(f"{year}/12/1"))
        return solstice.datetime()
    
    @staticmethod
    def SpringEquinox(latitude, year):
        """
        Returns the spring (vernal) equinox date for the input latitude and year. See https://en.wikipedia.org/wiki/March_equinox.

        Parameters
        ----------
        latitude : float
            The input latitude.
        year : integer , optional
            The input year. The default is the current year.

        Returns
        -------
        datetime
            The datetime of the summer solstice
        """
        import os
        import warnings
        try:
            import ephem
        except:
            print("Sun.SpringEquinox - Information: Installing required ephem library.")
            try:
                os.system("pip install ephem")
            except:
                os.system("pip install ephem --user")
            try:
                import ephem
                print("Sun.SpringEquinox - Infromation: ephem library installed correctly.")
            except:
                warnings.warn("Sun.SpringEquinox - Error: Could not import ephem. Please try to install ephem manually. Returning None.")
                return None
        
        if latitude >= 0:
            equinox = ephem.next_equinox(ephem.Date(f"{year}/3/1"))
        else:
            equinox = ephem.next_equinox(ephem.Date(f"{year}/9/1"))
        return equinox.datetime()
    
    @staticmethod
    def AutumnEquinox(latitude, year):
        """
        Returns the autumnal equinox date for the input latitude and year. See https://en.wikipedia.org/wiki/September_equinox.

        Parameters
        ----------
        latitude : float
            The input latitude. See https://en.wikipedia.org/wiki/Latitude.
        year : integer , optional
            The input year. The default is the current year.

        Returns
        -------
        datetime
            The datetime of the summer solstice
        """
        import os
        import warnings
        try:
            import ephem
        except:
            print("Sun.AutumnEquinox - Information: Installing required ephem library.")
            try:
                os.system("pip install ephem")
            except:
                os.system("pip install ephem --user")
            try:
                import ephem
                print("Sun.AutumnEquinox - Infromation: ephem library installed correctly.")
            except:
                warnings.warn("Sun.AutumnEquinox - Error: Could not import ephem. Please try to install ephem manually. Returning None.")
                return None
        
        if latitude >= 0:
            equinox = ephem.next_equinox(ephem.Date(f"{year}/9/1"))
        else:
            equinox = ephem.next_equinox(ephem.Date(f"{year}/3/1"))
        return equinox.datetime()
    
    @staticmethod
    def Azimuth(latitude, longitude, date):
        """
        Returns the Azimuth angle. See https://en.wikipedia.org/wiki/Azimuth.

        Parameters
        ----------
        latitude : float
            The input latitude. See https://en.wikipedia.org/wiki/Latitude.
        longitude : float
            The input longitude. See https://en.wikipedia.org/wiki/Longitude.
        date : datetime
            The input datetime.

        Returns
        -------
        float
            The azimuth angle.
        """
        import os
        import warnings
        import math
        try:
            import ephem
        except:
            print("Sun.Azimuth - Information: Installing required ephem library.")
            try:
                os.system("pip install ephem")
            except:
                os.system("pip install ephem --user")
            try:
                import ephem
                print("Sun.Azimuth - Infromation: ephem library installed correctly.")
            except:
                warnings.warn("Sun.Azimuth - Error: Could not import ephem. Please try to install ephem manually. Returning None.")
                return None
        
        observer = ephem.Observer()
        observer.date = date
        observer.lat = str(latitude)
        observer.lon = str(longitude)
        sun = ephem.Sun(observer)
        sun.compute(observer)
        azimuth = math.degrees(sun.az)        
        return azimuth

    @staticmethod
    def Altitude(latitude, longitude, date):
        """
        Returns the Altitude angle. See https://en.wikipedia.org/wiki/Altitude.

        Parameters
        ----------
        latitude : float
            The input latitude. See https://en.wikipedia.org/wiki/Latitude.
        longitude : float
            The input longitude. See https://en.wikipedia.org/wiki/Longitude.
        date : datetime
            The input datetime.

        Returns
        -------
        float
            The altitude angle.
        """

        import os
        import warnings
        import math
        try:
            import ephem
        except:
            print("Sun.Altitude - Information: Installing required ephem library.")
            try:
                os.system("pip install ephem")
            except:
                os.system("pip install ephem --user")
            try:
                import ephem
                print("Sun.Altitude - Infromation: ephem library installed correctly.")
            except:
                warnings.warn("Sun.Altitude - Error: Could not import ephem. Please try to install ephem manually. Returning None.")
                return None
        
        observer = ephem.Observer()
        observer.date = date
        observer.lat = str(latitude)
        observer.lon = str(longitude)
        sun = ephem.Sun(observer)
        sun.compute(observer)
        altitude = math.degrees(sun.alt)
        return altitude
        
    @staticmethod
    def Sunrise(latitude, longitude, date):
        """
        Returns the Sunrise datetime.

        Parameters
        ----------
        latitude : float
            The input latitude. See https://en.wikipedia.org/wiki/Latitude.
        longitude : float
            The input longitude. See https://en.wikipedia.org/wiki/Longitude.
        date : datetime
            The input datetime.

        Returns
        -------
        datetime
            The Sunrise datetime.
        """

        import os
        import warnings
        try:
            import ephem
        except:
            print("Sun.Sunrise - Information: Installing required ephem library.")
            try:
                os.system("pip install ephem")
            except:
                os.system("pip install ephem --user")
            try:
                import ephem
                print("Sun.Sunrise - Infromation: ephem library installed correctly.")
            except:
                warnings.warn("Sun.Sunrise - Error: Could not import ephem. Please try to install ephem manually. Returning None.")
                return None
        
        date = date.replace(hour=12, minute=0, second=0, microsecond=0)
        observer = ephem.Observer()
        observer.lat = str(latitude)
        observer.lon = str(longitude)
        observer.date = date
        sunrise = observer.previous_rising(ephem.Sun()).datetime()
        return sunrise
    
    @staticmethod
    def Sunset(latitude, longitude, date):
        """
        Returns the Sunset datetime.

        Parameters
        ----------
        latitude : float
            The input latitude. See https://en.wikipedia.org/wiki/Latitude.
        longitude : float
            The input longitude. See https://en.wikipedia.org/wiki/Longitude.
        date : datetime
            The input datetime.

        Returns
        -------
        datetime
            The Sunset datetime.
        """

        import os
        import warnings
        try:
            import ephem
        except:
            print("Sun.Sunset - Information: Installing required ephem library.")
            try:
                os.system("pip install ephem")
            except:
                os.system("pip install ephem --user")
            try:
                import ephem
                print("Sun.Sunset - Infromation: ephem library installed correctly.")
            except:
                warnings.warn("Sun.Sunset - Error: Could not import ephem. Please try to install ephem manually. Returning None.")
                return None
        
        date = date.replace(hour=12, minute=0, second=0, microsecond=0)
        observer = ephem.Observer()
        observer.lat = str(latitude)
        observer.lon = str(longitude)
        observer.date = date
        sunset = observer.next_setting(ephem.Sun()).datetime()
        return sunset

    @staticmethod
    def Vector(latitude, longitude, date, north=0):
        """
        Returns the Sun as a vector.

        Parameters
        ----------
        latitude : float
            The input latitude. See https://en.wikipedia.org/wiki/Latitude.
        longitude : float
            The input longitude. See https://en.wikipedia.org/wiki/Longitude.
        date : datetime
            The input datetime.
        north : float, optional
            The desired compass angle of the north direction. The default is 0 which points in the positive Y-axis direction.

        Returns
        -------
        list
            The sun vector pointing from the location of the sun towards the origin.
        """
        
        from topologicpy.Vector import Vector
        azimuth = Sun.Azimuth(latitude=latitude, longitude=longitude, date=date)
        altitude = Sun.Altitude(latitude=latitude, longitude=longitude, date=date)
        return Vector.ByAzimuthAltitude(azimuth=azimuth, altitude=altitude, north=north, reverse=True)

    @staticmethod
    def Position(latitude, longitude, date, origin=None, radius=0.5, north=0, mantissa=6):
        """
        Returns the Sun as a position ([X,Y,Z]).

        Parameters
        ----------
        latitude : float
            The input latitude. See https://en.wikipedia.org/wiki/Latitude.
        longitude : float
            The input longitude. See https://en.wikipedia.org/wiki/Longitude.
        date : datetime
            The input datetime.
        origin : topologic.Vertex , optional
            The desired origin of the world. If set to None, the origin will be set to (0,0,0). The default is None.
        radius : float , optional
            The desired radius of the sun orbit. The default is 0.5.
        north : float, optional
            The desired compass angle of the north direction. The default is 0 which points in the positive Y-axis direction.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        
        
        Returns
        -------
        topologic.Vertex
            The sun represented as a vertex.
        """
        from topologicpy.Vertex import Vertex
        sun_v = Sun.Vertex(latitude=latitude, longitude=longitude, date=date, origin=origin, radius=radius, north=north)
        return Vertex.Coordinates(sun_v, mantissa=mantissa)
    
    @staticmethod
    def Vertex(latitude, longitude, date, origin=None, radius=0.5, north=0):
        """
        Returns the Sun as a vertex.

        Parameters
        ----------
        latitude : float
            The input latitude. See https://en.wikipedia.org/wiki/Latitude.
        longitude : float
            The input longitude. See https://en.wikipedia.org/wiki/Longitude.
        date : datetime
            The input datetime.
        origin : topologic.Vertex , optional
            The desired origin of the world. If set to None, the origin will be set to (0,0,0). The default is None.
        radius : float , optional
            The desired radius of the sun orbit. The default is 0.5.
        north : float, optional
            The desired compass angle of the north direction. The default is 0 which points in the positive Y-axis direction.

        Returns
        -------
        topologic.Vertex
            The sun represented as a vertex.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector

        if origin == None:
            origin = Vertex.Origin()
        vector = Vector.Reverse(Sun.Vector(latitude=latitude, longitude=longitude, date=date, north=north))
        sun_v = Topology.TranslateByDirectionDistance(origin, direction=vector, distance=radius)
        return sun_v

    @staticmethod
    def Edge(latitude, longitude, date, origin=None, radius=0.5, north=0):
        """
        Returns the Sun as a vector.

        Parameters
        ----------
        latitude : float
            The input latitude. See https://en.wikipedia.org/wiki/Latitude.
        longitude : float
            The input longitude. See https://en.wikipedia.org/wiki/Longitude.
        date : datetime
            The input datetime.
        origin : topologic.Vertex , optional
            The desired origin of the world. If set to None, the origin will be set to (0,0,0). The default is None.
        radius : float , optional
            The desired radius of the sun orbit. The default is 0.5.
        north : float, optional
            The desired compass angle of the north direction. The default is 0 which points in the positive Y-axis direction.

        Returns
        -------
        topologic.Edge
            The sun represented as an edge pointing from the location of the sun towards the origin.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector

        if origin == None:
            origin = Vertex.Origin()
        vector = Vector.Reverse(Sun.Vector(latitude=latitude, longitude=longitude, date=date, north=north))
        sun_v = Topology.TranslateByDirectionDistance(origin, direction=vector, distance=radius)
        edge = Edge.ByVertices(sun_v, origin)
        return edge

    @staticmethod
    def VerticesByDate(latitude, longitude, date, startTime=None, endTime=None, interval=60, origin=None, radius=0.5, north=0):
        """
        Returns the Sun locations as vertices based on the input parameters.

        Parameters
        ----------
        latitude : float
            The input latitude. See https://en.wikipedia.org/wiki/Latitude.
        longitude : float
            The input longitude. See https://en.wikipedia.org/wiki/Longitude.
        date : datetime
            The input datetime.
        startTime : datetime , optional
            The desired start time to compute the sun location. If set to None, Sun.Sunrise is used. The default is None.
        endTime : datetime , optional
            The desired end time to compute the sun location. If set to None, Sun.Sunset is used. The default is None.
        interval : int , optional
            The interval in minutes to compute the sun location. The default is 60.
        origin : topologic.Vertex , optional
            The desired origin of the world. If set to None, the origin will be set to (0,0,0). The default is None.
        radius : float , optional
            The desired radius of the sun orbit. The default is 0.5.
        north : float, optional
            The desired compass angle of the north direction. The default is 0 which points in the positive Y-axis direction.

        Returns
        -------
        list
            The sun locations represented as a list of vertices.
        """
        from datetime import timedelta
        if startTime == None:
            startTime = Sun.Sunrise(latitude=latitude, longitude=longitude, date=date)
        if endTime == None:
            endTime = Sun.Sunset(latitude=latitude, longitude=longitude, date=date)
        vertices = []
        current_time = startTime
        while current_time <= endTime:
            v = Sun.Vertex(latitude=latitude, longitude=longitude, date=current_time, origin=origin, radius=radius, north=north)
            vertices.append(v)
            current_time += timedelta(minutes=interval)
        return vertices

    @staticmethod
    def PathByDate(latitude, longitude, date, startTime=None, endTime=None, interval=60, origin=None, radius=0.5, sides=None, north=0):
        """
        Returns the sun path based on the input parameters.

        Parameters
        ----------
        latitude : float
            The input latitude. See https://en.wikipedia.org/wiki/Latitude.
        longitude : float
            The input longitude. See https://en.wikipedia.org/wiki/Longitude.
        date : datetime
            The input datetime.
        startTime : datetime , optional
            The desired start time to compute the sun location. If set to None, Sun.Sunrise is used. The default is None.
        endTime : datetime , optional
            The desired end time to compute the sun location. If set to None, Sun.Sunset is used. The default is None.
        interval : int , optional
            The interval in minutes to compute the sun location. The default is 60.
        origin : topologic.Vertex , optional
            The desired origin of the world. If set to None, the origin will be set to (0,0,0). The default is None.
        radius : float , optional
            The desired radius of the sun orbit. The default is 0.5.
        sides : int , optional
            If set to None, the path is divided based on the interval. Otherwise, it is equally divided into the number of sides.
            The default is None.
        north : float, optional
            The desired compass angle of the north direction. The default is 0 which points in the positive Y-axis direction.

        Returns
        -------
        topologic.Wire
            The sun path represented as a wire.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        if origin == None:
            origin = Vertex.Origin()
        if startTime == None:
            startTime = Sun.Sunrise(latitude=latitude, longitude=longitude, date=date)
        if endTime == None:
            endTime = Sun.Sunset(latitude=latitude, longitude=longitude, date=date)
        vertices = Sun.VerticesByDate(latitude=latitude, longitude=longitude, date=date,
                                      startTime=startTime, endTime=endTime, interval=interval,
                                      origin=origin, radius=radius, north=north)
        if len(vertices) < 2:
            return None
        wire = Wire.ByVertices(vertices, close=False)
        if not sides == None:
            vertices = []
            for i in range(sides):
                u = float(i)/float(sides)
                v = Wire.VertexByParameter(wire, u)
                vertices.append(v)
            wire = Wire.ByVertices(vertices, close=False)
        d = Dictionary.ByKeysValues(["latitude", "longitude", "date", "startTime", "endTime", "interval", "type"],
                                    [latitude, longitude, str(date), str(startTime), str(endTime), interval, "date"])
        wire = Topology.SetDictionary(wire, d)
        return wire

    @staticmethod
    def VerticesByHour(latitude, longitude, hour, startDay=1, endDay=365, interval=5, origin=None, radius=0.5, north=0):
        """
        Returns the sun locations based on the input parameters.

        Parameters
        ----------
        latitude : float
            The input latitude. See https://en.wikipedia.org/wiki/Latitude.
        longitude : float
            The input longitude. See https://en.wikipedia.org/wiki/Longitude.
        hour : datetime
            The input hour.
        startDay : integer , optional
            The desired start day to compute the sun location. The default is 1.
        endDay : integer , optional
            The desired end day to compute the sun location. The default is 365.
        interval : int , optional
            The interval in days to compute the sun location. The default is 5.
        origin : topologic.Vertex , optional
            The desired origin of the world. If set to None, the origin will be set to (0,0,0). The default is None.
        radius : float , optional
            The desired radius of the sun orbit. The default is 0.5.
        north : float, optional
            The desired compass angle of the north direction. The default is 0 which points in the positive Y-axis direction.

        Returns
        -------
        list
            The sun locations represented as a list of vertices.
        """
        from datetime import datetime
        from datetime import timedelta
        def day_of_year_to_datetime(year, day_of_year):
            # Construct a datetime object for the first day of the year
            base_date = datetime(year, 1, 1)
            # Add the number of days to get to the specified day of the year
            target_date = base_date + timedelta(days=day_of_year - 1)
            return target_date
        
        now = datetime.now()
        # Get the year component
        year = now.year
        vertices = []
        for day_of_year in range(startDay, endDay, interval):
            date = day_of_year_to_datetime(year, day_of_year)
            date += timedelta(hours=hour)
            v = Sun.Vertex(latitude=latitude, longitude=longitude, date=date, origin=origin, radius=radius, north=north)
            vertices.append(v)
        return vertices

    @staticmethod
    def PathByHour(latitude, longitude, hour, startDay=1, endDay=365, interval=5, origin=None, radius=0.5, sides=None, north=0):
        """
        Returns the sun locations based on the input parameters.

        Parameters
        ----------
        latitude : float
            The input latitude. See https://en.wikipedia.org/wiki/Latitude.
        longitude : float
            The input longitude. See https://en.wikipedia.org/wiki/Longitude.
        hour : datetime
            The input hour.
        startDay : integer , optional
            The desired start day of the year to compute the sun location. The default is 1.
        endDay : integer , optional
            The desired end day of the year to compute the sun location. The default is 365.
        interval : int , optional
            The interval in days to compute the sun location. The default is 5.
        origin : topologic.Vertex , optional
            The desired origin of the world. If set to None, the origin will be set to (0,0,0). The default is None.
        radius : float , optional
            The desired radius of the sun orbit. The default is 0.5.
        sides : int , optional
            If set to None, the path is divided based on the interval. Otherwise, it is equally divided into the number of sides.
            The default is None.
        north : float, optional
            The desired compass angle of the north direction. The default is 0 which points in the positive Y-axis direction.

        Returns
        -------
        topologic.Wire
            The sun path represented as a topologic wire.
        """

        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        
        vertices = Sun.VerticesByHour(latitude=latitude, longitude=longitude, hour=hour,
                                      startDay=startDay, endDay=endDay, interval=interval,
                                      origin=origin, radius=radius, north=north)
        if len(vertices) < 2:
            return None
        wire = Wire.ByVertices(vertices, close=False)
        if not sides == None:
            vertices = []
            for i in range(sides):
                u = float(i)/float(sides)
                v = Wire.VertexByParameter(wire, u)
                vertices.append(v)
            wire = Wire.ByVertices(vertices, close=True)
        d = Dictionary.ByKeysValues(["latitude", "longitude", "hour", "startDay", "endDay", "interval", "type"],
                                    [latitude, longitude, hour, startDay, endDay, interval, "hour"])
        wire = Topology.SetDictionary(wire, d)
        return wire

    @staticmethod
    def Diagram(latitude, longitude, minuteInterval=30, dayInterval=15,
                origin=None, radius=0.5, uSides=180, vSides=180, north=0,
                compass = False, shell=False):
        """
        Returns the sun diagram based on the input parameters. See https://hyperfinearchitecture.com/how-to-read-sun-path-diagrams/.

        Parameters
        ----------
        latitude : float
            The input latitude. See https://en.wikipedia.org/wiki/Latitude.
        longitude : float
            The input longitude. See https://en.wikipedia.org/wiki/Longitude.
        minuteInterval : int , optional
            The interval in minutes to compute the sun location for the date path. The default is 30.
        dayInterval : int , optional
            The interval in days for the hourly path to compute the sun location. The default is 15.
        origin : topologic.Vertex , optional
            The desired origin of the world. If set to None, the origin will be set to (0,0,0). The default is None.
        radius : float , optional
            The desired radius of the sun orbit. The default is 0.5.
        uSides : int , optional
            The number of sides to divide the diagram horizontally (along the azimuth). The default is 180.
        vSides : int , optional
            The number of sides to divide the diagram paths vertically (along the altitude). The default is 180.
        north : float, optional
            The desired compass angle of the north direction. The default is 0 which points in the positive Y-axis direction.
        compass : bool , optional
            If set to True, a compass (shell) is included. Othwerwise, it is is not.
        shell : bool , optional
            If set to True, the total surface (shell) of the sun paths is incldued. Otherwise, it is not.

        Returns
        -------
        dict
            A dictionary of the sun diagram shapes. The keys in this dictionary are:
            'date_paths': These are the sun paths (wire) for the winter solstice, equinox, and summer solstice
            'hourly_paths': These are the figure-8 (wire) for the sun location on the same hour on each selected day of the year.
            'shell': This is the total surface (shell) of the sun paths. This is included only if the shell option is set to True.
            'compass': This is the compass (shell) on the ground. It is made of 36 sides and 10 rings. This is included only if the compass option is set to True.
            'center' : This is a cross-shape (wire) at the center of the diagram. This is included only if the compass option is set to True.
            'ground' : This is a circle (face) on the ground. It is made of 36 sides. This is included only if the compass option is set to False.
        """

        from datetime import datetime
        from datetime import timedelta
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if origin == None:
            origin = Vertex.Origin()

        cutter = Cell.Prism(origin=origin, width=radius*4, length=radius*4, height=radius*2)
        cutter = Topology.Rotate(cutter, origin=origin, angle=-north)
        cutter = Topology.Translate(cutter, 0,0,-radius)
        now = datetime.now()
        year = now.year
        diagram = {}
        winter_solstice = Sun.WinterSolstice(latitude=latitude, year=year)
        summer_solstice = Sun.SummerSolstice(latitude=latitude, year=year)
        equinox = Sun.AutumnEquinox(latitude=latitude, year=year)
        dates = [winter_solstice, equinox, summer_solstice]
        date_paths = []
        for date in dates:
            startTime = Sun.Sunrise(latitude=latitude, longitude=longitude, date=date) - timedelta(hours=2)
            endTime = Sun.Sunset(latitude=latitude, longitude=longitude, date=date) + timedelta(hours=2)
            path = Sun.PathByDate(latitude=latitude, longitude=longitude, date=date,
                                  startTime=startTime, endTime=endTime, interval=minuteInterval,
                                 origin=origin, radius=radius, sides=uSides, north=north)
            # Capture the path's dictionary to re-apply later
            d = Topology.Dictionary(path)
            # Clip the path to above ground level
            path = Topology.Difference(path, cutter)
            path = Topology.SetDictionary(path, d)
            date_paths.append(path)
        diagram['date_paths'] = date_paths
        # Hourly paths
        hourly_paths = []
        for hour in range (0, 24, 1):
            hourly_path = Sun.PathByHour(latitude, longitude, hour, startDay=0, endDay=365, interval=dayInterval,
                                         origin=origin, radius=radius, sides=vSides*2, north=north)
            d = Topology.Dictionary(hourly_path)
            hourly_path = Topology.Difference(hourly_path, cutter)
            if Topology.IsInstance(hourly_path, "topology"):
                hourly_path = Topology.SetDictionary(hourly_path, d)
                hourly_paths.append(hourly_path)
        diagram['hourly_paths'] = hourly_paths
        if shell:
            shell_paths = []
            dates = [summer_solstice]
            delta = (winter_solstice - summer_solstice)/12
            for i in range(1,12):
                a_date = summer_solstice + delta*i
                if abs(a_date - equinox) < timedelta(hours=24*5):
                    dates.append(equinox)
                else:
                    dates.append(a_date)
            dates.append(winter_solstice)
            for date in dates:
                startTime = Sun.Sunrise(latitude=latitude, longitude=longitude, date=date) - timedelta(hours=2)
                endTime = Sun.Sunset(latitude=latitude, longitude=longitude, date=date) + timedelta(hours=2)
                shell_path = Sun.PathByDate(latitude=latitude, longitude=longitude, date=date,
                                      startTime=startTime, endTime=endTime, interval=minuteInterval,
                                      origin=origin, radius=radius, sides=uSides, north=north)
                # Clip the path to above ground level
                shell_path = Topology.Difference(shell_path, cutter)
                path_vertices = []
                for i in range(uSides+1):
                    u = float(i)/float(uSides)
                    v = Wire.VertexByParameter(shell_path, u)
                    path_vertices.append(v)
                shell_path = Wire.ByVertices(path_vertices, close=False)
                shell_paths.append(shell_path)
            a_shell = Shell.ByWires(shell_paths, triangulate=True, silent=True)
            d = Dictionary.ByKeysValues(["latitude", "longitude", "type"], [latitude, longitude, "shell"])
            a_shell = Topology.SetDictionary(a_shell, d)
            diagram['shell']= a_shell
        else:
            diagram['shell'] = None

        if compass:
            compass = Shell.Pie(origin=origin, radiusA=radius, radiusB=radius*0.1, sides=36, rings=10)
            d = Dictionary.ByKeysValues(["latitude", "longitude", "type"], [latitude, longitude, "compass"])
            compass = Topology.SetDictionary(compass, d)
            diagram['compass'] = compass
            edges = []
            for i in range(0, 4, 1):
                v2 = Topology.Translate(origin, 0, radius/float(9), 0)
                edge = Edge.ByVertices(origin, v2)
                edge = Topology.Rotate(edge, origin=origin, angle=90*i)
                edge = Topology.Rotate(edge, origin=origin, angle=-north)
                edges.append(edge)
            center = Topology.SelfMerge(Cluster.ByTopologies(edges))
            d = Dictionary.ByKeysValues(["latitude", "longitude", "type"], [latitude, longitude, "center"])
            center = Topology.SetDictionary(center, d)
            diagram['center'] = center
        else:
            ground = Wire.Circle(origin=origin, radius=radius, sides=36)
            d = Dictionary.ByKeysValues(["latitude", "longitude", "type"], [latitude, longitude, "ground"])
            ground = Topology.SetDictionary(ground, d)
            diagram['compass'] = None
            diagram['center'] = None
            diagram['ground']= ground
        return diagram