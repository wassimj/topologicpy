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


class Sun:
    """Solar positions, drawing geometry and geometric solar access.

    Weather and PV electrical modelling are intentionally separate.
    """

    @staticmethod
    def _Season(latitude, year, event):
        from datetime import datetime
        latitude = Sun._Number(latitude,'latitude',-90,90)
        year = datetime.now().year if year is None else year
        if isinstance(year,bool) or not isinstance(year,int) or not 1<=year<=9998:
            raise ValueError('year must be an integer in 1..9998')
        ephem = Sun._Ephem()
        months = {'winter':12,'summer':6,'spring':3,'autumn':9}
        month = months[event]
        if latitude<0:
            month = (month+5)%12+1
        method = ephem.next_solstice if event in ('winter','summer') else ephem.next_equinox
        return method(ephem.Date(f'{year}/{month}/1')).datetime()


    @staticmethod
    def WinterSolstice(latitude, year=None):
        """Winter solstice for the hemisphere and year, as naive UTC."""
        return Sun._Season(latitude,year,'winter')


    @staticmethod
    def SummerSolstice(latitude, year=None):
        """Summer solstice for the hemisphere and year, as naive UTC."""
        return Sun._Season(latitude,year,'summer')


    @staticmethod
    def SpringEquinox(latitude, year=None):
        """Spring equinox for the hemisphere and year, as naive UTC."""
        return Sun._Season(latitude,year,'spring')


    @staticmethod
    def AutumnEquinox(latitude, year=None):
        """Autumn equinox for the hemisphere and year, as naive UTC."""
        return Sun._Season(latitude,year,'autumn')


    @staticmethod
    def Vertex(latitude, longitude, date, origin=None, radius=0.5, north=0):
        """Sun drawing vertex at a positive model-unit radius from origin."""
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        radius = Sun._Number(radius,'radius',0)
        if radius<=0:
            raise ValueError('radius must be positive')
        origin = Vertex.Origin() if origin is None else origin
        if not Topology.IsInstance(origin,'Vertex'):
            raise ValueError('origin must be a Vertex')
        return Topology.TranslateByDirectionDistance(origin,
            direction=Sun.Vector(latitude,longitude,date,north,reverse=True),distance=radius)


    @staticmethod
    def Position(latitude, longitude, date, origin=None, radius=0.5, north=0, mantissa=6):
        """Sun drawing position as XYZ coordinates; use PositionData for physics."""
        from topologicpy.Vertex import Vertex
        return Vertex.Coordinates(Sun.Vertex(latitude,longitude,date,origin,radius,north),mantissa=mantissa)


    @staticmethod
    def Edge(latitude, longitude, date, origin=None, radius=0.5, north=0):
        """Drawing edge from the sun vertex toward the origin."""
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        origin = Vertex.Origin() if origin is None else origin
        return Edge.ByVertices(Sun.Vertex(latitude,longitude,date,origin,radius,north),origin)


    @staticmethod
    def _Zone(timeZone):
        from datetime import timezone, tzinfo
        from zoneinfo import ZoneInfo
        if timeZone is None:
            return timezone.utc
        if isinstance(timeZone, str):
            return ZoneInfo(timeZone)
        if isinstance(timeZone, tzinfo):
            return timeZone
        raise ValueError('timeZone must be an IANA zone name or tzinfo')


    @staticmethod
    def _Date(date, timeZone=None):
        """Normalize an instant to aware UTC; naive dates default to UTC.

        Ambiguous/nonexistent naive civil times are rejected. Supply an aware
        datetime with explicit fold/offset to select an ambiguous instant.
        """
        from datetime import datetime, timezone
        if not isinstance(date, datetime):
            raise ValueError('date must be a datetime')
        if date.tzinfo is None:
            zone = Sun._Zone(timeZone)
            a, b = date.replace(tzinfo=zone,fold=0), date.replace(tzinfo=zone,fold=1)
            if a.utcoffset() != b.utcoffset():
                raise ValueError('Ambiguous or nonexistent local time; supply an aware datetime')
            date = a
        result = date.astimezone(timezone.utc)
        if result.astimezone(date.tzinfo).replace(tzinfo=None) != date.replace(tzinfo=None):
            raise ValueError('Nonexistent local time')
        return result


    @staticmethod
    def _Number(value, name, lower=None, upper=None):
        import math
        if isinstance(value, bool):
            raise ValueError(name + ' must be a finite number')
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(name + ' must be a finite number') from exc
        if not math.isfinite(value) or (lower is not None and value < lower) or (upper is not None and value > upper):
            raise ValueError(name + ' is outside its valid range')
        return value


    @staticmethod
    def _Ephem():
        try:
            import ephem
        except ImportError as exc:
            raise ImportError('Sun requires ephem. Install it in this environment with: python -m pip install "topologicpy[sun]"') from exc
        return ephem


    @staticmethod
    def PositionsByDates(latitude, longitude, dates, north=0, *, engine='ephem',
                         elevation=0, pressure=101325, temperature=15,
                         timeZone=None, apparent=True):
        """Numeric records at explicit instants; no geometry objects are created.

        Latitude is -90..90; east longitude -180..180. elevation is metres,
        pressure Pa, temperature Celsius. Naive datetimes mean UTC unless an
        explicit timeZone is supplied. Output timestamps are aware UTC.
        engine is explicitly 'ephem' or optional 'pvlib'; no implicit fallback.
        Azimuth is geographic clockwise from north. Vectors are model XYZ,
        +Z up, model north rotated clockwise from +Y by north degrees.
        Apparent elevation includes refraction; geometric elevation does not.
        toSun/fromSun use the selected elevation and have unit length without
        display rounding. All callers should share this same record series.
        """
        import math
        latitude = Sun._Number(latitude,'latitude',-90,90)
        longitude = Sun._Number(longitude,'longitude',-180,180)
        north = Sun._Number(north,'north')
        elevation = Sun._Number(elevation,'elevation')
        pressure = Sun._Number(pressure,'pressure',0)
        temperature = Sun._Number(temperature,'temperature',-273.0)
        dates = [Sun._Date(d,timeZone) for d in dates]
        if engine not in ('ephem','pvlib'):
            raise ValueError("engine must be 'ephem' or 'pvlib'")
        if not dates:
            return []
        positions = []
        if engine == 'pvlib':
            try:
                import pvlib
                import pandas as pd
            except ImportError as exc:
                raise ImportError('The selected Sun engine requires pvlib') from exc
            table = pvlib.solarposition.get_solarposition(pd.DatetimeIndex(dates),latitude,longitude,
                altitude=elevation,pressure=pressure,temperature=temperature)
            positions = [(float(row.azimuth),float(row.elevation),float(row.apparent_elevation))
                         for row in table.itertuples()]
        else:
            ephem = Sun._Ephem()
            observer = ephem.Observer()
            observer.lat,observer.lon = str(latitude),str(longitude)
            observer.elevation,observer.temp = elevation,temperature
            body = ephem.Sun()
            for date in dates:
                observer.date = date
                observer.pressure = 0
                body.compute(observer)
                az, geometric = math.degrees(body.az),math.degrees(body.alt)
                observer.pressure = pressure/100.0
                body.compute(observer)
                positions.append((az,geometric,math.degrees(body.alt)))
        records = []
        for date,(az,geometric,refracted) in zip(dates,positions):
            altitude = refracted if apparent else geometric
            a,h = math.radians(az+north),math.radians(altitude)
            vector = [math.cos(h)*math.sin(a),math.cos(h)*math.cos(a),math.sin(h)]
            records.append({'timestamp':date,'azimuth':az,'geometricElevation':geometric,
                'apparentElevation':refracted,'altitude':altitude,'zenith':90-altitude,
                'aboveHorizon':altitude>0,'toSun':vector,'fromSun':[-v for v in vector],
                'engine':engine,'apparent':bool(apparent),'north':north,
                'elevation':elevation,'pressure':pressure,'temperature':temperature})
        return records


    @staticmethod
    def PositionData(latitude, longitude, date, north=0, **kwargs):
        """One record; see PositionsByDates for units, time and vector conventions."""
        return Sun.PositionsByDates(latitude,longitude,[date],north,**kwargs)[0]


    @staticmethod
    def Azimuth(latitude, longitude, date):
        """Geographic clockwise azimuth in degrees; naive datetime means UTC."""
        return Sun.PositionData(latitude,longitude,date)['azimuth']


    @staticmethod
    def Altitude(latitude, longitude, date):
        """Apparent solar-centre elevation, degrees; naive datetime means UTC."""
        return Sun.PositionData(latitude,longitude,date)['altitude']


    @staticmethod
    def Vector(latitude, longitude, date, north=0, *, reverse=False, **kwargs):
        """Unit incoming sunlight vector (sun to origin), preserving legacy sign.

        Set reverse=True for the receiver-to-sun visibility direction. Numerical
        output is unrounded. See PositionsByDates for optional engine/atmosphere.
        """
        record = Sun.PositionData(latitude,longitude,date,north,**kwargs)
        return record['toSun'] if reverse else record['fromSun']


    @staticmethod
    def EventsByDate(latitude, longitude, date, *, timeZone='UTC', elevation=0,
                     pressure=101325, temperature=15, horizon=0, useCenter=False):
        """Rising/setting events INSIDE one civil date in the specified zone.

        Output events and day bounds are aware datetimes in that zone. Sunrise
        and sunset may occur in either order in a UTC civil day. They must not
        be independently paired as a daylight duration. daylightIntervals gives
        clipped, correctly paired intervals, including midnight-spanning light.
        status is normal, polar_day, polar_night, or transition (one event).
        Defaults use the apparent upper limb at horizon=0, matching ephem's
        event convention. useCenter=True selects the solar centre. PositionData
        aboveHorizon instead refers to the solar centre; it is a distinct test.
        """
        from datetime import datetime, time, timedelta, timezone
        ephem = Sun._Ephem()
        latitude = Sun._Number(latitude,'latitude',-90,90)
        longitude = Sun._Number(longitude,'longitude',-180,180)
        elevation = Sun._Number(elevation,'elevation')
        pressure = Sun._Number(pressure,'pressure',0)
        temperature = Sun._Number(temperature,'temperature',-273)
        horizon = Sun._Number(horizon,'horizon',-90,90)
        zone = Sun._Zone(timeZone)
        civil = date.astimezone(zone).date() if isinstance(date,datetime) and date.tzinfo else date.date() if isinstance(date,datetime) else date
        start = datetime.combine(civil,time(),zone)
        end = datetime.combine(civil+timedelta(days=1),time(),zone)
        a,b = start.astimezone(timezone.utc),end.astimezone(timezone.utc)
        if b <= a:
            raise ValueError('Civil date has no positive duration in this timezone')
        observer = ephem.Observer()
        observer.lat,observer.lon = str(latitude),str(longitude)
        observer.elevation,observer.pressure,observer.temp = elevation,pressure/100,temperature
        observer.horizon = str(horizon)
        events = []
        for kind,method in [('sunrise',observer.next_rising),('sunset',observer.next_setting)]:
            observer.date = a-timedelta(microseconds=1)
            try:
                event = method(ephem.Sun(),use_center=useCenter).datetime().replace(tzinfo=timezone.utc)
            except (ephem.AlwaysUpError,ephem.NeverUpError):
                continue
            while event < b:
                if event >= a:
                    events.append((event,kind))
                observer.date = event+timedelta(seconds=1)
                try:
                    event = method(ephem.Sun(),use_center=useCenter).datetime().replace(tzinfo=timezone.utc)
                except (ephem.AlwaysUpError,ephem.NeverUpError):
                    break
        events.sort()
        if events:
            daylight = events[0][1] == 'sunset'
        else:
            observer.date = a+(b-a)/2
            body = ephem.Sun(observer)
            import math
            limb = 0 if useCenter else math.degrees(body.radius)
            daylight = math.degrees(body.alt)+limb > horizon
        initial = daylight
        cursor, intervals = a,[]
        for instant,kind in events:
            if daylight and instant>cursor:
                intervals.append((cursor.astimezone(zone),instant.astimezone(zone)))
            cursor,daylight = instant,kind=='sunrise'
        if daylight and b>cursor:
            intervals.append((cursor.astimezone(zone),end))
        status = ('polar_day' if initial else 'polar_night') if not events else 'normal' if len(events)>1 else 'transition'
        return {'date':civil.isoformat(),'timeZone':str(zone),'start':start,'end':end,
                'sunrise':next((t.astimezone(zone) for t,k in events if k=='sunrise'),None),
                'sunset':next((t.astimezone(zone) for t,k in events if k=='sunset'),None),
                'events':[{'timestamp':t.astimezone(zone),'type':k} for t,k in events],
                'daylightIntervals':intervals,'status':status,'useCenter':bool(useCenter),'horizon':horizon}


    @staticmethod
    def Sunrise(latitude, longitude, date, *, timeZone='UTC'):
        """Sunrise inside the civil date, returned as naive UTC; None if absent.

        Use EventsByDate for timezone-aware records and polar/interval status.
        """
        from datetime import timezone
        event = Sun.EventsByDate(latitude,longitude,date,timeZone=timeZone)['sunrise']
        return event.astimezone(timezone.utc).replace(tzinfo=None) if event else None


    @staticmethod
    def Sunset(latitude, longitude, date, *, timeZone='UTC'):
        """Sunset inside the civil date as naive UTC, or None; see EventsByDate."""
        from datetime import timezone
        event = Sun.EventsByDate(latitude,longitude,date,timeZone=timeZone)['sunset']
        return event.astimezone(timezone.utc).replace(tzinfo=None) if event else None


    @staticmethod
    def _Times(start, end, interval):
        from datetime import timedelta
        interval = Sun._Number(interval,'interval',0)
        if interval <= 0:
            raise ValueError('interval must be positive')
        start,end = Sun._Date(start),Sun._Date(end)
        if end < start:
            raise ValueError('endTime must not precede startTime')
        if (end-start).total_seconds()/60/interval > 1000000:
            raise ValueError('Too many time samples')
        delta = timedelta(minutes=interval)
        if delta.total_seconds() <= 0:
            raise ValueError('interval is below datetime resolution')
        values,current = [],start
        while current < end:
            values.append(current)
            current += delta
        values.append(end)
        return values


    @staticmethod
    def VerticesByDate(latitude, longitude, date, startTime=None, endTime=None,
                       interval=60, origin=None, radius=0.5, north=0, *, timeZone='UTC'):
        """Time-sampled sun vertices, including exact interval endpoints.

        With no start/end, sample the civil day's daylight intervals; polar
        night returns []. Explicit bounds include nighttime and denote instants.
        Endpoint vertices are drawing samples, not integration weights.
        """
        radius = Sun._Number(radius,'radius',0)
        interval = Sun._Number(interval,'interval',0)
        if radius<=0:
            raise ValueError('radius must be positive')
        if interval <= 0:
            raise ValueError('interval must be positive')
        if startTime is None and endTime is None:
            events = Sun.EventsByDate(latitude,longitude,date,timeZone=timeZone)
            spans = events['daylightIntervals']
        else:
            if startTime is None or endTime is None:
                events = Sun.EventsByDate(latitude,longitude,date,timeZone=timeZone)
                startTime = startTime if startTime is not None else events['start']
                endTime = endTime if endTime is not None else events['end']
            spans = [(Sun._Date(startTime,timeZone),Sun._Date(endTime,timeZone))]
        times = [t for a,b in spans for t in Sun._Times(a,b,interval)]
        return [Sun.Vertex(latitude,longitude,t,origin,radius,north) for t in times]


    @staticmethod
    def _Path(vertices, sides=None, close=False):
        from topologicpy.Wire import Wire
        if sides is not None and (isinstance(sides,bool) or not isinstance(sides,int) or sides<1):
            raise ValueError('sides must be a positive integer')
        if close and sides is not None and sides<3:
            raise ValueError('A closed path needs at least three sides')
        if len(vertices)<2:
            return None
        wire = Wire.ByVertices(vertices,close=close)
        if sides is not None and wire is not None:
            count = sides if close else sides+1
            values = [Wire.VertexByParameter(wire,i/sides) for i in range(count)]
            wire = Wire.ByVertices(values,close=close)
        return wire

    @staticmethod
    def _TagPath(path, metadata):
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        if path is None:
            return None
        return Topology.SetDictionary(path,Dictionary.ByKeysValues(list(metadata),list(metadata.values())))


    @staticmethod
    def PathByDate(latitude, longitude, date, startTime=None, endTime=None,
                   interval=60, origin=None, radius=0.5, sides=None, north=0, *, timeZone='UTC'):
        """Open daylight path(s); disconnected civil-day spans return a Cluster.

        Explicit bounds create one open path, including nighttime if requested.
        sides resamples drawing geometry, never the numerical time series.
        """
        from topologicpy.Cluster import Cluster
        if startTime is None and endTime is None:
            spans = Sun.EventsByDate(latitude,longitude,date,timeZone=timeZone)['daylightIntervals']
            paths = [Sun._Path(Sun.VerticesByDate(latitude,longitude,date,a,b,interval,
                origin,radius,north,timeZone=timeZone),sides) for a,b in spans]
            paths = [p for p in paths if p is not None]
            path = paths[0] if len(paths)==1 else Cluster.ByTopologies(paths) if paths else None
        else:
            path = Sun._Path(Sun.VerticesByDate(latitude,longitude,date,startTime,endTime,
                interval,origin,radius,north,timeZone=timeZone),sides)
        return Sun._TagPath(path,{'latitude':latitude,'longitude':longitude,'date':str(date),
            'startTime':str(startTime),'endTime':str(endTime),'interval':interval,'type':'date',
            'timeZone':str(timeZone)})


    @staticmethod
    def VerticesByHour(latitude, longitude, hour, startDay=1, endDay=None, interval=5,
                       origin=None, radius=0.5, north=0, *, year=None, timeZone='UTC'):
        """Sun vertices at numeric civil hour [0,24), inclusive day range.

        year defaults to the current year; endDay defaults to 365/366. An exact
        endDay sample is appended when spacing does not divide the range.
        Ambiguous/nonexistent DST wall-clock times raise; use explicit dated
        instants with PositionsByDates when a fixed civil hour is ambiguous.
        """
        from datetime import datetime,timedelta
        import calendar
        year = datetime.now().year if year is None else year
        if isinstance(year,bool) or not isinstance(year,int) or not 1<=year<=9999:
            raise ValueError('year must be an integer in 1..9999')
        hour = Sun._Number(hour,'hour',0,24)
        if hour>=24:
            raise ValueError('hour must be less than 24')
        final = 366 if calendar.isleap(year) else 365
        endDay = final if endDay is None else endDay
        for name,value in [('startDay',startDay),('endDay',endDay),('interval',interval)]:
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(name+' must be a positive integer')
        if startDay>endDay or endDay>final:
            raise ValueError('Invalid day-of-year range')
        days = list(range(startDay,endDay+1,interval))
        if days[-1]!=endDay:
            days.append(endDay)
        dates = [Sun._Date(datetime(year,1,1)+timedelta(days=d-1,hours=hour),timeZone) for d in days]
        return [Sun.Vertex(latitude,longitude,d,origin,radius,north) for d in dates]


    @staticmethod
    def PathByHour(latitude, longitude, hour, startDay=1, endDay=None, interval=5,
                   origin=None, radius=0.5, sides=None, north=0, *, year=None,
                   timeZone='UTC', close=False):
        """Open hourly path by default; close=True explicitly joins endpoints."""
        from datetime import datetime
        import calendar
        year = datetime.now().year if year is None else year
        endDay = (366 if calendar.isleap(year) else 365) if endDay is None else endDay
        path = Sun._Path(Sun.VerticesByHour(latitude,longitude,hour,startDay,endDay,
            interval,origin,radius,north,year=year,timeZone=timeZone),sides,close)
        return Sun._TagPath(path,{'latitude':latitude,'longitude':longitude,'hour':hour,
            'startDay':startDay,'endDay':endDay,'interval':interval,'type':'hour',
            'year':year,'timeZone':str(timeZone)})


    @staticmethod
    def _Receivers(points, normals):
        import numpy as np
        from topologicpy.BVH import BVH
        p,n = BVH._RayArrays(points,normals)
        return p,n


    @staticmethod
    def Visibility(points, normals, positions, scene, *, batchSize=4096,
                   minDistance=None, maxDistance=float('inf'), maxPairs=10000000, cancelled=None):
        """Geometric sun visibility for receiver XYZs/normals and position records.

        Returns uint8 states shaped (receivers,times): 0 visible, 1 occluded,
        2 back-facing, 3 below horizon (precedence: horizon, back-facing, hit).
        Points and normals are paired; a single normal broadcasts. Model-space
        directions come from toSun, with no implicit sign conversion. Explicit
        numeric directions must first be wrapped as records with toSun and
        aboveHorizon. This is point sampling, not panel electrical shading.
        Work is chunked; maxPairs bounds the materialized output. The caller
        supplies receiver cell areas and integration durations separately.
        """
        import numpy as np
        from topologicpy.BVH import BVH
        points,normals = Sun._Receivers(points,normals)
        positions = list(positions)
        if isinstance(batchSize,bool) or not isinstance(batchSize,int) or batchSize<1:
            raise ValueError('batchSize must be a positive integer')
        count = len(points)*len(positions)
        if not isinstance(maxPairs,int) or maxPairs<0 or count>maxPairs:
            raise ValueError('Visibility output exceeds maxPairs')
        # Validate scene and distance bounds even if all directions are night.
        BVH.IsOccluded(scene,[],[],minDistance,maxDistance,batchSize,cancelled)
        if positions:
            _,directions = BVH._RayArrays([0,0,0],[p['toSun'] for p in positions])
        else:
            directions = np.empty((0,3))
        above = np.asarray([bool(p['aboveHorizon']) for p in positions])
        flat = np.empty(count,dtype=np.uint8)
        for start in range(0,count,batchSize):
            if cancelled is not None and cancelled():
                raise InterruptedError('Solar visibility cancelled')
            ids = np.arange(start,min(start+batchSize,count))
            r,t = ids//len(positions),ids%len(positions)
            front = np.einsum('ij,ij->i',normals[r],directions[t])>1e-12
            state = np.where(~above[t],3,np.where(~front,2,0)).astype(np.uint8)
            active = np.flatnonzero(state==0)
            if len(active):
                blocked = BVH.IsOccluded(scene,points[r[active]],directions[t[active]],
                    minDistance,maxDistance,batchSize,cancelled)
                state[active[blocked]] = 1
            flat[start:start+len(ids)] = state
        return {'states':flat.reshape(len(points),len(positions)),
                'legend':{0:'visible',1:'occluded',2:'back_facing',3:'below_horizon'},
                'timestamps':[p.get('timestamp') for p in positions],
                'sampleCount':len(points),'positionCount':len(positions)}


    @staticmethod
    def SolarAccess(points, normals, positions, scene, durations=None, *,
                    timeEdges=None, areas=None, **kwargs):
        """Integrate point sunlight hours, with explicit interval durations.

        Supply either durations in seconds (one per position), or N+1 aware/
        UTC timeEdges. Positions must lie inside their respective intervals.
        Duration weights are NOT inferred from drawing paths or treated as
        cosine weights. This is geometric sunlight duration, not irradiation.
        Optional positive cell areas give an area-weighted mean, not XY area.
        Zero-duration endpoint samples are allowed. Total duration must be >0.
        """
        import numpy as np
        positions = list(positions)
        if (durations is None)==(timeEdges is None):
            raise ValueError('Supply durations OR timeEdges')
        if timeEdges is not None:
            edges = [Sun._Date(d) for d in timeEdges]
            if len(edges)!=len(positions)+1:
                raise ValueError('timeEdges must have N+1 entries')
            seconds = np.asarray([(b-a).total_seconds() for a,b in zip(edges,edges[1:])])
            if np.any(seconds<=0):
                raise ValueError('timeEdges must be strictly increasing')
            for i,record in enumerate(positions):
                stamp = Sun._Date(record['timestamp'])
                if not edges[i]<=stamp<=edges[i+1]:
                    raise ValueError('Solar position is outside its time interval')
        else:
            seconds = np.asarray(durations,dtype=float)
        if seconds.shape!=(len(positions),) or not np.isfinite(seconds).all() or np.any(seconds<0) or not np.isfinite(seconds.sum()) or seconds.sum()<=0:
            raise ValueError('Durations must be finite, nonnegative and total positive')
        result = Sun.Visibility(points,normals,positions,scene,**kwargs)
        states = result['states']
        hours = (states==0) @ (seconds/3600)
        potential = (states<=1) @ (seconds/3600)
        result.update(sunlightHours=hours,unshadedHours=potential,
            occludedHours=(states==1) @ (seconds/3600),durationSeconds=seconds,
            sunlightFraction=hours/(seconds.sum()/3600))
        if areas is not None:
            weights = np.asarray(areas,dtype=float)
            if weights.shape!=(len(hours),) or not np.isfinite(weights).all() or np.any(weights<=0):
                raise ValueError('areas must be positive and finite, one per receiver')
            result['areaWeightedSunlightHours'] = float(np.average(hours,weights=weights))
            result['area'] = float(weights.sum())
        return result


    @staticmethod
    def SkyPatches(azimuthDivisions=36, altitudeDivisions=12, north=0):
        """Equal-solid-angle upper-hemisphere patches, sampled at mu midpoints.

        mu=sin(altitude); patch solidAngle is steradians. Directions are model
        space. These are quadrature patches, not a named standard sky basis.
        """
        import math
        for value in (azimuthDivisions,altitudeDivisions):
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError('Sky divisions must be positive integers')
        if azimuthDivisions*altitudeDivisions>1000000:
            raise ValueError('Too many sky patches')
        north = Sun._Number(north,'north')
        patches = []
        solid = 2*math.pi/(azimuthDivisions*altitudeDivisions)
        for j in range(altitudeDivisions):
            mu = (j+0.5)/altitudeDivisions
            h = math.asin(mu)
            for i in range(azimuthDivisions):
                az = (i+0.5)*360/azimuthDivisions
                a = math.radians(az+north)
                patches.append({'toSun':[math.cos(h)*math.sin(a),math.cos(h)*math.cos(a),mu],
                    'aboveHorizon':True,'azimuth':az,'altitude':math.degrees(h),'solidAngle':solid})
        return patches


    @staticmethod
    def SkyVisibility(points, normals, scene, *, azimuthDivisions=36,
                      altitudeDivisions=12, north=0, **kwargs):
        """Return patch visibility and projected sky-view factors.

        skyViewFactor=sum(visible * max(n dot direction,0) * solidAngle)/pi.
        An unobstructed upward surface is 1, a vertical surface approximately
        0.5, and a downward surface 0. Ground reflection is excluded. Patch
        solid angles and weights are retained for anisotropic sky integration;
        this method itself assumes uniform sky radiance. Refine divisions for
        convergence. Visibility can be reused while geometry is unchanged.
        """
        import numpy as np
        points,normals = Sun._Receivers(points,normals)
        patches = Sun.SkyPatches(azimuthDivisions,altitudeDivisions,north)
        result = Sun.Visibility(points,normals,patches,scene,**kwargs)
        directions = np.asarray([p['toSun'] for p in patches])
        solid = np.asarray([p['solidAngle'] for p in patches])
        weights = np.maximum(normals @ directions.T,0)*solid[None,:]/np.pi
        visible = result['states']==0
        result.update(patches=patches,solidAngles=solid,projectedWeights=weights,
                      skyViewFactor=np.sum(weights*visible,axis=1),
                      unobstructedSkyViewFactor=weights.sum(axis=1))
        return result


    @staticmethod
    def Diagram(latitude, longitude, minuteInterval=30, dayInterval=15,
                origin=None, radius=0.5, uSides=180, vSides=180, north=0,
                compass = False, shell=False, *, year=None, timeZone="UTC"):
        """
        Returns the sun diagram based on the input parameters. See https://hyperfinearchitecture.com/how-to-read-sun-path-diagrams/.

        Parameters
        ----------
        latitude : float
            The input latitude. See https://en.wikipedia.org/wiki/Latitude.
        longitude : float
            The input longitude. See https://en.wikipedia.org/wiki/Longitude.
        minuteInterval : int , optional
            The interval in minutes to compute the sun location for the date path. Default is 30.
        dayInterval : int , optional
            The interval in days for the hourly path to compute the sun location. Default is 15.
        origin : topologic_core.Vertex , optional
            The desired origin of the world. If set to None, the origin will be set to (0,0,0). Default is None.
        radius : float , optional
            The desired radius of the sun orbit. Default is 0.5.
        uSides : int , optional
            The number of sides to divide the diagram horizontally (along the azimuth). Default is 180.
        vSides : int , optional
            The number of sides to divide the diagram paths vertically (along the altitude). Default is 180.
        north : float, optional
            The desired compass angle of the north direction. Default is 0 which points in the positive Y-axis direction.
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

        Sun._Number(radius,"radius",0)
        if radius <= 0:
            raise ValueError("radius must be positive")
        Sun._Number(latitude,"latitude",-90,90)
        Sun._Number(longitude,"longitude",-180,180)
        if origin == None:
            origin = Vertex.Origin()

        cutter = Cell.Prism(origin=origin, width=radius*4, length=radius*4, height=radius*2)
        cutter = Topology.Rotate(cutter, origin=origin, angle=-north)
        cutter = Topology.Translate(cutter, 0,0,-radius)
        now = datetime.now()
        year = now.year if year is None else year
        diagram = {}
        winter_solstice = Sun.WinterSolstice(latitude=latitude, year=year)
        summer_solstice = Sun.SummerSolstice(latitude=latitude, year=year)
        equinox = Sun.AutumnEquinox(latitude=latitude, year=year)
        dates = [winter_solstice, equinox, summer_solstice]
        date_paths = []
        for date in dates:
            startTime = Sun.Sunrise(latitude=latitude, longitude=longitude, date=date, timeZone=timeZone)
            endTime = Sun.Sunset(latitude=latitude, longitude=longitude, date=date, timeZone=timeZone)
            if startTime is None or endTime is None:
                continue
            if endTime <= startTime:
                endTime += timedelta(days=1)
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
            hourly_path = Sun.PathByHour(latitude, longitude, hour, startDay=1, endDay=None, interval=dayInterval,
                                         origin=origin, radius=radius, sides=vSides*2, north=north, year=year, timeZone=timeZone, close=True)
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
                startTime = Sun.Sunrise(latitude=latitude, longitude=longitude, date=date, timeZone=timeZone)
                endTime = Sun.Sunset(latitude=latitude, longitude=longitude, date=date, timeZone=timeZone)
                if startTime is None or endTime is None:
                    continue
                if endTime <= startTime:
                    endTime += timedelta(days=1)
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
            a_shell = Shell.ByWires(shell_paths, triangulate=True, silent=True) if len(shell_paths)>1 else None
            d = Dictionary.ByKeysValues(["latitude", "longitude", "type"], [latitude, longitude, "shell"])
            a_shell = Topology.SetDictionary(a_shell, d) if a_shell is not None else None
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
            ground = Wire.Circle(origin=origin, radius=radius, sides=36, polyline=True)
            d = Dictionary.ByKeysValues(["latitude", "longitude", "type"], [latitude, longitude, "ground"])
            ground = Topology.SetDictionary(ground, d)
            diagram['compass'] = None
            diagram['center'] = None
            diagram['ground']= ground
        return diagram
