# Copyright (C) 2025
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

from specklepy.api.client import SpeckleClient
from specklepy.api.wrapper import StreamWrapper
from specklepy.api import operations
from specklepy.objects import Base
from specklepy.objects.other import Collection
from specklepy.objects.other import RenderMaterial
from specklepy.transports.server import ServerTransport
from specklepy.objects.geometry import (Mesh, Point, Polyline)

from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Face import Face
from topologicpy.Cell import Cell
from topologicpy.CellComplex import CellComplex
from topologicpy.Graph import Graph
from topologicpy.Dictionary import Dictionary

from collections import deque
from typing import List

class Speckle:

    @staticmethod
    def mesh_to_speckle(topology) -> Base:
        b = Base()
        b["name"] = "Topologic_Object"
        b["@displayValue"] = Speckle.mesh_to_speckle_mesh(topology)
        return b

    @staticmethod
    def mesh_to_speckle_mesh(topology, mantissa: int = 6) -> Mesh:

        geom = Topology.Geometry(topology, mantissa=mantissa)
        vertices = geom['vertices']
        faces = geom['faces']
        #m_verts: List[float] = []
        #m_faces: List[int] = []
        m_verts = []
        m_faces = []

        for v in vertices:
            m_verts.append(v[0])
            m_verts.append(v[1])
            m_verts.append(v[2])
        
        for f in faces:
            n = len(f)
            m_faces.append(n)
            for i in range(n):
                m_faces.append(f[i])

        speckle_mesh = Mesh(
            vertices=m_verts,
            faces=m_faces,
        )      
        speckle_mat = RenderMaterial()
        speckle_mat['Opacity'] = 0.5
        speckle_mesh["renderMaterial"] = speckle_mat
        return speckle_mesh

    @staticmethod
    def SpeckleBranchByID(branch_list, branch_id):
        """
        Parameters
        ----------
        branch_list : TYPE
            DESCRIPTION.
        branch_id : TYPE
            DESCRIPTION.

        Returns
        -------
        branch : TYPE
            DESCRIPTION.

        """
        # branch_list, branch_id = item
        for branch in branch_list:
            if branch.id == branch_id:
                return branch
        return None

    @staticmethod
    def BranchesByStream(client, stream):
        """
        Parameters
        ----------
        client : TYPE
            DESCRIPTION.
        stream : TYPE
            DESCRIPTION.

        Returns
        -------
        branches : TYPE
            DESCRIPTION.

        """
        # client, stream = item
        bList = client.branch.list(stream.id)
        branches = []
        for b in bList:
            branches.append(client.branch.get(stream.id, b.name))
        return branches
    
    @staticmethod
    def ClientByURL(url="https://app.speckle.systems/", token=None):
        """
        Parameters
        ----------
        url : TYPE
            DESCRIPTION.
        token : TYPE
            DESCRIPTION.

        Returns
        -------
        client : TYPE
            DESCRIPTION.

        """
        import getpass
        if token == None:
            token = getpass.getpass()
        if token == None:
            print("Speckle.ClientByHost - Error: Could not retrieve token. Returning None.")
            return None
        client = SpeckleClient(host=url) # or whatever your host is
        client.authenticate_with_token(token)
        return client
    
    @staticmethod
    def CommitByID(commit_list, commit_id):
        """
        Parameters
        ----------
        commit_list : TYPE
            DESCRIPTION.
        commit_id : TYPE
            DESCRIPTION.

        Returns
        -------
        commit : TYPE
            DESCRIPTION.

        """
        # commit_list, commit_id = item
        for commit in commit_list:
            if commit.id == commit_id:
                return commit
        return None
    
    @staticmethod
    def SpeckleCommitByURL(url, token):
        """
        Parameters
        ----------
        url : TYPE
            DESCRIPTION.
        token : TYPE
            DESCRIPTION.

        Returns
        -------
        commit : TYPE
            DESCRIPTION.

        """
        # url, token = item
        
        def streamByID(item):
            stream_list, stream_id = item
            for stream in stream_list:
                if stream.id == stream_id:
                    return stream
            return None

        def streamsByClient(client):
            return client.stream.list()
        
        def commitByID(item):
            commit_list, commit_id = item
            for commit in commit_list:
                if commit.id == commit_id:
                    return commit
            return None
        
        # provide any stream, branch, commit, object, or globals url
        wrapper = StreamWrapper(url)
        client = wrapper.get_client()
        client.authenticate_with_token(token)
        print("Client", client)
        streams = streamsByClient(client)
        print("Streams", streams)
        stream = streamByID([streams, wrapper.stream_id])
        print("Stream", stream)
        commits = client.commit.list(wrapper.stream_id)
        commit = commitByID([commits, wrapper.commit_id])
        print(commit)
        return commit
    
    @staticmethod
    def SpeckleCommitDelete(client, stream, commit, confirm):
        """
        Parameters
        ----------
        client : TYPE
            DESCRIPTION.
        stream : TYPE
            DESCRIPTION.
        commit : TYPE
            DESCRIPTION.
        confirm : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # client, stream, commit, confirm = item
        if confirm:
            try:
                deleted = client.commit.delete(stream_id=stream.id, commit_id=commit.id)
                return deleted
            except:
                return False
        return False
    
    @staticmethod
    def CommitsByBranch(branch):
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
        return branch.commits.items
    
    @staticmethod
    def SpeckleGlobalsByStream(client, stream):
        """
        Parameters
        ----------
        client : TYPE
            DESCRIPTION.
        stream : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # client, stream = item
        
        def processBase(base):
            dictionary = {}
            dynamic_member_names = base.get_dynamic_member_names()
            for dynamic_member_name in dynamic_member_names:
                attribute = base[dynamic_member_name]
                if isinstance(attribute, float) or isinstance(attribute, int) or isinstance(attribute, str) or isinstance(attribute, list):
                    dictionary[dynamic_member_name] = attribute
                if isinstance(attribute, Base):
                    dictionary[dynamic_member_name] = processBase(attribute)
            return dictionary
        
        transport = ServerTransport(client=client, stream_id=stream.id)

        # get the `globals` branch
        branch = client.branch.get(stream.id, "globals")

        # get the latest commit
        if len(branch.commits.items) > 0:
            latest_commit = branch.commits.items[0]

            # receive the globals object
            globs = operations.receive(latest_commit.referencedObject, transport)
            return processBase(globs)
        return None
    
    @staticmethod
    def Send(client, stream, branch, description, message, key, data, run):
        """
        Parameters
        ----------
        client : TYPE
            DESCRIPTION.
        stream : TYPE
            DESCRIPTION.
        branch : TYPE
            DESCRIPTION.
        description : TYPE
            DESCRIPTION.
        message : TYPE
            DESCRIPTION.
        key : TYPE
            DESCRIPTION.
        data : TYPE
            DESCRIPTION.
        run : TYPE
            DESCRIPTION.

        Returns
        -------
        commit : TYPE
            DESCRIPTION.

        """
        # client, stream, branch, description, message, key, data, run = item
        if not run:
            return None
        # create a base object to hold data
        base = Base()
        base[key] = data
        transport = ServerTransport(stream.id, client)
        # and send the data to the server and get back the hash of the object
        obj_id = operations.send(base, [transport])

        # now create a commit on that branch with your updated data!
        commit_id = client.commit.create(
            stream.id,
            obj_id,
            key,
            message=message,
        )
        print("COMMIT ID", commit_id)
        for commit in branch.commits.items:
            print("  VS. COMMIT.ID", commit.id)
            if commit.id == commit_id:
                return commit
        return None
    
    @staticmethod
    def Object(client, stream, branch, commit):
        #from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        def add_vertices(speckle_mesh, scale=1.0):
            sverts = speckle_mesh.vertices
            vertices = []
            if sverts and len(sverts) > 0:
                for i in range(0, len(sverts), 3):
                    vertices.append([float(sverts[i]) * scale, float(sverts[i + 1]) * scale, float(sverts[i + 2]) * scale])
            return vertices

        def add_faces(speckle_mesh):
            sfaces = speckle_mesh.faces
            faces = []
            if sfaces and len(sfaces) > 0:
                i = 0
                while i < len(sfaces):
                    n = sfaces[i]
                    if n < 3:
                        n += 3  # 0 -> 3, 1 -> 4
                    i += 1
                    faces.append([int(x) for x in sfaces[i : i + n]])
                    i += n
                return faces

        def mesh_to_native(speckle_mesh, scale=1.0):
            vertices = add_vertices(speckle_mesh, scale)
            faces = add_faces(speckle_mesh)
            topology = Topology.ByGeometry(vertices=vertices, edges=[], faces=faces)
            return topology
        
        transport = ServerTransport(stream.id, client)
        last_obj_id = commit.referencedObject
        speckle_mesh = operations.receive(obj_id=last_obj_id, remote_transport=transport)
        # print(speckle_mesh)
        return mesh_to_native(speckle_mesh["@display_value"])
    
    @staticmethod
    def SpeckleSendObjects(client, stream, branch, description, message, key, data, run):
        """
        Parameters
        ----------
        client : TYPE
            DESCRIPTION.
        stream : TYPE
            DESCRIPTION.
        branch : TYPE
            DESCRIPTION.
        description : TYPE
            DESCRIPTION.
        message : TYPE
            DESCRIPTION.
        key : TYPE
            DESCRIPTION.
        data : TYPE
            DESCRIPTION.
        run : TYPE
            DESCRIPTION.

        Returns
        -------
        commit : TYPE
            DESCRIPTION.

        """
        # client, stream, branch, description, message, key, data, run = item
        if not run:
            return None
        # create a base object to hold data
        base = Base()
        base[key] = data
        transport = ServerTransport(stream.id, client)
        # and send the data to the server and get back the hash of the object
        obj_id = operations.send(base, [transport])

        # now create a commit on that branch with your updated data!
        commit_id = client.commit.create(
            stream.id,
            obj_id,
            "gbxml",
            message=message,
        )
        print("COMMIT ID", commit_id)
        for commit in branch.commits.items:
            print("  VS. COMMIT.ID", commit.id)
            if commit.id == commit_id:
                return commit
        return None
    
    @staticmethod
    def SpeckleStreamByID(stream_list, stream_id):
        """
        Parameters
        ----------
        stream_list : TYPE
            DESCRIPTION.
        stream_id : TYPE
            DESCRIPTION.

        Returns
        -------
        stream : TYPE
            DESCRIPTION.

        """
        # stream_list, stream_id = item
        for stream in stream_list:
            if stream.id == stream_id:
                return stream
        return None
    
    @staticmethod
    def SpeckleStreamByURL(url, token):
        """
        Parameters
        ----------
        url : TYPE
            DESCRIPTION.
        token : TYPE
            DESCRIPTION.

        Returns
        -------
        stream : TYPE
            DESCRIPTION.

        """
        # url, token = item
        
        def streamByID(item):
            stream_list, stream_id = item
            for stream in stream_list:
                if stream.id == stream_id:
                    return stream
            return None

        def streamsByClient(client):
            return client.stream.list()
        
        # provide any stream, branch, commit, object, or globals url
        wrapper = StreamWrapper(url)
        client = wrapper.get_client()
        client.authenticate_with_token(token)
        streams = streamsByClient(client)
        stream = streamByID([streams, wrapper.stream_id])
        return stream

    @staticmethod
    def StreamsByClient(item):
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
        return item.stream.list()

    #######################################################
    # NEW METHODS TO PROCESS IFC

    @staticmethod
    def SpeckleObject(client, stream, branch, commit):
        transport = ServerTransport(stream.id, client)
        last_obj_id = commit.referencedObject
        speckle_mesh = operations.receive(obj_id=last_obj_id, remote_transport=transport)
        
        return speckle_mesh
    
    @staticmethod
    def get_name_from_IFC_name(name) ->  str:
        "Function is used to get clean speckle element name" 

        import re
        mapping = {
            "IFC_WALL_REG": r"(Wall)",
            "IFC_SLAB_REG": r"(Slab)",
            "IFC_WINDOW_REG": r"(Window)",
            "IFC_COLUMN_REG": r"(Column)",
            "IFC_DOOR_REG": r"(Door)",
            "IFC_SPACE_REG": r"(Space)"
        }

        for expression in mapping.values():
            res = re.findall(expression, name)
            if res:
                return res[0]
        

    @staticmethod
    def TopologyBySpeckleObject(obj):

        def from_points_to_vertices_3D(points) -> List[Vertex]:
            points_array_length = len(points)
            i = 0
            topologic_vertices = []
            while i < points_array_length:
                x = points[i]
                y = points[i+1]
                z = points[i+2]
                v = Vertex.ByCoordinates(x, y, z)
                topologic_vertices.append(v)

                i+=3
            return topologic_vertices

        def make_faces_from_indices(vertices, face_indices) -> List[Face]:
            "Helping function used to make a triangular faces"
            l = len(face_indices)
            i = 0
            triangles = []

            while i < l:
                curr = face_indices[i]
                i+=1
                lis = face_indices[i:curr+i]
                triangles.append(lis)
                i+=curr
            
            topologic_vertices = from_points_to_vertices_3D(vertices)
            faces_list = []

            for triangle in triangles:
                v1 = topologic_vertices[triangle[0]]
                v2 = topologic_vertices[triangle[1]]
                v3 = topologic_vertices[triangle[2]]
                f = Face.ByVertices([v1, v2, v3])
                faces_list.append(f)

            return faces_list

        #Stack used for the tree traversal
        stack = deque()
        stack.append(obj)

        #list of coordinates
        coords = []

        while len(stack) > 0:
            head = stack.pop()
                
            if isinstance(head, Collection):
                for el in head.elements:
                    stack.append(el)

            elif isinstance(head, Base):

                if hasattr(head, '@displayValue'):
                    meshes = head['@displayValue']
                    # print(head.name)
                    if isinstance(meshes, List):
                        full_name = head.name
                        short_name = Speckle.get_name_from_IFC_name(full_name)
                        speckle_id = head.id

                        mesh_faces = []
                        for mesh in meshes:
                            # print(mesh.get_serializable_attributes())
                            faces = make_faces_from_indices(vertices=mesh.vertices,
                                                            face_indices=mesh.faces)
                            mesh_faces.append(faces)

                        yield [short_name, full_name, speckle_id, mesh_faces]

                # stack.pop()