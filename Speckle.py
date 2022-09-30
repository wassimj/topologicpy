from specklepy.api.client import SpeckleClient
from specklepy.api.wrapper import StreamWrapper
from specklepy.api import operations
from specklepy.objects import Base
from specklepy.transports.server import ServerTransport

class Speckle:
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
    def SpeckleBranchesByStream(client, stream):
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
    def SpeckleClientByHost(url, token):
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
        # url, token = item
        client = SpeckleClient(host=url) # or whatever your host is
        client.authenticate_with_token(token)
        return client
    
    @staticmethod
    def SpeckleClientByURL(url, token):
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
        # url, token = item
        # provide any stream, branch, commit, object, or globals url
        wrapper = StreamWrapper(url)
        client = wrapper.get_client()
        client.authenticate_with_token(token)
        return client
    
    @staticmethod
    def SpeckleCommitByID(commit_list, commit_id):
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
    def SpeckleCommitsByBranch(item):
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
        return item.commits.items
    
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
    def SpeckleSend(client, stream, branch, description, message, key, data, run):
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
    def SpeckleStreamsByClient(item):
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