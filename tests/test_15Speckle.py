import sys
sys.path.append("/workspaces/topologicpy/src")

import topologicpy 
from topologicpy.Speckle import Speckle
from topologicpy.Topology import Topology

def run_test(token):
    TOKEN = token

    client = Speckle.ClientByURL(url="https://app.speckle.systems/", token=TOKEN)

    streams = Speckle.StreamsByClient(client)
    stream = streams[1]

    branches = Speckle.BranchesByStream(client=client, stream=stream)
    branch = branches[0]

    commits = Speckle.CommitsByBranch(branch=branch)
    commit = commits[0]

    obj = Speckle.SpeckleObject(client=client, stream=stream, branch=branch, commit=commit)

    topology_obj = Speckle.TopologyBySpeckleObject(obj)

    first_cell = next(topology_obj)
    Topology.Show(first_cell, renderer='browser')

#To run test enter your token and select desired stram, branch, and commit
run_test("")