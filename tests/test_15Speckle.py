import sys
sys.path.append("/workspaces/topologicpy/src")

import topologicpy 
from topologicpy.Speckle import Speckle

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

    toplogy_obj = Speckle.TopologyBySpeckleObject(obj)

    for top in toplogy_obj:
        print(top)

#To run test enter your token and select desired stram, branch, and commit
# run_test()