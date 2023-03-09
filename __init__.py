import sys
import os, re
from sys import platform

__version__ = '0.2.6'
__version_info__ = tuple([ int(num) for num in __version__.split('.')])

if platform == 'win32':
    os_name = 'windows'
else:
    os_name = 'linux'

sitePackagesFolderName = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bin", os_name)
test1 = os.path.dirname(os.path.realpath(__file__))
topologicFolderName = [filename for filename in os.listdir(sitePackagesFolderName) if filename.startswith("topologic")][0]
topologicPath = os.path.join(sitePackagesFolderName, topologicFolderName)
sys.path.append(topologicPath)

import topologic

