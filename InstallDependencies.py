import sys, subprocess, pkg_resources

dependency_list = ['ipfshttpclient',
            'ladybug',
            'honeybee-energy',
            'honeybee-radiance',
            'openstudio',
            'py2neo',
            'pyvisgraph',
            'numpy',
            'pandas',
            'scipy',
            'torch',
            'networkx',
            'tqdm',
            'sklearn',
            'dgl',
            'plotly',
            'specklepy']

def install_dependency(module):
    # upgrade pip
    call = [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip']
    print(f"Installing {module}")
    if module == 'dgl':
        call = [sys.executable, '-m', 'pip', 'install', module, 'dglgo', '-f', 'https://data.dgl.ai/wheels/repo.html', '--upgrade', '-t', sys.path[0]]
    elif module == 'ladybug':
        call = [sys.executable, '-m', 'pip', 'install', 'ladybug-core', '-U', '--upgrade', '-t', sys.path[0]]
    elif module == 'honeybee-energy':
        call = [sys.executable, '-m', 'pip', 'install', 'honeybee-energy', '-U', '--upgrade', '-t', sys.path[0]]
    elif module == 'honeybee-radiance':
        call = [sys.executable, '-m', 'pip', 'install', 'honeybee-radiance', '-U', '--upgrade', '-t', sys.path[0]]
    else:
        call = [sys.executable, '-m', 'pip', 'install', module, '-t', sys.path[0]]
    subprocess.run(call)

def checkInstallation(module):
    returnValue = False
    if module == 'honeybee-energy':
        try:
            import honeybee
            import honeybee_energy
            from honeybee.model import Model
            returnValue = True
        except:
            pass
    elif module == 'honeybee-radiance':
        try:
            import honeybee_radiance
            returnValue = True
        except:
            pass
    else:
        try:
            import module
            returnValue = True
        except:
            pass

class InstallDependencies:
    @staticmethod
    def InstallDependencies():
        """
        Returns
        -------
        status : TYPE
            DESCRIPTION.

        """
        status = []
        installed_packages = {i.key: i.version for i in pkg_resources.working_set}
        flag = ''
        for module in dependency_list:
            if module in installed_packages.keys():
                flag = 'Already installed'
            else:
                try:
                    install_dependency(module)
                    flag = 'Successfully installed'
                except:
                    if checkInstallation(module):
                        flag = 'Successfully installed'
                    else:
                        flag = 'Failed to insall'
                    flag = 'Failed to install'
            status.append(f"{module}: {flag}.")
        return status