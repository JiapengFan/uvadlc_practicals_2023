import yaml

file_name = 'dl2023_gpu'

with open(f'{file_name}.yml', 'r') as yaml_file:
    data = yaml.load(yaml_file, Loader=yaml.FullLoader)

    requirements = []
    for dep in data['dependencies']:
        if isinstance(dep, str):
            package, package_version = dep.split('=')
            if package != 'python' or package != 'pip':
                requirements.append(package + '==' + package_version)
        elif isinstance(dep, dict):
            for preq in dep.get('pip', []):
                requirements.append(preq)

with open(f'{file_name}.txt', 'w') as fp:
    for requirement in requirements:
       print(requirement, file=fp)