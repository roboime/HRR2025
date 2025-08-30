from setuptools import setup, find_packages

package_name = 'roboime_navigation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/navigation.launch.py']),
        ('share/' + package_name + '/config', ['config/navigation_config.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='RoboIME',
    maintainer_email='keller.felipe@ime.eb.br',
    description='Sistema de navegação para robô de futebol RoboIME - Localização, planejamento e controle',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Nó principal de localização (usando linhas do campo)
            'localization = navigation.localization_node:main',
            
            # Planejador de trajetória para futebol robótico
            'path_planner = navigation.path_planner_node:main',
            
            # Coordenador principal de navegação
            'navigation_manager = navigation.navigation_manager:main',
        ],
    },
)
