#!/usr/bin/env python3

import os

# Definição dos pacotes
packages = [
    {
        'name': 'roboime_behavior',
        'description': 'Pacote de comportamento para o robô de futebol RoboIME',
        'dependencies': ['rclpy', 'std_msgs', 'geometry_msgs', 'roboime_msgs']
    },
    {
        'name': 'roboime_motion',
        'description': 'Pacote de controle de movimento para o robô de futebol RoboIME',
        'dependencies': ['rclpy', 'std_msgs', 'geometry_msgs', 'roboime_msgs', 'sensor_msgs']
    },
    {
        'name': 'roboime_navigation',
        'description': 'Pacote de navegação para o robô de futebol RoboIME',
        'dependencies': ['rclpy', 'std_msgs', 'geometry_msgs', 'roboime_msgs', 'nav_msgs']
    },
    {
        'name': 'roboime_perception',
        'description': 'Pacote de percepção para o robô de futebol RoboIME',
        'dependencies': ['rclpy', 'std_msgs', 'geometry_msgs', 'roboime_msgs', 'sensor_msgs', 'cv_bridge']
    },
    {
        'name': 'roboime_msgs',
        'description': 'Pacote de mensagens para o robô de futebol RoboIME',
        'dependencies': ['std_msgs', 'geometry_msgs', 'sensor_msgs', 'rosidl_default_generators'],
        'build_type': 'ament_cmake'
    },
    {
        'name': 'roboime_bringup',
        'description': 'Pacote de inicialização para o robô de futebol RoboIME',
        'dependencies': ['rclpy', 'roboime_behavior', 'roboime_motion', 'roboime_navigation', 'roboime_perception']
    }
]

# Criar os pacotes
for package in packages:
    # Definir o tipo de build e buildtool
    build_type = package.get('build_type', 'ament_python')
    buildtool = 'ament_cmake' if build_type == 'ament_cmake' else 'ament_python'
    
    # Criar o diretório do pacote
    package_dir = f'src/{package["name"].replace("roboime_", "")}'
    os.makedirs(package_dir, exist_ok=True)
    
    # Criar package.xml
    package_xml = f'''<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>{package['name']}</name>
  <version>0.1.0</version>
  <description>{package['description']}</description>
  <maintainer email="seu-email@example.com">Seu Nome</maintainer>
  <license>MIT</license>

  <buildtool_depend>{buildtool}</buildtool_depend>
'''
    
    # Adicionar dependências
    for dep in package['dependencies']:
        package_xml += f'  <depend>{dep}</depend>\n'
    
    # Finalizar package.xml
    package_xml += f'''
  <export>
    <build_type>{build_type}</build_type>
  </export>
</package>
'''
    
    with open(f'{package_dir}/package.xml', 'w') as f:
        f.write(package_xml)
    
    # Criar setup.py para pacotes Python
    if build_type == 'ament_python':
        # Criar diretório resource
        os.makedirs(f'{package_dir}/resource', exist_ok=True)
        
        # Criar arquivo de marcador de pacote
        with open(f'{package_dir}/resource/{package["name"]}', 'w') as f:
            pass
        
        # Criar setup.py
        setup_py = f'''from setuptools import setup

package_name = '{package["name"]}'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Seu Nome',
    maintainer_email='seu-email@example.com',
    description='{package["description"]}',
    license='MIT',
    tests_require=['pytest'],
    entry_points={{
        'console_scripts': [
        ],
    }},
)
'''
        
        with open(f'{package_dir}/setup.py', 'w') as f:
            f.write(setup_py)
        
        # Criar diretório do pacote Python
        os.makedirs(f'{package_dir}/{package["name"]}', exist_ok=True)
        
        # Criar __init__.py
        with open(f'{package_dir}/{package["name"]}/__init__.py', 'w') as f:
            f.write('# Pacote ROS2 para o robô de futebol RoboIME\n')

print("Pacotes criados com sucesso!") 