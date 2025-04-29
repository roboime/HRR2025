import os
from setuptools import setup, find_packages

package_name = 'perception'

# Encontrar subpacotes automaticamente
packages = [package_name]
packages.extend([f"{package_name}.{subpackage}" for subpackage in find_packages(where=package_name)])

# Adicionar diretório de dados
data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    # Adicionar arquivos de configuração
    (os.path.join('share', package_name, 'config'), [os.path.join('config', f) for f in os.listdir('config') if os.path.isfile(os.path.join('config', f))]),
    # Adicionar arquivos de lançamento
    (os.path.join('share', package_name, 'launch'), [os.path.join('launch', f) for f in os.listdir('launch') if os.path.isfile(os.path.join('launch', f))]),
    # Adicionar recursos
    (os.path.join('share', package_name, 'resources'), [os.path.join('resources', f) for f in os.listdir('resources') if os.path.isfile(os.path.join('resources', f))]),
]

setup(
    name=package_name,
    version='1.1.0',
    packages=packages,
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='RoboIME',
    maintainer_email='keller.felipe@ime.eb.br',
    description='Sistema de percepção visual para robôs de futebol usando o modelo YOLOv4-Tiny',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_pipeline = perception.vision_pipeline:main',
            'ball_detector = perception.ball_detector:main',
            'field_detector = perception.field_detector:main',
            'line_detector = perception.line_detector:main',
            'goal_detector = perception.goal_detector:main',
            'obstacle_detector = perception.obstacle_detector:main',
            'yolo_detector = perception.yoeo_detector_node:main',
            'yolo_visualizer = perception.scripts.yoeo_visualizer_node:main',
        ],
    },
) 