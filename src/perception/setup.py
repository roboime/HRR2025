from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'perception'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resources/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='RoboIME',
    maintainer_email='keller.felipe@ime.eb.br',
    description='Sistema de percepção visual para robôs de futebol usando o modelo YOEO',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yoeo_detector = perception.yoeo_detector_node:main',
            'yoeo_visualizer = perception.scripts.yoeo_visualizer_node:main',
        ],
    },
) 