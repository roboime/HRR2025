from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'resource'), glob('resource/**/*', recursive=True)),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='RoboIME',
    maintainer_email='roboime@example.com',
    description='Pacote de percepção para o robô de futebol RoboIME',
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
            'yoeo_detector = perception.yoeo_detector_node:main',
        ],
    },
) 