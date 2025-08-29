from setuptools import setup, find_packages

package_name = 'roboime_behavior'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='RoboIME',
    maintainer_email='seu-email@example.com',
    description='Pacote de comportamento para o robô de futebol RoboIME',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'behavior_node = roboime_behavior.behavior_node:main',
        ],
    },
) 