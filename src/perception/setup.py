from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'perception'

setup(
    name=package_name,
    version='3.0.0',  # Versão 3.0 - Sistema YOLOv8 + Geometria 3D Avançada
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Resource files (models, calibration, etc.)
        (os.path.join('share', package_name, 'resources'), glob('resources/**/*', recursive=True)),
        # Scripts
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.py')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.sh')),
    ],
    install_requires=[
        'setuptools',
        # ROS2 Humble core packages
        'rclpy>=3.3.0',
        'sensor_msgs',
        'geometry_msgs', 
        'cv_bridge',
        'std_msgs',
        
        # YOLOv8 and modern AI stack for Jetson Orin Nano Super
        'ultralytics>=8.0.0',      # YOLOv8 - detector unificado
        'torch>=2.1.0',            # PyTorch com CUDA support
        'torchvision>=0.16.0',     # TorchVision compatível
        'opencv-python>=4.8.0',    # OpenCV moderno
        
        # Scientific computing optimized for Python 3.10+
        'numpy>=1.24.0',           # NumPy moderno
        'pillow>=10.0.0',          # PIL/Pillow moderno
        'pyyaml>=6.0',             # YAML parser
        'scipy>=1.11.0',           # SciPy para cálculos 3D avançados
        
        # 3D Geometry and Advanced Perception
        'scikit-learn>=1.3.0',     # Machine learning para validação
        'matplotlib>=3.7.0',       # Visualização e debugging
        
        # Performance and utilities
        'psutil>=5.9.0',           # System monitoring
        'tqdm>=4.65.0',            # Progress bars (para treinamento)
    ],
    zip_safe=True,
    maintainer='RoboIME',
    maintainer_email='keller.felipe@ime.eb.br',
    description='Sistema de Percepção 3D Avançado RoboIME HSL2025 - YOLOv8 + Geometria 3D',
    long_description="""
    Sistema completo de percepção para futebol robótico humanóide com:
    
    🎯 YOLOv8 Simplificado (7 classes):
       - Estratégia: ball, robot  
       - Localização: penalty_mark, goal, center_circle, field_corner, area_corner
    
    📐 Geometria 3D Avançada:
       - Cálculo de posições reais usando calibração da câmera
       - Conversão pixel ↔ coordenadas mundo real
       - Validação de tamanhos baseada em objetos conhecidos
       - Correção de perspectiva e distorção
    
    🔄 Pipeline Avançado:
       - Tracking temporal de objetos
       - Predição de trajetórias (especialmente bola)
       - Análise de física (gravidade, resistência do ar)
       - Fusão temporal com filtros adaptativos
    
    ⚡ Otimizado para Jetson Orin Nano Super:
       - CUDA/TensorRT acceleration
       - FP16 precision para performance
       - Pipeline assíncrono
       - Baixa latência (~10-15ms/frame)
    """,
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Nós principais
            'yolov8_detector = perception.yolov8_detector_node:main',
            'csi_camera = perception.csi_camera_node:main',
            'usb_camera = perception.usb_camera_node:main',
        ],
    },
    # Classificadores para PyPI
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Framework :: Robot Framework :: Tool',
    ],
    python_requires='>=3.10',
    keywords=[
        'robotic-soccer', 'computer-vision', 'yolov8', '3d-geometry', 
        'object-detection', 'jetson', 'ros2', 'humanoid-robot',
        'real-time-tracking', 'camera-calibration', 'robocup'
    ],
) 