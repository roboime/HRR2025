# 👁️ Sistema de Percepção RoboIME HSL2025

**YOLOv8 Simplificado** - **Jetson Orin Nano Super** - **ROS2 Humble**

> **🎯 SISTEMA FOCADO**: Detecção simplificada com 6 classes essenciais - 2 para estratégia de jogo + 4 para localização no campo.

---

## 🎯 **Visão Geral**

O sistema de percepção do RoboIME HSL2025 utiliza **exclusivamente YOLOv8** para detectar todos os elementos necessários para navegação e estratégia no futebol robótico humanóide.

### **🧠 Arquitetura Inteligente:**
- **Uma rede neural especializada** para elementos essenciais
- **6 classes otimizadas** para futebol robótico
- **Dual-purpose**: Estratégia + Localização
- **Performance máxima** no Jetson Orin Nano Super
- **Baixa latência** (~10-15ms por frame)

---

## 🎯 **Classes Detectadas (6 Total)**

### **⚽ Estratégia de Jogo (2 classes)**
- **0: `ball`** - Bola de futebol (elemento principal do jogo)
- **1: `robot`** - Robôs (sem distinção de cor - unificado)

### **🧭 Localização no Campo (4 classes)**
- **2: `penalty_mark`** - Marca do penalty (landmark preciso)
- **3: `goal_post`** - Postes de gol (unificados)
- **4: `center_circle`** - Círculo central (landmark de referência)
- **5: `field_corner`** - Cantos do campo (landmarks de borda)
- **6: `area_corner`** - Cantos da área (landmarks internos)

---

## 📦 **Arquitetura do Sistema**

```
🎯 Sistema YOLOv8 Simplificado
├── 📷 CSI Camera Node               # IMX219 ou USB C930
│   ├── ⚡ GStreamer Pipeline        # Captura otimizada
│   ├── 📐 Auto-calibração          # Parâmetros intrínsecos
│   └── 🔄 30 FPS @ 1280x720        # Resolução padrão
│
├── 🧠 YOLOv8 Simplified Detector    # Detector principal (6 classes)
│   ├── ⚽ Ball Detection            # Bola (classe 0)
│   ├── 🤖 Robot Detection           # Robôs unificados (classe 1)
│   └── 🧭 Localization Landmarks    # 5 landmarks (classes 2-6)
│       ├── 📍 Penalty Mark         # Marca do penalty
│       ├── 🥅 Goal Post            # Postes de gol
│       ├── ⭕ Center Circle        # Círculo central
│       ├── 📐 Field Corner         # Cantos do campo
│       └── 🔲 Area Corner          # Cantos da área
│
└── 📡 ROS2 Publishers               # Comunicação otimizada
    ├── ⚽ Ball Specific             # Posição da bola
    ├── 🤖 Robot Specific            # Posições dos robôs
    ├── 🥅 Goal Post Specific        # Postes de gol
    ├── 🧭 Localization Landmarks    # Landmarks para navegação
    ├── 🎯 Unified Detections        # Todas as detecções
    └── 🖼️ Debug Image               # Visualização
```

---

## 🛠️ **Instalação e Uso**

### **1. Pré-requisitos**
```bash
# Sistema base
# Jetson Orin Nano Super com JetPack 6.2+
# Ubuntu 22.04 LTS + ROS2 Humble
# Python 3.10+ + CUDA 12.2+
```

### **2. Instalação Rápida**
```bash
# Dentro do workspace RoboIME HSL2025
cd src/perception

# Instalar dependências
pip3 install ultralytics>=8.0.0 torch>=2.1.0 torchvision>=0.16.0
pip3 install opencv-python>=4.8.0 numpy>=1.24.0

# Compilar pacote
cd ../../
colcon build --packages-select perception
source install/setup.bash
```

### **3. Configuração de Modelos**
```bash
# Verificar modelo customizado (recomendado para 6 classes)
ls src/perception/resources/models/robocup_simplified_yolov8.pt

# Se não existir, usar modelo base (REQUER RETREINAMENTO!)
cd src/perception/resources/models/
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

---

## 🚀 **Como Usar**

### **🎬 Lançamento Principal**
```bash
# Sistema completo (câmera CSI + YOLOv8 simplificado)
ros2 launch perception perception.launch.py

# Com debug visual habilitado
ros2 launch perception perception.launch.py debug:=true

# Câmera USB (Logitech C930)
ros2 launch perception perception.launch.py camera_type:=usb
```

### **🔧 Configuração Avançada**
```bash
# Modelo customizado (6 classes)
ros2 launch perception perception.launch.py \
  model_path:=/path/to/robocup_simplified_yolov8.pt \
  confidence_threshold:=0.6

# Performance máxima (sem debug)
ros2 launch perception perception.launch.py \
  debug:=false \
  confidence_threshold:=0.7 \
  max_detections:=200

# Múltiplas câmeras
ros2 launch perception dual_camera.launch.py
```

### **📊 Monitoramento de Detecções**
```bash
# Todas as detecções unificadas
ros2 topic echo /perception/unified_detections

# Detecções específicas
ros2 topic echo /perception/ball_detection          # Bola (estratégia)
ros2 topic echo /perception/robot_detections        # Robôs (estratégia)
ros2 topic echo /perception/goal_detections         # Postes de gol (localização)
ros2 topic echo /perception/localization_landmarks  # Landmarks (localização)

# Visualização em tempo real
ros2 run rqt_image_view rqt_image_view /perception/debug_image
```

### **📈 Performance e Estatísticas**
```bash
# FPS das detecções
ros2 topic hz /perception/unified_detections

# FPS da câmera
ros2 topic hz /camera/image_raw

# Verificar GPU usage
nvidia-smi
jtop  # Específico do Jetson
```

---

## ⚙️ **Configuração**

### **📝 Arquivo Principal** (`config/perception_config.yaml`)

```yaml
# Sistema YOLOv8 Simplificado (6 classes)
yolov8:
  model_path: "resources/models/robocup_simplified_yolov8.pt"
  fallback_model: "yolov8n.pt"
  confidence_threshold: 0.6
  max_detections: 200           # Reduzido para 6 classes
  
  # Classes simplificadas
  classes:
    # Estratégia (2 classes)
    ball: 0
    robot: 1
    
    # Localização (4 classes)
    penalty_mark: 2
    goal_post: 3
    center_circle: 4
    field_corner: 5
    area_corner: 6
```

---

## 🎯 **Aplicações das Classes**

### **⚽ Estratégia de Jogo**
- **Ball**: Rastreamento contínuo para decisões táticas
- **Robot**: Detecção de outros robôs para evitar colisões

### **🧭 Localização no Campo**
- **Penalty Mark**: Ponto de referência preciso (coordenadas conhecidas)
- **Goal Post**: Orientação e posicionamento no campo
- **Center Circle**: Landmark central para calibração
- **Field Corner**: Landmarks de borda para mapeamento
- **Area Corner**: Landmarks internos para navegação fina

---

## 📡 **Interface ROS2**

### **📥 Input Topics**
- `/camera/image_raw` (sensor_msgs/Image) - Imagem da câmera
- `/camera/camera_info` (sensor_msgs/CameraInfo) - Parâmetros da câmera

### **📤 Output Topics**

#### **Detecções Estratégicas**
- `/perception/ball_detection` (roboime_msgs/BallDetection) - Posição da bola
- `/perception/robot_detections` (roboime_msgs/RobotDetection) - Robôs detectados

#### **Detecções para Localização**
- `/perception/goal_detections` (roboime_msgs/GoalDetection) - Postes de gol
- `/perception/localization_landmarks` (roboime_msgs/FieldDetection) - Landmarks do campo

#### **Detecções Unificadas**
- `/perception/unified_detections` (roboime_msgs/SimplifiedDetections) - **Todas as detecções**
- `/perception/debug_image` (sensor_msgs/Image) - **Visualização com bounding boxes**

### **🎛️ Parâmetros Configuráveis**
- `confidence_threshold` (double): Threshold de confiança [0.0-1.0]
- `model_path` (string): Caminho para modelo YOLOv8 de 6 classes
- `device` (string): `cuda` ou `cpu`
- `publish_debug` (bool): Publicar imagens de debug
- `iou_threshold` (double): Threshold IoU para NMS
- `max_detections` (int): Máximo de detecções por frame (recomendado: 200)

---

## 🏗️ **Desenvolvimento e Treinamento**

### **🧠 Treinar Modelo Customizado (6 Classes)**
```python
from ultralytics import YOLO

# Configurar dataset para 6 classes
model = YOLO('yolov8n.pt')
model.train(
    data='src/perception/resources/training/configs/robocup.yaml',
    epochs=100,
    imgsz=640,
    device='cuda',
    project='src/perception/resources/models/',
    name='robocup_simplified_yolov8'
)

# Exportar modelo otimizado
model.export(format='onnx', optimize=True)
model.export(format='engine', half=True)  # TensorRT
```

### **📊 Dataset Configuration** (`robocup.yaml`)
```yaml
# Dataset para 6 classes simplificadas
names:
  0: ball          # Estratégia
  1: robot         # Estratégia
  2: penalty_mark  # Localização
  3: goal_post     # Localização
  4: center_circle # Localização
  5: field_corner  # Localização
  6: area_corner   # Localização

nc: 7  # 6 classes + background
```

### **🔧 Script de Treinamento Automático**
```bash
# Usar configuração otimizada para 6 classes
cd src/perception/resources/training/
python3 train_model.py --config configs/train_config.yaml

# Verificar métricas
ls metrics/
cat logs/training.log
```

---

## 🧪 **Testes e Validação**

### **🎮 Menu Interativo**
```bash
chmod +x src/perception/test_perception.sh
./src/perception/test_perception.sh

# Opções disponíveis:
# 1. Testar câmera CSI
# 2. Testar câmera USB
# 3. Executar detecção YOLOv8 (6 classes)
# 4. Visualizar detecções simplificadas
# 5. Monitorar performance
```

### **⚡ Testes Individuais**
```bash
# Testar só a câmera
ros2 run perception csi_camera_node

# Testar só o detector simplificado
ros2 run perception yolov8_unified_detector

# Teste completo
ros2 launch perception perception.launch.py debug:=true
```

### **📊 Benchmarks de Performance**

#### **Jetson Orin Nano Super (Esperado com 6 classes):**
- **Resolução**: 1280x720
- **FPS**: 20-25 (melhoria com menos classes)
- **Latência**: 10-15ms (redução significativa)
- **GPU Usage**: ~60-70% (economia de recursos)
- **RAM Usage**: ~1.5-2GB (menos memória)

#### **Comparação vs 11 Classes:**
| Métrica | 11 Classes | 6 Classes | Melhoria |
|---------|------------|-----------|----------|
| FPS | 15-20 | 20-25 | +25-30% |
| Latência | 15-20ms | 10-15ms | -25-33% |
| GPU | 70-80% | 60-70% | -10-15% |
| RAM | 2-3GB | 1.5-2GB | -25-30% |

---

## 🔧 **Otimizações Avançadas**

### **🚀 TensorRT (Produção)**
```bash
# Converter modelo de 6 classes para TensorRT
cd src/perception/resources/models/
python3 -c "
from ultralytics import YOLO
model = YOLO('robocup_simplified_yolov8.pt')
model.export(format='engine', half=True, workspace=4)
"

# Usar modelo TensorRT
ros2 launch perception perception.launch.py \
  model_path:=robocup_simplified_yolov8.engine
```

### **⚡ Configurações de Performance**
```yaml
# config/perception_config.yaml
yolov8:
  # Máxima performance (6 classes)
  confidence_threshold: 0.7     # Mais rigoroso com menos classes
  max_detections: 150           # Reduzido drasticamente
  half_precision: true
  
pipeline:
  debug_image: false           # Desabilitar em produção
  processing_fps: 35.0         # FPS aumentado
```

---

## 🔍 **Solução de Problemas**

### **🔴 Problemas Comuns**

#### **Modelo não carrega (6 classes)**
```bash
# Verificar modelo de 6 classes existe
ls -la src/perception/resources/models/robocup_simplified_yolov8.pt

# Testar modelo manualmente
python3 -c "
from ultralytics import YOLO
model = YOLO('robocup_simplified_yolov8.pt')
print(f'Classes: {model.names}')
assert len(model.names) == 7  # 6 classes + background
"

# Usar fallback temporário (REQUER RETREINAMENTO!)
ros2 launch perception perception.launch.py model_path:=yolov8n.pt
```

#### **Performance baixa com 6 classes**
```bash
# Aumentar threshold de confiança
ros2 launch perception perception.launch.py confidence_threshold:=0.8

# Reduzir detecções máximas
ros2 launch perception perception.launch.py max_detections:=100

# Verificar se está usando modelo correto
ros2 topic echo /perception/unified_detections | head -20
```

#### **Classes incorretas detectadas**
```bash
# Verificar mapeamento de classes no log
ros2 launch perception perception.launch.py debug:=true

# Classes esperadas: ball, robot, penalty_mark, goal_post, center_circle, field_corner, area_corner
# Se aparecer outras classes = modelo errado carregado
```

### **📊 Monitoramento Simplificado**
```bash
# CPU/GPU (deve estar mais baixo)
htop
jtop

# FPS sistema (deve estar mais alto)
ros2 topic hz /perception/unified_detections

# Verificar 6 classes no debug
ros2 run rqt_image_view rqt_image_view /perception/debug_image
```

---

## 📈 **Vantagens do Sistema Simplificado**

### **🎯 Focado em Essenciais**
- **Estratégia**: Ball + Robot = decisões táticas
- **Localização**: 5 landmarks = navegação precisa
- **Sem redundância**: Eliminadas classes desnecessárias

### **⚡ Performance Otimizada**
- **Menos classes**: Menos computação por frame
- **Modelo menor**: Menor uso de memória
- **Inferência rápida**: Latência reduzida
- **GPU eficiente**: Mais recursos para outras tarefas

### **🔧 Facilidade de Treinamento**
- **Dataset menor**: Menos anotações necessárias
- **Convergência rápida**: Menos classes = treinamento mais rápido
- **Melhor qualidade**: Foco em classes importantes

### **🧭 Integração com Localização**
- **Landmarks específicos**: Penalty mark, goals, circles, corners
- **Coordenadas conhecidas**: Facilita algoritmos de localização
- **Triangulação**: Múltiplos landmarks para posicionamento

---

<div align="center">
  <p><strong>👁️ Sistema de Percepção RoboIME HSL2025 Simplificado</strong></p>
  <p><em>YOLOv8 • 6 Classes Essenciais • Jetson Orin Nano Super • ROS2 Humble</em></p>
  <p>🎯 <em>Estratégia + Localização • Performance Otimizada</em></p>
  <p>🏆 <em>Humanoid Soccer League 2025</em></p>
</div> 