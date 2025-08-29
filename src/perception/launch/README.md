# 🚀 Sistema de Percepção - Arquivos de Lançamento

**YOLOv8 Unificado** - **Jetson Orin Nano Super** - **ROS2 Humble**

Sistema de percepção moderno com suporte dual para câmeras CSI IMX219 e USB Logitech C930, usando exclusivamente YOLOv8 para detecção de 6 classes simplificadas do futebol robótico.

---

## 📁 Estrutura de Launch Files

### **`perception.launch.py`** 🎯

**Arquivo principal** para execução do sistema de percepção YOLOv8 unificado.

#### **Características:**
- ✅ **YOLOv8 Simplificado**: Detecção de 6 classes otimizadas (2 estratégia + 4 localização)
- ✅ **Suporte Dual**: CSI IMX219 ou USB Logitech C930
- ✅ **Auto-seleção**: Câmera detectada automaticamente
- ✅ **GPU Otimizado**: Aceleração CUDA completa
- ✅ **Debug Integrado**: Visualização em tempo real

#### **Parâmetros Principais:**

##### **`camera_type`** 📷
- `csi` (padrão): Câmera CSI IMX219
  - 📏 **1280x720@30fps**
  - ⚡ **Baixíssima latência** (~20-30ms)
  - 🔧 **GStreamer nativo**
  - 🎯 **Ideal para competição**
  
- `usb`: Câmera USB Logitech C930
  - 📏 **1280x720@30fps**
  - 🔍 **Auto focus**
  - 📐 **Campo de visão amplo** (90°)
  - 🔧 **Plug & play**

##### **`model_path`** 🧠
- Caminho para modelo YOLOv8 customizado
- Padrão: `resources/models/robocup_yolov8.pt`
- Fallback: `yolov8n.pt` (se customizado não encontrado)

##### **`confidence_threshold`** 🎯
- Threshold de confiança para detecções
- Padrão: `0.6`
- Range: `0.0` (aceita tudo) a `1.0` (muito rigoroso)

##### **`debug`** 🐛
- `true` (padrão): Visualização com bounding boxes
- `false`: Modo produção (melhor performance)

##### **Parâmetros Avançados:**
- `device`: Dispositivo (`cuda`/`cpu`)
- `iou_threshold`: Threshold IoU para NMS (0.45)
- `max_detections`: Máximo detecções por frame (300)

### **`dual_camera.launch.py`** 🎥

**Sistema multi-câmera** para configurações avançadas com múltiplos pontos de vista.

#### **Características:**
- ✅ **Múltiplas Câmeras**: CSI + USB simultaneamente
- ✅ **Fusão de Dados**: Combina detecções de múltiplas fontes
- ✅ **Flexibilidade**: Escolha dinâmica de câmeras ativas
- ✅ **Load Balancing**: Distribuição de processamento

---

## 🚀 **Exemplos de Uso**

### **1. Lançamento Básico (CSI)**
```bash
# Sistema completo com câmera CSI padrão
ros2 launch perception perception.launch.py

# Equivalente explícito
ros2 launch perception perception.launch.py camera_type:=csi
```

### **2. Câmera USB**
```bash
# Sistema com câmera USB Logitech C930
ros2 launch perception perception.launch.py camera_type:=usb
```

### **3. Configuração de Performance**
```bash
# Máxima performance (sem debug)
ros2 launch perception perception.launch.py \
    debug:=false \
    confidence_threshold:=0.8 \
    iou_threshold:=0.45

# Performance balanceada
ros2 launch perception perception.launch.py \
    debug:=true \
    confidence_threshold:=0.6 \
    max_detections:=200
```

### **4. Modelo Customizado**
```bash
# Usar modelo treinado customizado
ros2 launch perception perception.launch.py \
    model_path:=/path/to/robocup_yolov8.pt \
    confidence_threshold:=0.7

# Modelo TensorRT otimizado
ros2 launch perception perception.launch.py \
    model_path:=resources/models/robocup_yolov8.engine \
    confidence_threshold:=0.8
```

### **5. Múltiplas Câmeras**
```bash
# Sistema dual camera
ros2 launch perception dual_camera.launch.py

# Com configurações específicas
ros2 launch perception dual_camera.launch.py \
    primary_camera:=csi \
    secondary_camera:=usb \
    model_path:=robocup_yolov8.pt
```

---

## 🎯 **Nós Lançados Automaticamente**

### **📷 Nós de Câmera** (dinâmico)
- **`csi_camera_node`** - Câmera CSI IMX219 (se selecionada)
- **`usb_camera_node`** - Câmera USB C930 (se selecionada)

### **🧠 Nó de Percepção Principal**
- **`yolov8_simplified_detector`** - Detector YOLOv8 com 6 classes otimizadas:
  
  **⚽ Estratégia de Jogo (2 classes):**
  - 🏐 **ball** (classe 0) - Bola de futebol
  - 🤖 **robot** (classe 1) - Robôs unificados (sem distinção de cor)
  
  **🧭 Localização no Campo (4 classes):**
  - 📍 **penalty_mark** (classe 2) - Marcas de penalty
  - 🥅 **goal_post** (classe 3) - Postes de gol (unificados)
  - ⭕ **center_circle** (classe 4) - Círculo central
  - 📐 **field_corner** (classe 5) - Cantos do campo
  - 🔲 **area_corner** (classe 6) - Cantos das áreas

---

## 📊 **Comparação de Câmeras**

| **Aspecto** | **CSI IMX219** | **USB C930** |
|-------------|----------------|--------------|
| **Resolução Padrão** | 1280x720 | 1280x720 |
| **FPS Máximo** | 30 | 30 |
| **Latência** | ~20-30ms | ~40-60ms |
| **Campo de Visão** | 78° | 90° |
| **Auto Focus** | ❌ Manual | ✅ Automático |
| **Estabilidade** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Uso CPU** | Baixo | Médio |
| **Integração** | Nativa Jetson | USB UVC |
| **Recomendação** | 🏆 **Competição** | 🔧 **Desenvolvimento** |

---

## ⚙️ **Configuração YAML**

### **Arquivo Principal** (`config/perception_config.yaml`)

```yaml
# Configuração de Pipeline
pipeline:
  processing_fps: 30.0
  debug_image: true             # Desabilitar em produção
  log_fps: true

# Modelo YOLOv8
yolov8:
  model_path: "resources/models/robocup_yolov8.pt"
  fallback_model: "yolov8n.pt"
  confidence_threshold: 0.6
  iou_threshold: 0.45
  max_detections: 300
  device: "cuda"
  half_precision: true

# Configurações de Câmera
camera:
  csi:
    width: 1280
    height: 720
    fps: 30
    flip_method: 2              # Rotação se necessário
    
  usb:
    width: 1280
    height: 720
    fps: 30
    device_id: 0
    auto_exposure: true
```

---

## 📡 **Tópicos ROS2 Publicados**

### **📤 Output Principal**
- `/perception/unified_detections` - Todas as detecções unificadas
- `/perception/debug_image` - Visualização com bounding boxes
- `/camera/image_raw` - Imagem da câmera

### **📤 Detecções Específicas**
- `/perception/ball_detection` - Posição da bola
- `/perception/robot_detections` - Robôs detectados (azuis/vermelhos)
- `/perception/goal_detections` - Postes de gol
- `/perception/field_detection` - Estruturas do campo
- `/perception/line_detection` - Linhas e landmarks

---

## 🔧 **Troubleshooting**

### **❌ Problemas Comuns**

#### **Câmera não detectada**
```bash
# Verificar dispositivos de vídeo
ls /dev/video*

# Testar câmera CSI
gst-launch-1.0 nvarguscamerasrc ! nvoverlaysink

# Testar câmera USB
v4l2-ctl --list-devices
```

#### **Modelo YOLOv8 não carrega**
```bash
# Verificar modelo existe
ls -la src/perception/resources/models/

# Usar modelo padrão temporariamente
ros2 launch perception perception.launch.py model_path:=yolov8n.pt
```

#### **FPS baixo**
```bash
# Reduzir resolução para performance
ros2 launch perception perception.launch.py \
    camera_width:=640 \
    camera_height:=480

# Desabilitar debug
ros2 launch perception perception.launch.py debug:=false
```
#### **CUDA não disponível**
```bash
# Verificar GPU
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"

# Usar CPU (última opção)
ros2 launch perception perception.launch.py device:=cpu
```

---

## 📈 **Monitoramento de Performance**

### **📊 Verificar FPS**
```bash
# FPS do sistema de detecção
ros2 topic hz /perception/unified_detections

# FPS da câmera
ros2 topic hz /camera/image_raw

# Latência end-to-end
ros2 topic delay /camera/image_raw /perception/debug_image
```

### **💻 Uso de Recursos**
```bash
# Monitoring geral
htop

# Monitoring específico Jetson
jtop

# GPU usage
nvidia-smi -l 1
```

---

## 🎮 **Desenvolvimento e Debug**

### **🔍 Visualização**
```bash
# Ver imagem com detecções
ros2 run rqt_image_view rqt_image_view /perception/debug_image

# Ver todas as detecções em texto
ros2 topic echo /perception/unified_detections

# Ver detecções específicas
ros2 topic echo /perception/ball_detection
```

### **🧪 Testes Interativos**
```bash
# Menu de testes
chmod +x src/perception/test_perception.sh
./src/perception/test_perception.sh

# Teste manual de componentes
ros2 run perception csi_camera_node
ros2 run perception yolov8_unified_detector
```

---

## 🚀 **Integração com Sistema Completo**

### **Integração com Bringup**
```bash
# Sistema completo do robô (incluindo percepção)
ros2 launch bringup robot.launch.py

# Sistema completo com configurações específicas
ros2 launch bringup robot.launch.py \
    camera_type:=csi \
    perception_debug:=true \
    model_path:=robocup_yolov8.pt
```

### **Comunicação com Outros Módulos**
- **Behavior**: Usa `/perception/ball_detection`, `/perception/goal_detections`
- **Navigation**: Usa `/perception/field_detection`, `/perception/line_detection`
- **Motion**: Recebe comandos baseados nas percepções

---

<div align="center">
  <p><strong>🚀 Sistema de Lançamento RoboIME HSL2025</strong></p>
  <p><em>YOLOv8 Unificado • Dual Camera • Jetson Optimized</em></p>
  <p>🏆 <em>Humanoid Soccer League 2025</em></p>
</div> 

