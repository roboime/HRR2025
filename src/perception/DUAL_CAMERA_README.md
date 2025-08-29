# 🎥 Sistema Dual de Câmeras - Percepção RoboIME

Sistema YOLOv8 simplificado (6 classes) com suporte independente para **duas câmeras**:
- **CSI IMX219** (nativa Jetson)
- **USB Logitech C930** (externa)

## 📋 Visão Geral

O sistema permite escolher **dinamicamente** qual câmera usar, mantendo toda a funcionalidade do pipeline YOLOv8 simplificado (6 classes otimizadas).

### 🔧 Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    SISTEMA DUAL                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🎥 CÂMERA CSI                     🎥 CÂMERA USB           │
│  ┌─────────────────┐              ┌─────────────────┐       │
│  │   IMX219        │              │  Logitech C930  │       │
│  │   640x480@30fps │              │ 1280x720@30fps  │       │
│  │   GStreamer     │              │   V4L2/OpenCV   │       │
│  └─────────────────┘              └─────────────────┘       │
│           │                               │                 │
│           └─────────┬─────────────────────┘                 │
│                     ▼                                       │
│           ┌─────────────────────────┐                       │
│           │    PIPELINE YOLOV8     │                       │
│           │   6 Classes Otimizadas  │                       │
│           └─────────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Como Usar

### **Opção 1: CSI IMX219 (Padrão)**

```bash
# Configurar para CSI
sed -i 's/camera_type: ".*"/camera_type: "csi"/' src/perception/config/perception_config.yaml

# Iniciar sistema
./src/perception/scripts/start_perception_csi.sh
```

### **Opção 2: USB Logitech C930**

```bash
# Configurar para USB
sed -i 's/camera_type: ".*"/camera_type: "usb"/' src/perception/config/perception_config.yaml

# Testar câmera primeiro (recomendado)
./src/perception/scripts/test_c930_camera.sh

# Iniciar sistema
./src/perception/scripts/start_perception_usb.sh
```

### **Opção 3: Launch Manual**

```bash
# CSI
ros2 launch perception dual_camera_perception.launch.py camera_type:=csi

# USB
ros2 launch perception dual_camera_perception.launch.py camera_type:=usb
```

## 📖 Configurações Detalhadas

### 🔧 CSI IMX219 (Padrão)

```yaml
camera_type: "csi"

camera_csi:
  width: 640
  height: 480
  fps: 30
  camera_mode: 1          # 1920x1080@60fps nativo
  exposure_time: 10000
  gain: 1.2
  awb_mode: 1            # Auto white balance
  saturation: 1.3
```

**Características:**
- ✅ **Baixa Latência**: Interface CSI direta
- ✅ **Integração Nativa**: GStreamer + CUDA
- ✅ **Estável**: Sem dependência USB
- ⚠️ **Resolução**: Limitada para performance (640x480)

### 🔧 USB Logitech C930

```yaml
camera_type: "usb"

camera_usb:
  width: 1280            # C930 sweet spot
  height: 720
  fps: 30
  device_path: "/dev/video0"
  fourcc: "MJPG"         # Codec otimizado
  
  # Configurações avançadas C930
  auto_exposure: true
  brightness: 128
  contrast: 128
  saturation: 150        # Otimizado para detecção
  auto_focus: true       # C930 auto focus
  zoom: 100             # Sem zoom
```

**Características:**
- ✅ **Alta Resolução**: 1280x720 (melhor qualidade)
- ✅ **Auto Focus**: Foco automático inteligente
- ✅ **Flexibilidade**: Configurações avançadas
- ✅ **Campo de Visão**: 90° (vs 78° da CSI)
- ⚠️ **Latência**: Ligeiramente maior (USB)

## 🛠️ Instalação e Configuração

### **1. Preparar Sistema**

```bash
# Instalar v4l-utils (para USB)
sudo apt update
sudo apt install v4l-utils

# Verificar dispositivos disponíveis
ls -la /dev/video*
```

### **2. Verificar Câmeras**

```bash
# CSI IMX219
dmesg | grep -i imx219

# USB Logitech
lsusb | grep Logitech
v4l2-ctl --device=/dev/video0 --info
```

### **3. Construir Workspace**

```bash
# No root do workspace
colcon build --packages-select perception
source install/setup.bash
```

## 🧪 Testes e Diagnósticos

### **Teste Específico C930**

```bash
./src/perception/scripts/test_c930_camera.sh
```

Este script verifica:
1. 🔍 Detecção USB
2. 📹 Dispositivo de vídeo
3. 📋 Capacidades da câmera
4. 🐍 Teste OpenCV
5. 🤖 Teste nó ROS2

### **Teste Individual de Nós**

```bash
# Nó CSI
ros2 run perception jetson_camera --ros-args --params-file src/perception/config/perception_config.yaml

# Nó USB
ros2 run perception usb_camera_node --ros-args --params-file src/perception/config/perception_config.yaml
```

### **Monitoramento**

```bash
# Verificar tópicos
ros2 topic list | grep camera

# Monitorar FPS
ros2 topic hz /camera/image_raw

# Visualizar stream
ros2 run rqt_image_view rqt_image_view
```

## 🚨 Solução de Problemas

### **❌ CSI não detectada**

```bash
# Verificar cabo CSI
dmesg | tail -20

# Recarregar driver
sudo modprobe -r imx219
sudo modprobe imx219
```

### **❌ USB não funciona**

```bash
# Verificar detecção
lsusb | grep Logitech

# Testar dispositivos
ls -la /dev/video*

# Verificar permissões
sudo chmod 666 /dev/video0

# Testar diferentes dispositivos
v4l2-ctl --device=/dev/video1 --info  # Tente video1, video2, etc.
```

### **❌ Performance baixa**

```bash
# USB: Verificar USB 3.0
lsusb -t

# Verificar CPU/GPU
nvidia-smi
htop
```

### **❌ Problemas de sync**

```bash
# Limpar buffer USB
echo 1 > /sys/bus/usb/devices/usb1/authorized
echo 0 > /sys/bus/usb/devices/usb1/authorized
```

## 📊 Comparação de Performance

| Aspecto | CSI IMX219 | USB C930 |
|---------|------------|----------|
| **Resolução** | 640x480 | 1280x720 |
| **FPS Máximo** | 30 estável | 30 estável |
| **Latência** | ~50ms | ~80ms |
| **Campo de Visão** | 78° | 90° |
| **Auto Focus** | ❌ | ✅ |
| **CUDA Support** | ✅ Nativo | ✅ Via OpenCV |
| **Configurabilidade** | Limitada | Avançada |
| **Estabilidade** | Excelente | Boa |

## 🎯 Recomendações de Uso

### **Use CSI IMX219 quando:**
- Precisar de **baixa latência**
- Priorizar **estabilidade**
- Sistema **embarcado/móvel**
- Não precisar de alta resolução

### **Use USB C930 quando:**
- Precisar de **alta qualidade** de imagem
- Importante ter **auto focus**
- Necessário **campo de visão amplo**
- Desenvolvimento/**testes detalhados**

## 🔄 Migração Entre Câmeras

### **CSI → USB**

1. Editar config: `camera_type: "usb"`
2. Conectar C930
3. Testar: `./test_c930_camera.sh`
4. Iniciar: `./start_perception_usb.sh`

### **USB → CSI**

1. Editar config: `camera_type: "csi"`
2. Verificar cabo CSI
3. Iniciar: `./start_perception_csi.sh`

## 📝 Configuração Avançada

### **Otimização C930 para Robótica**

```yaml
camera_usb:
  # Resolução para performance
  width: 1280
  height: 720
  fps: 30
  
  # Configurações para campo
  auto_exposure: true
  saturation: 150          # Cores mais vivas
  sharpness: 140           # Bordas mais definidas
  contrast: 135            # Melhor contraste
  
  # Foco para robótica
  auto_focus: true         # Adapta automaticamente
  
  # Estabilidade
  power_line_frequency: 2  # 60Hz (evita flicker)
  backlight_compensation: 1 # Compensação de luz
```

### **Performance Máxima CSI**

```yaml
camera_csi:
  # Resolução balanceada
  width: 640
  height: 480
  fps: 30
  
  # Modo otimizado
  camera_mode: 1          # 1920x1080@60fps nativo
  exposure_time: 8000     # Exposição mais rápida
  gain: 1.0              # Ganho conservativo
  saturation: 1.4        # Cores destacadas
```

## 🎉 Conclusão

O sistema dual oferece **flexibilidade máxima** para diferentes cenários:

- **Competição**: CSI para estabilidade
- **Desenvolvimento**: USB para qualidade
- **Demonstrações**: USB para melhor visual
- **Deploy Final**: CSI para confiabilidade

**Escolha a câmera certa para sua necessidade e tenha o melhor dos dois mundos!** 🚀 