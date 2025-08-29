# Guia Rápido - Sistema de Percepção da RoboIME

Este guia contém instruções rápidas para usar o sistema de percepção simplificado da RoboIME.

## Instalação Rápida

```bash
# 1. Clonar o repositório (se ainda não tiver feito)
cd ~/ros2_ws/src
git clone https://github.com/roboime/perception.git

# 2. Compilar
cd ~/ros2_ws
colcon build --packages-select perception
source install/setup.bash

# 3. Instalar TensorFlow (opcional, apenas para YOEO)
pip install tensorflow
```

## Comandos Básicos

### Iniciar o Sistema

```bash
# Sistema YOLOv8 simplificado (6 classes)
ros2 launch perception perception.launch.py

# Sistema YOLOv8 com câmera CSI
ros2 launch perception perception.launch.py camera_type:=csi

# Sistema YOLOv8 com câmera USB
ros2 launch perception perception.launch.py camera_type:=usb
```

### Configurar Câmera

```bash
# Câmera USB
ros2 launch perception perception.launch.py camera_src:=usb

# Câmera CSI (Jetson)
ros2 launch perception perception.launch.py camera_src:=csi
```

### Executar Testes

```bash
# Menu interativo de testes
./src/perception/test_perception.sh
```

## Configurações Comuns

Para configurar o sistema de percepção YOLOv8 simplificado, edite o arquivo `config/perception_config.yaml`:

```yaml
yolov8:
  # Sistema YOLOv8 simplificado (6 classes essenciais)
  model_path: "resources/models/robocup_simplified_yolov8.pt"
  confidence_threshold: 0.6     # Threshold de confiança
  iou_threshold: 0.45           # Non-maximum suppression
  max_detections: 200           # Máximo de detecções por frame
  device: "cuda"                # GPU (Orin Nano Super)
  half_precision: true          # FP16 para melhor performance
  
  # Classes detectadas (6 classes total)
  classes:
    # Estratégia de jogo (2 classes)
    ball: 0              # Bola de futebol
    robot: 1             # Robôs (sem distinção de cor)
    
    # Landmarks para localização (4 classes)
    penalty_mark: 2      # Marca do penalty
    goal_post: 3         # Postes de gol (unificados)
    center_circle: 4     # Círculo central
    field_corner: 5      # Cantos do campo
```

## Visualização

Para visualizar o resultado do sistema:

```bash
# Visualizar imagem de debug
ros2 run rqt_image_view rqt_image_view /vision/debug_image
```

## Exemplos de Uso Comum

1. **Sistema YOLOv8 com câmera USB**:
   ```bash
   ros2 launch perception perception.launch.py camera_type:=usb
   ```

2. **Sistema completo com debug desabilitado** (para melhor desempenho):
   ```bash
   ros2 launch perception perception.launch.py debug:=false
   ```

3. **YOLOv8 com modelo customizado e threshold específico**:
   ```bash
   ros2 launch perception perception.launch.py model_path:=robocup_simplified_yolov8.pt confidence_threshold:=0.7
   ```

## Solução de Problemas

- **Erro na câmera**: Verifique se a câmera está conectada e o parâmetro `camera_type` está correto (csi ou usb).
- **YOLOv8 não carrega**: Verifique se o modelo está no caminho correto e se CUDA está disponível.
- **Desempenho lento**: Desabilite o debug com `debug:=false` e use modelo TensorRT (.engine) para máxima performance.

Para mais informações, consulte o [README.md](README.md) completo.

## Treinamento do Modelo YOLOv4-Tiny

Para treinar o modelo YOLOv4-Tiny personalizado, siga os passos abaixo:

1. **Organize o Dataset**:
   ```bash
   cd src/perception
   mkdir -p resources/dataset/{train,val,test}
   ```
   Coloque as imagens e anotações no formato COCO em cada diretório.

2. **Configure os Parâmetros de Treinamento**:
   Edite o arquivo `config/training_config.yaml` conforme necessário.

3. **Execute o Treinamento**:
   ```bash
   python -m perception.yoeo.train_yoeo --config=config/training_config.yaml
   ```

4. **Otimize com TensorRT** (para melhor desempenho):
   ```bash
   python -m perception.yoeo.tensorrt_converter \
     --model_path=resources/models/yoeo_model.h5 \
     --output_dir=resources/models \
     --precision=FP16
   ```

Para mais detalhes, consulte o guia completo em `perception/yoeo/utils/README.md`. 