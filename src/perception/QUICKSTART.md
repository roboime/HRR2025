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
# Sistema completo (YOEO + tradicional)
ros2 launch perception perception.launch.py

# Apenas YOEO
ros2 launch perception perception.launch.py mode:=yoeo

# Apenas detectores tradicionais
ros2 launch perception perception.launch.py mode:=traditional
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

Para configurar quais detectores usar para cada objeto, edite o arquivo `config/perception_config.yaml`:

```yaml
pipeline:
  # Escolha de detector para cada tipo de objeto (opções: 'yoeo' ou 'traditional')
  detector_ball: "yoeo"         # Detector para a bola
  detector_field: "traditional" # Detector para o campo
  detector_lines: "traditional" # Detector para as linhas
  detector_goals: "yoeo"        # Detector para os gols
  detector_robots: "yoeo"       # Detector para os robôs
```

## Visualização

Para visualizar o resultado do sistema:

```bash
# Visualizar imagem de debug
ros2 run rqt_image_view rqt_image_view /vision/debug_image
```

## Exemplos de Uso Comum

1. **Usando com câmera USB em modo YOEO**:
   ```bash
   ros2 launch perception perception.launch.py mode:=yoeo camera_src:=usb
   ```

2. **Sistema completo com debug desabilitado** (para melhor desempenho):
   ```bash
   ros2 launch perception perception.launch.py debug:=false
   ```

3. **Detectores tradicionais com parâmetros específicos**:
   ```bash
   ros2 launch perception perception.launch.py mode:=traditional detector_ball:=traditional detector_field:=traditional
   ```

## Solução de Problemas

- **Erro na câmera**: Verifique se a câmera está conectada e o parâmetro `camera_src` está correto.
- **YOEO não funciona**: Verifique se o TensorFlow está instalado e o modelo está no caminho correto.
- **Desempenho lento**: Desabilite o debug com `debug:=false` para melhor desempenho.

Para mais informações, consulte o [README.md](README.md) completo. 