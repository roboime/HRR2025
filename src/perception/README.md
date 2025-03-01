# Sistema de Percepção da RoboIME

Este pacote contém o sistema de percepção visual da equipe RoboIME, implementando algoritmos de visão computacional para detectar elementos do jogo de futebol de robôs, como a bola, o campo, as linhas e os outros robôs.

![Percepção RoboIME](resources/images/perception_header.jpg)

## Estrutura Simplificada

O sistema de percepção da RoboIME foi projetado para ser modular, eficiente e fácil de entender. Ele consiste em:

1. **Detectores Tradicionais** - Algoritmos clássicos de visão computacional baseados em OpenCV
2. **Detector YOEO** - Detector baseado em aprendizado profundo ("You Only Encode Once")
3. **Pipeline Unificado** - Sistema que integra ambos os detectores

## Arquitetura

```
percepção/
├── config/
│   └── perception_config.yaml    # Configuração unificada do sistema
├── launch/
│   └── perception.launch.py      # Arquivo de lançamento simplificado
├── src/
│   ├── vision_pipeline.py        # Pipeline principal que integra os detectores
│   ├── ball_detector.py          # Detector tradicional de bola
│   ├── field_detector.py         # Detector tradicional de campo
│   ├── line_detector.py          # Detector tradicional de linhas
│   ├── goal_detector.py          # Detector tradicional de gols
│   ├── obstacle_detector.py      # Detector tradicional de obstáculos/robôs
│   └── yoeo/                     # Detector baseado em aprendizado profundo
│       ├── yoeo_detector.py      # Detector principal YOEO
│       ├── yoeo_model.py         # Modelo neural para detecção
│       └── components/           # Componentes individuais do YOEO
├── test_perception.sh            # Script para testar o sistema
└── README.md                     # Este arquivo
```

## Como Usar

### Instalação

1. Clone este repositório dentro do seu workspace ROS 2:
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws/src
   git clone https://github.com/roboime/perception.git
   ```

2. Instale as dependências:
   ```bash
   cd ~/ros2_ws
   rosdep update
   rosdep install --from-paths src --ignore-src -r -y
   ```

3. Compile o pacote:
   ```bash
   colcon build --packages-select perception
   ```

4. (Opcional) Se você quiser usar o detector YOEO, também precisará instalar o TensorFlow:
   ```bash
   pip install tensorflow
   ```

### Executando o Sistema

Você pode executar o sistema usando o arquivo de lançamento simplificado:

```bash
# Inicia o sistema completo (YOEO + detectores tradicionais)
ros2 launch perception perception.launch.py

# Ou você pode especificar o modo:
ros2 launch perception perception.launch.py mode:=yoeo         # Apenas YOEO
ros2 launch perception perception.launch.py mode:=traditional  # Apenas detectores tradicionais

# Ou você pode especificar a fonte da câmera:
ros2 launch perception perception.launch.py camera_src:=usb    # Câmera USB
ros2 launch perception perception.launch.py camera_src:=csi    # Câmera CSI (Jetson)
```

### Script de Teste

Para facilitar o teste do sistema, use o script de teste incluído:

```bash
cd ~/ros2_ws
./src/perception/test_perception.sh
```

Este script fornece um menu interativo para testar diferentes aspectos do sistema de percepção.

## Configuração

Toda a configuração do sistema agora está centralizada no arquivo `config/perception_config.yaml`, que está organizado em seções claras:

1. **pipeline** - Configurações gerais e escolha de detector para cada tipo de objeto
2. **camera** - Configurações da câmera
3. **yoeo** - Configurações do detector YOEO
4. **traditional** - Configurações dos detectores tradicionais

Exemplo de configuração:

```yaml
pipeline:
  processing_fps: 30.0
  detector_ball: "yoeo"         # Usar YOEO para detectar a bola
  detector_field: "traditional"  # Usar detector tradicional para o campo
```

## Detectores Disponíveis

### Detectores Tradicionais:

1. **Ball Detector** - Detecta a bola usando segmentação por cor e transformada de Hough
2. **Field Detector** - Segmenta o campo verde usando filtros de cor HSV
3. **Line Detector** - Detecta linhas usando transformada de Hough e detector de bordas Canny
4. **Goal Detector** - Identifica os gols usando segmentação por cor
5. **Obstacle Detector** - Detecta outros robôs no campo

### Detector YOEO:

O detector YOEO ("You Only Encode Once") é uma abordagem baseada em aprendizado profundo que combina:

1. **Detecção de Objetos** - Para bolas, gols e robôs
2. **Segmentação Semântica** - Para campo e linhas

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes. 