# Sistema de Percepção - Arquivos de Lançamento

Este diretório contém os arquivos de lançamento para o sistema de percepção da RoboIME.

## Arquivos Simplificados

### perception.launch.py

Este é o arquivo de lançamento principal e unificado para todo o sistema de percepção. Ele oferece uma interface simples e flexível para iniciar o sistema com diferentes configurações.

#### Parâmetros Principais

- **mode**: Define o modo de operação do sistema
  - `unified` (padrão): Usa YOEO e detectores tradicionais juntos
  - `yoeo`: Usa apenas o detector YOEO
  - `traditional`: Usa apenas os detectores tradicionais

- **camera_src**: Define a fonte de imagens da câmera
  - `default` (padrão): Usa o tópico configurado em perception_config.yaml
  - `usb`: Configura para usar uma câmera USB
  - `csi`: Configura para usar uma câmera CSI (Jetson)
  - `simulation`: Usa imagens simuladas

- **debug**: Habilita ou desabilita a visualização de debug
  - `true` (padrão): Mostra imagens de debug
  - `false`: Desabilita visualização para melhor desempenho

- **config_file**: Caminho para o arquivo de configuração
  - Padrão: `perception_config.yaml`

#### Exemplos de Uso

```bash
# Iniciar o sistema completo com configurações padrão
ros2 launch perception perception.launch.py

# Iniciar apenas com detector YOEO
ros2 launch perception perception.launch.py mode:=yoeo

# Iniciar com detectores tradicionais e câmera USB
ros2 launch perception perception.launch.py mode:=traditional camera_src:=usb

# Iniciar sem visualização de debug (melhor desempenho)
ros2 launch perception perception.launch.py debug:=false
```

### jetson_camera.launch.py

Arquivo de lançamento específico para câmeras na plataforma Jetson.

#### Parâmetros

- **camera_type**: Tipo de câmera (`csi` ou `usb`)
- **camera_index**: Índice da câmera (geralmente 0)
- **width**: Largura da imagem
- **height**: Altura da imagem
- **fps**: Taxa de quadros

#### Exemplo de Uso

```bash
# Iniciar câmera USB da Jetson
ros2 launch perception jetson_camera.launch.py

# Iniciar câmera CSI na Jetson
ros2 launch perception jetson_camera.launch.py camera_type:=csi
```

## Fluxo de Dados

O sistema de percepção segue o seguinte fluxo:

1. A câmera publica imagens no tópico configurado (padrão: `/camera/image_raw`)
2. O nó `vision_pipeline` processa as imagens usando os detectores selecionados
3. Os resultados são publicados em tópicos específicos (posição da bola, máscara do campo, etc.)
4. Opcionalmente, as imagens de debug são publicadas para visualização

## Integração com Outros Sistemas

Para integrar o sistema de percepção com outros sistemas, configure os mesmos tópicos ROS em ambos os sistemas. 