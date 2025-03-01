# Recursos para o Sistema de Percepção

Este diretório contém recursos necessários para o sistema de percepção baseado no modelo YOEO, incluindo modelos pré-treinados, arquivos de calibração de câmera, e outros recursos estáticos.

## Estrutura de Diretórios

```
resources/
├── models/              # Modelos pré-treinados
│   ├── yoeo_model.h5    # Modelo YOEO em formato HDF5
│   ├── yoeo_model.trt   # Modelo YOEO otimizado para TensorRT
│   └── ...
├── calibration/         # Arquivos de calibração de câmera
│   ├── camera_info.yaml # Parâmetros intrínsecos da câmera
│   └── ...
├── test_images/         # Imagens de teste para validação
│   ├── field01.jpg
│   ├── field02.jpg
│   └── ...
└── labels/              # Arquivos de rótulos para classes
    ├── detection_classes.txt
    └── segmentation_classes.txt
```

## Modelos Pré-treinados

Os modelos pré-treinados estão disponíveis no diretório `models/`. Estes modelos foram treinados com o conjunto de dados da RoboCup e estão prontos para uso.

Para usar um modelo pré-treinado, especifique o caminho para o modelo no arquivo de configuração ou como parâmetro ao iniciar o nó detector:

```bash
ros2 run perception yoeo_detector_node --ros-args -p model_path:=resources/models/yoeo_model.h5
```

## Arquivos de Calibração

Os arquivos de calibração de câmera estão disponíveis no diretório `calibration/`. Estes arquivos contêm os parâmetros intrínsecos e extrínsecos da câmera, necessários para a estimativa precisa de posição 3D.

Para usar um arquivo de calibração, especifique o caminho para o arquivo no arquivo de configuração ou como parâmetro ao iniciar o nó detector:

```bash
ros2 run perception yoeo_detector_node --ros-args -p camera_calibration:=resources/calibration/camera_info.yaml
```

## Imagens de Teste

O diretório `test_images/` contém imagens de teste que podem ser usadas para validar o funcionamento do sistema de percepção. Estas imagens foram selecionadas para cobrir diferentes condições de iluminação e cenários de jogo.

## Arquivos de Rótulos

O diretório `labels/` contém arquivos de texto com os nomes das classes para detecção e segmentação. Estes arquivos são usados pelo sistema para mapear os índices de classe para nomes legíveis por humanos.

- `detection_classes.txt`: Lista de classes para detecção de objetos (bola, gol, robô, árbitro)
- `segmentation_classes.txt`: Lista de classes para segmentação semântica (fundo, campo, linha) 