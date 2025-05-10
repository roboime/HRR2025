# Migração para YOLOv5 Original (Python 3.6.9 compatível)

## Visão Geral

Este documento descreve as modificações feitas para migrar de YOLOv5 via API Ultralytics para a implementação original do YOLOv5. Esta mudança foi necessária para garantir compatibilidade com Python 3.6.9 usado na Jetson Nano.

## Mudanças Principais

1. **YOEOModel (yoeo_model.py)**
   - Removidas dependências da API Ultralytics
   - Adicionado suporte direto ao YOLOv5 original
   - Implementação de carregamento de modelos .pt, .engine (TensorRT) e .onnx
   - Adicionado mecanismo para importar diretamente do código fonte do YOLOv5

2. **YOEOHandler (yoeo_handler.py)**
   - Implementação de pré-processamento e pós-processamento compatível com YOLOv5 original
   - Criação de método `detect()` para substituir API Ultralytics
   - Suporte a modelos TensorRT para alta performance

3. **YOEODetector (yoeo_detector.py)**
   - Modificados tópicos ROS para refletir a nova implementação
   - Adaptada lógica de processamento para a nova interface
   - Melhorado sistema de debug e visualização

4. **Instalação (setup_jetson.sh)**
   - Script atualizado para clonar o YOLOv5 original (versão v5.0)
   - Uso de Python 3.6.9 compatível com Jetson Nano
   - Download automático de pesos pré-treinados
   - Configuração de variáveis de ambiente

## Como Usar

### Instalação

Execute o script de instalação para configurar o ambiente:

```
sudo ./src/perception/setup_jetson.sh
```

Este script irá:
1. Instalar dependências necessárias
2. Clonar o YOLOv5 v5.0 (compatível com Python 3.6.9)
3. Baixar pesos pré-treinados
4. Configurar variáveis de ambiente

### Treinar um Modelo

Para treinar um modelo personalizado:

1. Prepare o dataset no formato YOLO (arquivos .txt com classes e coordenadas normalizadas)
2. Crie um arquivo `data.yaml` com a configuração do dataset

```yaml
# Exemplo data.yaml
path: /caminho/para/dados
train: images/train
val: images/val
test: images/test

# Classes
nc: 3  # Número de classes
names: ['ball', 'goal', 'robot']  # Nomes das classes
```

3. Execute o treinamento:

```python
from perception.yoeo.yoeo_model import train_model

# Treinar um modelo
model = train_model(
    data_yaml_path="/caminho/para/data.yaml", 
    epochs=100, 
    imgsz=640, 
    batch=16
)
```

### Inferência

Para usar o modelo treinado:

```python
from perception.yoeo.yoeo_model import YOEOModel
from perception.yoeo.yoeo_handler import YOEOHandler
import cv2

# Carregar o modelo
model_path = "/caminho/para/best.pt"  # ou .engine ou .onnx
handler = YOEOHandler({
    "model_path": model_path,
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45
})

# Inferência em uma imagem
image = cv2.imread("imagem.jpg")
detections = handler.detect(image)

# Acessar as detecções
for i, (box, conf, cls) in enumerate(zip(
        detections['boxes'], 
        detections['confidences'], 
        detections['classes'])):
    print(f"Detecção {i}: classe={cls}, confiança={conf:.2f}, bbox={box}")
```

## Exportação para TensorRT

Para otimizar a performance na Jetson, recomenda-se exportar o modelo para TensorRT:

```
cd $YOLOV5_PATH
python export.py --weights /caminho/para/best.pt --include engine --device 0
```

O modelo .engine resultante pode ser carregado diretamente:

```python
handler = YOEOHandler({
    "model_path": "/caminho/para/best.engine",
    "use_tensorrt": True
})
```

## Troubleshooting

Se encontrar erros relacionados ao caminho do YOLOv5, verifique se a variável de ambiente está configurada:

```bash
echo $YOLOV5_PATH
```

Se não estiver definida, adicione ao seu `.bashrc`:

```bash
export YOLOV5_PATH="/opt/yolov5"
```

## Referências

- [YOLOv5 Original Repository](https://github.com/ultralytics/yolov5/tree/v5.0)
- [TensorRT Optimization Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [Jetson Nano Documentation](https://developer.nvidia.com/embedded/jetson-nano-developer-kit) 