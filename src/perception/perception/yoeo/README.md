# Treinamento do Modelo YOLOv4-Tiny para Futebol de Robôs

Este documento explica como treinar o modelo YOLOv4-Tiny para detecção de objetos no contexto de futebol de robôs, com foco especificamente em detecção de **bola**, **gol** e **robô**.

## 1. Preparação do Dataset

### Estrutura de Pastas

O dataset deve estar organizado na seguinte estrutura:

```
src/perception/resources/dataset/
├── train/                     # Conjunto de treinamento
│   ├── imagens/*.jpg          # Imagens de treinamento
│   └── annotations.json       # Anotações no formato COCO
├── val/                       # Conjunto de validação
│   ├── imagens/*.jpg
│   └── annotations.json
└── test/                      # Conjunto de teste (opcional)
    ├── imagens/*.jpg
    └── annotations.json
```

### Formato das Anotações

As anotações devem estar no formato COCO JSON, contendo bounding boxes para as três classes de interesse:

```json
{
  "images": [
    {"id": 1, "file_name": "image1.jpg", "width": 1280, "height": 720}
  ],
  "annotations": [
    {
      "id": 1, 
      "image_id": 1, 
      "category_id": 1, 
      "bbox": [x, y, width, height], 
      "area": 1000, 
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "bola", "supercategory": "objeto"},
    {"id": 2, "name": "gol", "supercategory": "objeto"},
    {"id": 3, "name": "robo", "supercategory": "objeto"}
  ]
}
```

### Ferramentas de Anotação

Recomendamos o uso de:
- [Roboflow](https://roboflow.com/) - Plataforma completa para gerenciamento de datasets
- [CVAT](https://github.com/opencv/cvat) - Ferramenta de anotação open-source
- [LabelImg](https://github.com/tzutalin/labelImg) - Ferramenta leve para anotação de bounding boxes

## 2. Configuração do Treinamento

### Arquivo de Configuração

O treinamento é controlado pelo arquivo `src/perception/config/training_config.yaml`. Principais configurações:

```yaml
# Classes a serem detectadas
classes: ['bola', 'gol', 'robo']

# Diretórios de dados
train_dir: "resources/dataset/train"
val_dir: "resources/dataset/val"

# Parâmetros de treinamento
input_width: 416
input_height: 416
batch_size: 8
epochs: 100
learning_rate: 0.001

# Pesos das perdas (foco maior na detecção)
detection_loss_weight: 1.0
segmentation_loss_weight: 0.1  # Valor baixo, foco apenas na detecção

# Diretórios de saída
checkpoint_dir: "resources/models/checkpoints"
output_dir: "resources/models"
```

## 3. Executando o Treinamento

### Requisitos

Certifique-se de ter instalado:
- TensorFlow 2.x
- OpenCV
- PyYAML
- tqdm
- requests

### Comando para Treinamento

```bash
cd src/perception
python perception/yoeo/train_yoeo.py --config config/training_config.yaml --download_pretrained
```

Parâmetros disponíveis:
- `--config`: Caminho para o arquivo de configuração
- `--download_pretrained`: Baixa e converte pesos pré-treinados do YOLOv4-Tiny
- `--gpu_id`: ID da GPU a ser utilizada (se houver múltiplas GPUs)

### Monitoramento do Treinamento

O progresso do treinamento pode ser monitorado via:
- TensorBoard: `tensorboard --logdir=resources/models/logs`
- Logs no console: O script mostra progresso por época e métricas
- Arquivos CSV: Métricas detalhadas são salvas em `resources/models/logs`

## 4. Conversão para TensorRT

Após o treinamento, o modelo pode ser otimizado para inferência rápida com TensorRT:

```bash
python perception/yoeo/tensorrt_converter.py \
    --model_path resources/models/yolov4_tiny_full.h5 \
    --output_dir resources/models \
    --precision FP16
```

### Opções do Conversor TensorRT

- `--model_path`: Caminho para o modelo treinado em formato H5
- `--output_dir`: Diretório onde o modelo otimizado será salvo
- `--precision`: Formato de precisão (FP32, FP16, INT8)
- `--input_height` e `--input_width`: Dimensões da entrada (padrão: 416x416)

## 5. Usando o Modelo Treinado

O modelo treinado pode ser usado diretamente pelo sistema de percepção. Edite o arquivo de configuração `perception_config.yaml`:

```yaml
yoeo:
  model_path: "resources/models/yolov4_tiny_full.h5"  # ou o modelo TensorRT
  input_width: 416
  input_height: 416
  confidence_threshold: 0.5
  iou_threshold: 0.45
  use_tensorrt: true  # Se estiver usando um modelo TensorRT
  components:
    ball: true
    goals: true
    robots: true
```

## 6. Dicas para Melhores Resultados

1. **Qualidade do Dataset**:
   - Use imagens variadas com diferentes condições de iluminação
   - Inclua imagens de diferentes perspectivas
   - Balance as classes (similar número de exemplos por classe)

2. **Ajustes de Hiperparâmetros**:
   - Aumente o `batch_size` se tiver memória GPU suficiente
   - Reduza o `learning_rate` se o treinamento estiver instável
   - Aumente `epochs` para datasets maiores

3. **Data Augmentation**:
   - Use as opções de augmentação para aumentar artificialmente o tamanho do dataset
   - Ajuste `rotation_range`, `brightness_range` e outras opções para seu caso específico

4. **Transferência de Aprendizado**:
   - Sempre use `--download_pretrained` para iniciar com pesos pré-treinados
   - Para datasets pequenos, considere congelar as camadas iniciais do modelo

5. **Avaliação**:
   - Verifique a qualidade das detecções em imagens de teste
   - Ajuste os limiares de confiança (`confidence_threshold`) conforme necessário
   - Monitore as métricas de validação para evitar overfitting 