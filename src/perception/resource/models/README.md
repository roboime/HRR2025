# Modelos Pré-treinados para o Sistema de Visão RoboIME

Este diretório contém modelos pré-treinados para o sistema de visão da RoboIME.

## Modelos YOEO

O modelo YOEO (You Only Encode Once) é usado para detecção de múltiplos objetos e segmentação semântica em tempo real, incluindo:

### Detecção de Objetos
- Bola
- Gols
- Robôs
- Árbitro

### Segmentação Semântica
- Campo
- Linhas
- Fundo

## Arquivos de Modelo

- `yoeo_model.h5`: Modelo Keras padrão
- `yoeo_model_tensorrt_fp16.trt`: Modelo otimizado com TensorRT em precisão FP16 (recomendado para Jetson Nano)
- `yoeo_model_tensorrt_fp32.trt`: Modelo otimizado com TensorRT em precisão FP32
- `yoeo_model_tensorrt_int8.trt`: Modelo otimizado com TensorRT em precisão INT8 (mais rápido, menos preciso)

## Arquitetura do Modelo

O modelo YOEO é baseado em uma arquitetura que combina:

1. **Backbone**: MobileNetV2 para extração de características eficiente
2. **Feature Pyramid Network (FPN)**: Para detecção em múltiplas escalas
3. **Cabeças de Detecção**: Para localizar e classificar objetos
4. **Cabeça de Segmentação**: Para segmentação semântica do campo e linhas

Esta arquitetura permite que um único modelo realize tanto detecção de objetos quanto segmentação semântica, economizando recursos computacionais.

## Como Treinar um Modelo

Para treinar um modelo YOEO com seus próprios dados:

```bash
python3 src/perception/src/yoeo/train_yoeo.py \
  --train_annotations=caminho/para/anotacoes.json \
  --train_images=caminho/para/imagens \
  --output_dir=src/perception/resource/models \
  --batch_size=8 \
  --epochs=100 \
  --learning_rate=0.001 \
  --input_height=416 \
  --input_width=416 \
  --enable_segmentation=true
```

## Como Converter para TensorRT

Para converter um modelo Keras para TensorRT (otimizado para Jetson Nano):

```bash
python3 src/perception/src/yoeo/tensorrt_converter.py \
  --model_path=src/perception/resource/models/yoeo_model.h5 \
  --output_dir=src/perception/resource/models \
  --precision=FP16
```

## Desempenho

Desempenho esperado na Jetson Nano:

| Modelo                    | Precisão | FPS  | mAP (Detecção) | mIoU (Segmentação) |
|---------------------------|----------|------|----------------|-------------------|
| yoeo_model.h5             | FP32     | ~5-7 | 0.85           | 0.78              |
| yoeo_model_tensorrt_fp32  | FP32     | ~8-10| 0.85           | 0.78              |
| yoeo_model_tensorrt_fp16  | FP16     | ~12-15| 0.83          | 0.76              |
| yoeo_model_tensorrt_int8  | INT8     | ~18-22| 0.78          | 0.72              |

## Formato de Saída do Modelo

O modelo YOEO produz duas saídas principais:

1. **Detecções**: Lista de objetos detectados com suas caixas delimitadoras, classes e confiança
2. **Segmentação**: Mapa de segmentação semântica onde cada pixel é classificado como fundo, linha ou campo

## Download de Modelos Pré-treinados

Os modelos pré-treinados podem ser baixados do seguinte link:

[Link para modelos pré-treinados](https://github.com/seu-usuario/RoboIME-HSL2025/releases/tag/models-v1.0)

Após o download, coloque os arquivos neste diretório. 