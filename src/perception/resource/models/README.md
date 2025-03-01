# Modelos Pré-treinados para o Sistema de Visão RoboIME

Este diretório contém modelos pré-treinados para o sistema de visão da RoboIME.

## Modelos YOEO

O modelo YOEO (You Only Encode Once) é usado para detecção de múltiplos objetos em tempo real, incluindo:
- Bola
- Gols
- Robôs
- Árbitro

### Arquivos de Modelo

- `yoeo_model.h5`: Modelo Keras padrão
- `yoeo_model_tensorrt_fp16.trt`: Modelo otimizado com TensorRT em precisão FP16 (recomendado para Jetson Nano)
- `yoeo_model_tensorrt_fp32.trt`: Modelo otimizado com TensorRT em precisão FP32
- `yoeo_model_tensorrt_int8.trt`: Modelo otimizado com TensorRT em precisão INT8 (mais rápido, menos preciso)

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
  --input_width=416
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

| Modelo                    | Precisão | FPS  | Precisão mAP |
|---------------------------|----------|------|--------------|
| yoeo_model.h5             | FP32     | ~5-7 | 0.85         |
| yoeo_model_tensorrt_fp32  | FP32     | ~8-10| 0.85         |
| yoeo_model_tensorrt_fp16  | FP16     | ~12-15| 0.83        |
| yoeo_model_tensorrt_int8  | INT8     | ~18-22| 0.78        |

## Download de Modelos Pré-treinados

Os modelos pré-treinados podem ser baixados do seguinte link:

[Link para modelos pré-treinados](https://github.com/seu-usuario/RoboIME-HSL2025/releases/tag/models-v1.0)

Após o download, coloque os arquivos neste diretório. 