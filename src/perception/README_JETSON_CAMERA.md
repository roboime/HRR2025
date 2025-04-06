# Guia da Câmera CSI para Jetson no Docker

Este guia explica como configurar e utilizar corretamente a câmera CSI (IMX219) da Jetson em ambientes containerizados.

## Visão Geral

Nossa implementação foi melhorada para funcionar em containers Docker, mesmo quando o acesso nativo à câmera via NVIDIA Argus não está disponível. O sistema tenta vários métodos de inicialização em sequência:

1. Pipeline GStreamer completo com `nvarguscamerasrc`
2. Pipeline GStreamer simplificado
3. Acesso direto à câmera via OpenCV/V4L2
4. Câmera USB como último recurso

## Preparação do Host

Antes de iniciar o container, você deve preparar o ambiente host da Jetson:

```bash
# Tornar o script executável
chmod +x scripts/docker-helpers/prepare-jetson-camera.sh

# Executar como root
sudo ./scripts/docker-helpers/prepare-jetson-camera.sh
```

Este script:
- Verifica e inicia o serviço nvargus-daemon
- Configura permissões para dispositivos de vídeo e NVIDIA
- Prepara o socket Argus e configurações X11
- Testa a câmera CSI para verificar se está funcionando

## Iniciando o Container

Nosso script docker-run.sh já foi configurado para mapear os dispositivos necessários:

```bash
# Executar o container 
./scripts/docker-helpers/docker-run.sh
```

## Testando a Câmera no Container

Depois de entrar no container, você pode testar a câmera:

```bash
# Verificar se a câmera está visível
ls -l /dev/video*

# Lançar o nó da câmera
ros2 launch perception perception.launch.py camera_src:=csi camera_mode:=2 enable_display:=true
```

## Modos de Câmera Suportados

A câmera IMX219 suporta vários modos de operação:

| Modo | Resolução | FPS | Descrição |
|------|-----------|-----|-----------|
| 0 | 3280x2464 | 21 | Máxima resolução |
| 1 | 1920x1080 | 60 | Full HD |
| 2 | 1280x720 | 120 | HD Alto FPS (recomendado) |
| 3 | 1280x720 | 60 | HD |
| 4 | 1920x1080 | 30 | Full HD Baixo FPS |
| 5 | 1640x1232 | 30 | 4:3 |
| 6 | 1280x720 | 120 | IMX219 específico |

Exemplo de uso:
```bash
ros2 launch perception perception.launch.py camera_src:=csi camera_mode:=2 camera_fps:=120
```

## Visualização das Imagens

Para visualizar as imagens da câmera:

```bash
# Usando image_view (alternativa ao rqt_image_view)
ros2 run image_view image_view --ros-args -r image:=/camera/image_raw

# Ou habilitando display no próprio nó da câmera
ros2 launch perception perception.launch.py camera_src:=csi enable_display:=true
```

## Solução de Problemas Comuns

### 1. "nvarguscamerasrc não disponível"

Isso é normal no container. Nossa implementação agora detecta automaticamente esta situação e usa alternativas.

### 2. "Não foram encontrados dispositivos de vídeo"

O container não tem acesso ao dispositivo de vídeo. Verifique:
- Execute o script de preparação do host
- Verifique se o container está sendo iniciado com `--device /dev/video0`

### 3. Problemas com o visualizador rqt_image_view

Use a alternativa `image_view` ou ative a visualização integrada com `enable_display:=true`.

### 4. Baixo desempenho ou problemas na imagem

Ajuste os parâmetros da câmera:
```bash
ros2 launch perception perception.launch.py camera_src:=csi \
  exposure_time:=13333 \
  gain:=1.0 \
  camera_mode:=2 \
  enable_cuda:=true
```

## Recursos Extras

### Debug da Câmera

Para obter informações detalhadas sobre o que está acontecendo:

```bash
# Verificar tópicos da câmera
ros2 topic list | grep camera

# Verificar informações da câmera
ros2 topic echo /camera/camera_info

# Verificar mensagens de diagnóstico
ros2 topic echo /diagnostics
```

### Suporte para Outras Câmeras

O sistema também suporta câmeras USB como fallback:

```bash
ros2 launch perception perception.launch.py camera_src:=usb
``` 