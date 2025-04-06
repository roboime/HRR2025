#!/bin/bash

# Verifica se está sendo executado como root
if [ "$EUID" -ne 0 ]; then
  echo "Este script precisa ser executado como root (sudo)."
  exit 1
fi

echo "============================================================"
echo "🔧 Preparando ambiente Jetson para uso de câmera CSI"
echo "============================================================"

# Verifica e inicia serviço nvargus-daemon
if command -v systemctl &> /dev/null; then
  if ! systemctl is-active --quiet nvargus-daemon; then
    echo "Iniciando serviço nvargus-daemon..."
    systemctl start nvargus-daemon
    sleep 1
  else
    echo "✅ Serviço nvargus-daemon já está ativo."
  fi
fi

# Configura permissões para dispositivos de vídeo
if ls /dev/video* &> /dev/null; then
  echo "Configurando permissões para dispositivos de vídeo..."
  chmod 666 /dev/video*
  ls -la /dev/video*
else
  echo "⚠️ Nenhum dispositivo de vídeo encontrado!"
  echo "Verifique se a câmera está conectada corretamente."
fi

# Configura permissões para dispositivos NVIDIA
for device in /dev/nvhost* /dev/nvmap; do
  if [ -e "$device" ]; then
    echo "Configurando permissões para $device..."
    chmod 666 "$device"
  fi
done

# Verifica socket argus
ARGUS_SOCKET="/tmp/argus_socket"
if [ ! -d "$ARGUS_SOCKET" ]; then
  echo "Criando diretório para socket argus..."
  mkdir -p "$ARGUS_SOCKET"
fi
chmod 777 "$ARGUS_SOCKET"
echo "✅ Socket argus configurado: $(ls -ld $ARGUS_SOCKET)"

# Testa a câmera
echo "Testando câmera..."
if command -v nvgstcapture-1.0 &> /dev/null; then
  timeout 3s nvgstcapture-1.0 --sensor-id=0 --mode=2 --no-display &> /dev/null
  RESULT=$?
  if [ $RESULT -eq 0 ] || [ $RESULT -eq 124 ]; then
    echo "✅ Câmera CSI testada com sucesso!"
  else
    echo "⚠️ Teste de câmera falhou (código: $RESULT)"
  fi
else
  echo "⚠️ nvgstcapture-1.0 não encontrado, pulando teste da câmera."
fi

# Verifica se o ambiente é adequado para display X11
echo "Configurando ambiente X11..."
if [ -d "/tmp/.X11-unix" ]; then
  chmod 777 /tmp/.X11-unix
  chmod 777 /tmp/.X11-unix/X0 2>/dev/null || true
  echo "✅ Permissões de X11 configuradas."
else
  echo "⚠️ Diretório X11 não encontrado!"
fi

echo "============================================================"
echo "✅ Ambiente preparado! Você pode iniciar o container agora."
echo "============================================================"
echo ""
echo "Para iniciar o container, execute:"
echo "./scripts/docker-helpers/docker-run.sh"
echo ""
echo "Comando para testar a câmera dentro do container:"
echo "ros2 launch perception perception.launch.py camera_src:=csi camera_mode:=2 enable_display:=true" 