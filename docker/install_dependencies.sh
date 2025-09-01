#!/bin/bash

# Verificar arquitetura do sistema
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "Aviso: este script é destinado a sistemas aarch64 (Jetson). Arquitetura detectada: $ARCH."
fi

# Atualizar repositórios e garantir pip3 instalado
if ! command -v pip3 &> /dev/null; then
    apt-get update && apt-get install -y python3-pip
fi

# Instalar jetson-stats (inclui jtop) via pip
pip3 install -U jetson-stats

# Instrução para usar jtop
echo "Ferramentas instaladas: jtop (via jetson-stats) e tegrastats."
echo "Use o comando 'jtop' para monitorar GPU/CPU, e 'tegrastats' para estatísticas do SoC."

# Garantir que usuário tenha acesso ao grupo 'video' (câmera)
groupadd -f video
usermod -aG video "$(id -un)"

echo "Instalação de dependências concluída. Reinicie o container ou faça logout/login no terminal."