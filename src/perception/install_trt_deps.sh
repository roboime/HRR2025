#!/bin/bash
# Script para instalar as dependências do TensorRT na Jetson

# Definir cores para saída
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===== Instalando dependências para TensorRT =====${NC}"

# Verificar se é um sistema Jetson
if [[ $(uname -m) != "aarch64" ]]; then
  echo -e "${RED}Este script deve ser executado em uma plataforma ARM64 (Jetson).${NC}"
  exit 1
fi

# Verificar se o TensorRT já está instalado
if python3 -c "import tensorrt" &> /dev/null; then
  echo -e "${GREEN}TensorRT já está instalado e configurado corretamente!${NC}"
else
  echo -e "${YELLOW}Configurando o módulo TensorRT Python...${NC}"
  
  # Encontrar o caminho para o TensorRT
  TRT_PATH=$(find /usr -name tensorrt -type d | grep -i python | head -n 1)
  
  if [ -z "$TRT_PATH" ]; then
    echo -e "${RED}Não foi possível encontrar a instalação do TensorRT.${NC}"
    echo -e "${YELLOW}Tentando instalar o TensorRT via apt...${NC}"
    
    sudo apt-get update
    sudo apt-get install -y python3-libnvinfer python3-libnvinfer-dev
    
    # Verificar novamente
    if python3 -c "import tensorrt" &> /dev/null; then
      echo -e "${GREEN}TensorRT instalado com sucesso via apt!${NC}"
    else
      echo -e "${RED}Falha ao instalar TensorRT via apt.${NC}"
      echo -e "${YELLOW}Tentando instalação via pip...${NC}"
      
      pip3 install tensorrt
      
      # Verificar uma última vez
      if python3 -c "import tensorrt" &> /dev/null; then
        echo -e "${GREEN}TensorRT instalado com sucesso via pip!${NC}"
      else
        echo -e "${RED}Falha ao instalar TensorRT. O pacote pode não estar disponível para esta versão da Jetson.${NC}"
        echo -e "${RED}Verifique a instalação manual em https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html${NC}"
      fi
    fi
  else
    echo -e "${GREEN}TensorRT encontrado em: $TRT_PATH${NC}"
    # Adicionar o diretório ao PYTHONPATH
    if ! grep -q "export PYTHONPATH=.*$TRT_PATH" ~/.bashrc; then
      echo "export PYTHONPATH=\$PYTHONPATH:$TRT_PATH" >> ~/.bashrc
      echo -e "${GREEN}TensorRT adicionado ao PYTHONPATH em ~/.bashrc${NC}"
    fi
    
    # Criar um link simbólico em site-packages
    SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
    if [ ! -f "$SITE_PACKAGES/tensorrt.pth" ]; then
      echo "$TRT_PATH" > "$SITE_PACKAGES/tensorrt.pth"
      echo -e "${GREEN}Link simbólico criado em $SITE_PACKAGES/tensorrt.pth${NC}"
    fi
    
    echo -e "${YELLOW}Atualizando PYTHONPATH atual...${NC}"
    export PYTHONPATH=$PYTHONPATH:$TRT_PATH
    
    # Testar a importação
    if python3 -c "import tensorrt; print(f'TensorRT versão: {tensorrt.__version__}')" &> /dev/null; then
      echo -e "${GREEN}TensorRT está configurado corretamente!${NC}"
    else
      echo -e "${RED}Ainda não é possível importar o módulo tensorrt. Tente reiniciar o terminal.${NC}"
    fi
  fi
fi

echo -e "${GREEN}===== Verificando pyCUDA =====${NC}"
if python3 -c "import pycuda.autoinit; print('pyCUDA funcionando corretamente!')" &> /dev/null; then
  echo -e "${GREEN}pyCUDA está funcionando corretamente!${NC}"
else
  echo -e "${YELLOW}Instalando pyCUDA...${NC}"
  pip3 install pycuda
fi

echo -e "${GREEN}===== Dependências instaladas com sucesso! =====${NC}"
echo -e "${YELLOW}Nota: Pode ser necessário reiniciar o terminal para as mudanças terem efeito.${NC}" 