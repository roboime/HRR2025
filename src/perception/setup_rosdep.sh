#!/bin/bash
set -e

# Cores para formatação
VERDE='\033[0;32m'
VERMELHO='\033[0;31m'
AMARELO='\033[1;33m'
AZUL='\033[0;34m'
SEM_COR='\033[0m'

echo -e "${AZUL}=== Configurando rosdep para reconhecer dependências personalizadas ===${SEM_COR}"

# Verifica se está sendo executado como root
if [ "$EUID" -ne 0 ]; then
  echo -e "${VERMELHO}Este script precisa ser executado como root.${SEM_COR}"
  echo -e "Use: ${AMARELO}sudo ./setup_rosdep.sh${SEM_COR}"
  exit 1
fi

# Diretório principal de configuração do rosdep
ROSDEP_DIR="/etc/ros/rosdep/sources.list.d"

# Verifica e cria diretório de configuração do rosdep se não existir
mkdir -p "$ROSDEP_DIR"

# Verifica o diretório local do script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Caminho para o arquivo YAML de regras do rosdep
RULES_FILE="$SCRIPT_DIR/rosdep.yaml"

# Caminho para o arquivo de lista
LIST_FILE="$ROSDEP_DIR/60-roboime.list"

# Verifica se o arquivo YAML existe
if [ ! -f "$RULES_FILE" ]; then
  echo -e "${VERMELHO}Arquivo de regras não encontrado: $RULES_FILE${SEM_COR}"
  exit 1
fi

# Instala ou atualiza as regras do rosdep
echo -e "${AZUL}Adicionando arquivo de regras ao rosdep...${SEM_COR}"
echo "yaml file://$RULES_FILE" > "$LIST_FILE"
echo -e "${VERDE}✓${SEM_COR} Regras adicionadas: $LIST_FILE -> $RULES_FILE"

# Atualiza a base de dados do rosdep
echo -e "${AZUL}Atualizando a base de dados do rosdep...${SEM_COR}"
if rosdep update; then
  echo -e "${VERDE}✓${SEM_COR} Base de dados do rosdep atualizada com sucesso."
else
  echo -e "${VERMELHO}✗${SEM_COR} Erro ao atualizar a base de dados do rosdep."
  exit 1
fi

echo -e "\n${AZUL}Verificando resolução de dependências...${SEM_COR}"
cd /ros2_ws
if rosdep check --from-paths src --ignore-src; then
  echo -e "${VERDE}✓${SEM_COR} Todas as dependências podem ser resolvidas."
else
  echo -e "${AMARELO}!${SEM_COR} Algumas dependências ainda podem ter problemas."
  echo -e "Verifique o resultado acima para mais detalhes."
fi

echo -e "\n${AZUL}Configuração concluída!${SEM_COR}"
echo -e "Agora você pode executar: ${VERDE}rosdep install --from-paths src --ignore-src -r -y${SEM_COR}" 