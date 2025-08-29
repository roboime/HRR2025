# 🤖 RoboIME HSL2025 - Robô de Futebol

<div align="center">

![ROS2](https://img.shields.io/badge/ROS2-Eloquent-blue)
![Python](https://img.shields.io/badge/Python-3.6+-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4-orange)
![CUDA](https://img.shields.io/badge/CUDA-10.2-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

<img src="https://github.com/Kellerzinho/HSL2025/blob/master/RoboLogo.png" alt="RoboIME Logo" width="300"/>

**Sistema de robô de futebol humanóide baseado no trabalho do [Hamburg Bit-Bots](https://github.com/bit-bots/bitbots_main)**

</div>

## 📋 Índice

- [🔍 Visão Geral](#-visão-geral)
- [🧩 Estrutura do Projeto](#-estrutura-do-projeto)
- [💻 Requisitos do Sistema](#-requisitos-do-sistema)
- [🚀 Instalação e Uso](#-instalação-e-uso)
  - [🐳 Usando Docker](#-usando-docker-recomendado)
  - [🔧 Configurando o Ambiente](#-configurando-o-ambiente)
- [▶️ Executando o Robô](#️-executando-o-robô)
- [🔄 Desenvolvimento](#-desenvolvimento)
- [📡 Sincronização com o Jetson](#-sincronização-com-o-jetson)
- [🔍 Solução de Problemas](#-solução-de-problemas)
- [📜 Licença](#-licença)

## 🔍 Visão Geral

Este repositório contém o código para o robô de futebol humanóide da equipe RoboIME para a competição Humanoid Soccer League 2025. O sistema combina percepção visual avançada (utilizando o modelo YOEO), tomada de decisões e controle de movimentos para criar um jogador de futebol robótico autônomo.

## 🧩 Estrutura do Projeto

```
.
├── docker/                # Configurações do ambiente Docker
├── scripts/               # Scripts de automação e utilitários
├── src/                   # Código fonte
│   ├── behavior/          # Comportamento do robô e tomada de decisões
│   ├── bringup/           # Configurações de inicialização
│   ├── motion/            # Controle de movimento e cinemática
│   ├── perception/        # Visão computacional (YOEO)
│   ├── navigation/        # Navegação e localização
│   └── msgs/              # Mensagens ROS personalizadas
└── requirements.txt       # Dependências Python
```

## 💻 Requisitos do Sistema

Este projeto requer:

- **Hardware:**
  - NVIDIA Jetson Nano
  - Câmera compatível (CSI ou USB)
  - Servomotores para movimentação

- **Software:**
  - NVIDIA Jetpack 4.6
  - ROS 2 Eloquent
  - Python 3.6+
  - CUDA 10.2
  - TensorRT

## 🚀 Instalação e Uso

### 🐳 Usando Docker

Utilizar o ambiente Docker garante maior compatibilidade e facilidade de configuração:

1. **Clone o repositório:**

```bash
git clone https://github.com/Kellerzinho/HSL2025
cd HSL2025
```

2. **Construa a imagem Docker:**

```bash
docker build -t hsl:latest -f docker/Dockerfile.jetson .
```

3. **Execute o container:**

```bash
chmod +x ./scripts/docker-helpers/docker-run.sh

./scripts/docker-helpers/docker-run.sh
```

### 🔧 Configurando o Ambiente

Dentro do container, siga estas etapas para configurar corretamente o ambiente:

1. **Configurar bibliotecas Python para ROS:**

```bash
# Torne os scripts executáveis
chmod +x ./src/perception/setup_library_links.sh
chmod +x ./src/perception/setup_rosdep.sh

# Configure os links simbólicos para as bibliotecas Python
sudo ./src/perception/setup_library_links.sh

# Configure o rosdep para reconhecer dependências personalizadas
sudo ./src/perception/setup_rosdep.sh
```

4. **Dentro do container, instale as dependências e compile:**

```bash
# Instale as dependências Python (se necessário)
cd ..
/setup/install_dependencies.sh

# Instale as dependências do ROS
cd /ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# Compile os pacotes
colcon build --symlink-install

# Adicione os pacotes compilados ao ambiente
source install/setup.bash
```


## ▶️ Executando o Robô

### Sistema Completo

Para iniciar o sistema completo:

```bash
ros2 launch roboime_bringup robot.launch.py
```

### Somente Sistema de Percepção

Para testar apenas o sistema de percepção:

```bash
# Menu interativo de testes
./src/perception/test_perception.sh

# Ou diretamente:
ros2 launch perception perception.launch.py
```

## 🔄 Desenvolvimento

Para desenvolvimento contínuo, recomendamos:

1. Montar o diretório do projeto no container para facilitar a edição:
   ```bash
   ./scripts/docker-helpers/docker-run.sh
   ```

2. Em outro terminal, edite os arquivos normalmente com seu editor favorito

3. No container, recompile e teste:
   ```bash
   cd /ros2_ws
   colcon build --symlink-install --packages-select [pacote_modificado]
   ```

## 📡 Sincronização com o Jetson

Para sincronizar o código com a Jetson:

```bash
./scripts/sync_jetson.sh
```

Ou manualmente via `rsync`:

```bash
rsync -avz --exclude 'build/' --exclude 'install/' --exclude '.git/' ./ jetson@192.168.1.xxx:/home/jetson/roboime_ws/
```

## 🔍 Solução de Problemas

### Bibliotecas Python não encontradas

Se o TensorFlow ou OpenCV não forem encontrados pelo ROS:

```bash
# Execute o script de configuração de links
sudo ./src/perception/setup_library_links.sh
```

### Erros com dependências rosdep

Se o `rosdep` não conseguir resolver algumas dependências:

```bash
# Configure o rosdep personalizado
sudo ./src/perception/setup_rosdep.sh
```

### Modelo YOEO não encontrado

O modelo deve ser colocado em:

```
./src/perception/resources/models/yoeo_model.h5
```

Se não existir, o diretório será criado automaticamente pelo script `setup_library_links.sh`.

## 📜 Licença

Este projeto é licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

<div align="center">
  <p>Desenvolvido por <a href="https://github.com/Kellerzinho/HSL2025">RoboIME</a> - Instituto Militar de Engenharia (IME)</p>
</div>
