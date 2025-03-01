# RoboIME HSL2025 - Robô de Futebol

Este repositório contém o código para o robô de futebol da equipe RoboIME para a competição Humanoid Soccer League 2025, baseado no trabalho do [Hamburg Bit-Bots](https://github.com/bit-bots/bitbots_main).

## Configuração do Ambiente

Este projeto utiliza:
- NVIDIA Jetpack 4.6.1
- ROS 2 Eloquent
- Python 3.6+
- CUDA 10.2
- TensorRT

## Estrutura do Projeto

```
.
├── docker/             # Configurações do ambiente Docker
├── scripts/            # Scripts de automação
├── src/                # Código fonte
│   ├── behavior/       # Comportamento do robô
│   ├── motion/         # Controle de movimento
│   ├── perception/     # Visão computacional
│   ├── navigation/     # Navegação e localização
│   └── msgs/           # Mensagens ROS personalizadas
└── requirements.txt    # Dependências Python
```

## Instalação

1. Clone este repositório:
```bash
git clone https://github.com/seu-usuario/RoboIME-HSL2025.git
cd RoboIME-HSL2025
```

2. Construa o ambiente Docker:
```bash
cd docker
docker build -f Dockerfile.jetson -t roboime-hsl:latest .
```

3. Execute o container:
```bash
docker run --runtime nvidia -it --rm --network host -v $(pwd):/ros2_ws roboime-hsl:latest
```

4. Compile o código:
```bash
cd /ros2_ws
./scripts/build.sh
```

## Executando o Robô

Para iniciar o sistema completo:
```bash
ros2 launch roboime_bringup robot.launch.py
```

## Sincronização com o Jetson

Para sincronizar o código com o Jetson:
```bash
./scripts/sync_jetson.sh
```

## Licença

Este projeto é licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.
