# Solução para Problemas de Codificação UTF-8/ASCII no ROS2

Este documento apresenta soluções para corrigir problemas de codificação de caracteres ao executar nós ROS2, especialmente ao lidar com caracteres não-ASCII em mensagens de log e strings.

## O Problema

Ao executar nós ROS2, você pode encontrar erros como:

```
UnicodeEncodeError: 'ascii' codec can't encode character '\xe7' in position 5: ordinal not in range(128)
```

Isso ocorre porque o ROS2 (ou Python) está tentando usar a codificação ASCII para lidar com strings que contêm caracteres Unicode (como acentos, caracteres especiais, etc).

## Soluções

### 1. Para Windows: Configuração Permanente

Execute o script `set_utf8_env.bat` como administrador para configurar permanentemente as variáveis de ambiente UTF-8 no Windows:

```
cd src/perception/scripts
set_utf8_env.bat
```

Este script:
- Configura variáveis de ambiente UTF-8 em nível de sistema
- Cria um script de inicialização automática
- Adiciona entradas ao registro do Windows

Após executar o script, reinicie o computador para garantir que todas as configurações sejam aplicadas.

### 2. Para Uso Imediato no Windows

Use o script wrapper `ros2_launch_utf8.bat` para executar comandos ROS2 com as configurações UTF-8 corretas:

```
src\perception\scripts\ros2_launch_utf8.bat launch perception perception.launch.py camera_src:=csi
```

### 3. Para Linux/Docker: Configuração Permanente

Execute o script `setup_env.sh` como root para configurar permanentemente as variáveis de ambiente UTF-8:

```
sudo bash src/perception/scripts/setup_env.sh
```

### 4. Para Docker: Configuração no Container

Execute o script `docker_utf8_setup.sh` dentro do container para configurar o ambiente:

```
bash /ros2_ws/src/perception/scripts/docker_utf8_setup.sh
```

Também pode incluir esse script no seu Dockerfile:

```dockerfile
COPY scripts/docker_utf8_setup.sh /setup/
RUN bash /setup/docker_utf8_setup.sh
```

### 5. Para Uso Imediato em Linux/Docker

Use o wrapper `ros2_utf8` para executar comandos ROS2 com as configurações UTF-8 corretas:

```
./src/perception/scripts/ros2_utf8 launch perception perception.launch.py camera_src:=csi
```

## Variáveis de Ambiente Importantes

Para referência, estas são as variáveis de ambiente que precisam ser configuradas:

```
PYTHONIOENCODING=utf8
LANG=C.UTF-8
LC_ALL=C.UTF-8
```

Em alguns casos, também é importante definir o PYTHONPATH:

```
PYTHONPATH="/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages:$PYTHONPATH"
```

## Verificação da Configuração

Para verificar se as variáveis de ambiente foram configuradas corretamente:

```
# No Windows
echo %PYTHONIOENCODING%
echo %LANG%
echo %LC_ALL%

# No Linux/Docker
echo $PYTHONIOENCODING
echo $LANG
echo $LC_ALL
```

## Troubleshooting

Se ainda encontrar problemas:

1. Verifique se o Python está instalado corretamente e suporta UTF-8.
2. Verifique se a versão do ROS2 está atualizada.
3. Certifique-se de que o terminal/console usado também suporta UTF-8.
4. Para scripts Python específicos, você pode adicionar `# -*- coding: utf-8 -*-` no início.

## Suporte e Feedback

Se encontrar problemas com estas soluções ou tiver sugestões adicionais, por favor, abra uma issue no repositório do projeto. 