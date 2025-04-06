#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para corrigir o problema de codificação UTF-8 no ROS2.
Este script modifica os arquivos Python do ROS2 para garantir que usem UTF-8.

Execute com: sudo python3 patch_ros2_utf8.py
"""

import os
import sys
import glob
import shutil
import subprocess
from pathlib import Path

# Códigos de cores para saída formatada
GREEN = '\033[0;32m'
RED = '\033[0;31m'
BLUE = '\033[0;34m'
YELLOW = '\033[0;33m'
NC = '\033[0m'  # No Color

def print_header(message):
    """Imprime um cabeçalho formatado."""
    print(f"\n{BLUE}============================================================{NC}")
    print(f"{BLUE}{message}{NC}")
    print(f"{BLUE}============================================================{NC}\n")

def print_success(message):
    """Imprime uma mensagem de sucesso."""
    print(f"{GREEN}✓ {message}{NC}")

def print_error(message):
    """Imprime uma mensagem de erro."""
    print(f"{RED}✗ {message}{NC}")

def print_info(message):
    """Imprime uma mensagem informativa."""
    print(f"{YELLOW}ℹ {message}{NC}")

def check_root():
    """Verifica se o script está sendo executado como root."""
    if os.geteuid() != 0:
        print_error("Este script precisa ser executado como root.")
        print_info("Execute: sudo python3 patch_ros2_utf8.py")
        sys.exit(1)

def get_ros2_paths():
    """Encontra os caminhos relevantes do ROS2."""
    paths = {}
    
    # Encontrar o executável ros2
    try:
        ros2_executable = subprocess.check_output(['which', 'ros2']).decode('utf-8').strip()
        paths['ros2_executable'] = ros2_executable
    except subprocess.CalledProcessError:
        print_error("Executável ros2 não encontrado.")
        sys.exit(1)
    
    # Encontrar o diretório de instalação do ROS2
    try:
        # Usar o módulo Python rclpy como referência
        import rclpy
        rclpy_path = rclpy.__path__[0]
        ros2_root = str(Path(rclpy_path).parent.parent)
        paths['ros2_root'] = ros2_root
    except (ImportError, IndexError):
        print_error("Não foi possível encontrar o módulo rclpy do ROS2.")
        print_info("Tentando localizar diretório de instalação do ROS2 por outros meios...")
        # Tentar encontrar pelo diretório /opt/ros
        if os.path.exists('/opt/ros'):
            ros_distros = [d for d in os.listdir('/opt/ros') if os.path.isdir(os.path.join('/opt/ros', d))]
            if ros_distros:
                # Pegar a versão mais recente do ROS2
                ros_distros.sort()
                ros_distro = ros_distros[-1]
                ros2_root = f'/opt/ros/{ros_distro}'
                paths['ros2_root'] = ros2_root
                print_info(f"Usando diretório ROS2: {ros2_root}")
            else:
                print_error("Nenhuma versão do ROS2 encontrada em /opt/ros.")
                sys.exit(1)
        else:
            print_error("Não foi possível determinar o diretório de instalação do ROS2.")
            sys.exit(1)
    
    # Encontrar os diretórios de pacotes Python
    import site
    python_site_packages = site.getsitepackages()[0]
    paths['python_site_packages'] = python_site_packages
    
    return paths

def create_executable_wrapper(ros2_executable):
    """Cria um wrapper para o executável ros2 que configura UTF-8."""
    # Backup do executável original
    backup_path = f"{ros2_executable}.bak"
    if not os.path.exists(backup_path):
        shutil.copy2(ros2_executable, backup_path)
        print_success(f"Backup criado: {backup_path}")
    
    # Criar novo executável com UTF-8 embutido
    wrapper_content = """#!/bin/bash

# ROS2 wrapper com configurações UTF-8
export PYTHONIOENCODING=utf8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONPATH="/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages:$PYTHONPATH"

# Obter o diretório do script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Executar o binário original com as configurações corretas
"${SCRIPT_DIR}/ros2.bak" "$@"
"""
    
    with open(ros2_executable, 'w') as f:
        f.write(wrapper_content)
    
    # Tornar executável
    os.chmod(ros2_executable, 0o755)
    print_success(f"Executável ros2 modificado: {ros2_executable}")

def patch_python_files(ros2_root, python_site_packages):
    """Corrige arquivos Python do ROS2 para usar UTF-8."""
    # Lista de diretórios para verificar
    dirs_to_check = [
        os.path.join(ros2_root, 'lib/python3.*/site-packages/launch'),
        os.path.join(python_site_packages, 'launch'),
        os.path.join(ros2_root, 'lib/python3.*/site-packages/ros2launch'),
        os.path.join(python_site_packages, 'ros2cli')
    ]
    
    # Expansão de diretórios
    expanded_dirs = []
    for dir_pattern in dirs_to_check:
        expanded = glob.glob(dir_pattern)
        expanded_dirs.extend(expanded)
    
    # Remover duplicados
    expanded_dirs = list(set(expanded_dirs))
    
    files_patched = 0
    for directory in expanded_dirs:
        if os.path.exists(directory):
            print_info(f"Verificando diretório: {directory}")
            
            # Encontrar todos os arquivos Python
            for root, _, files in os.walk(directory):
                for filename in files:
                    if filename.endswith('.py'):
                        filepath = os.path.join(root, filename)
                        
                        # Ler conteúdo atual
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Verificar se precisa adicionar encoding UTF-8
                        needs_update = True
                        
                        if '# -*- coding: utf-8 -*-' in content:
                            needs_update = False
                        
                        if needs_update:
                            # Modificar o arquivo
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = f.readlines()
                            
                            # Adicionar encoding UTF-8 após o shebang
                            new_lines = []
                            if lines and lines[0].startswith('#!'):
                                new_lines.append(lines[0])
                                new_lines.append('# -*- coding: utf-8 -*-\n')
                                
                                # Adicionar configuração de ambiente
                                new_lines.append('import os\n')
                                new_lines.append('os.environ["PYTHONIOENCODING"] = "utf8"\n')
                                new_lines.append('os.environ["LANG"] = "C.UTF-8"\n')
                                new_lines.append('os.environ["LC_ALL"] = "C.UTF-8"\n')
                                new_lines.append('\n')
                                
                                # Adicionar restante do arquivo
                                new_lines.extend(lines[1:])
                            else:
                                new_lines.append('# -*- coding: utf-8 -*-\n')
                                new_lines.append('import os\n')
                                new_lines.append('os.environ["PYTHONIOENCODING"] = "utf8"\n')
                                new_lines.append('os.environ["LANG"] = "C.UTF-8"\n')
                                new_lines.append('os.environ["LC_ALL"] = "C.UTF-8"\n')
                                new_lines.append('\n')
                                new_lines.extend(lines)
                            
                            # Gravar conteúdo modificado
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.writelines(new_lines)
                            
                            files_patched += 1
                            print_success(f"Arquivo corrigido: {filepath}")
    
    print_info(f"Total de arquivos corrigidos: {files_patched}")

def setup_system_environment():
    """Configura as variáveis de ambiente do sistema."""
    # Criar arquivo em /etc/profile.d/
    profile_file = '/etc/profile.d/ros2_utf8.sh'
    
    # Conteúdo do arquivo
    content = """#!/bin/bash

# Configurações UTF-8 para ROS2
export PYTHONIOENCODING=utf8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONPATH="/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages:$PYTHONPATH"
"""
    
    # Criar arquivo
    with open(profile_file, 'w') as f:
        f.write(content)
    
    # Tornar executável
    os.chmod(profile_file, 0o755)
    print_success(f"Arquivo de ambiente criado: {profile_file}")
    
    # Adicionar ao /etc/environment
    with open('/etc/environment', 'r') as f:
        env_content = f.read()
    
    # Verificar se as variáveis já existem
    updates_needed = []
    if 'PYTHONIOENCODING=utf8' not in env_content:
        updates_needed.append('PYTHONIOENCODING=utf8')
    if 'LANG=C.UTF-8' not in env_content:
        updates_needed.append('LANG=C.UTF-8')
    if 'LC_ALL=C.UTF-8' not in env_content:
        updates_needed.append('LC_ALL=C.UTF-8')
    
    # Adicionar variáveis que não existem
    if updates_needed:
        with open('/etc/environment', 'a') as f:
            for update in updates_needed:
                f.write(f'\n{update}')
        print_success(f"Variáveis adicionadas ao /etc/environment: {', '.join(updates_needed)}")
    else:
        print_info("Variáveis já existem em /etc/environment")

def patch_specific_files(ros2_root):
    """Corrige arquivos específicos que podem causar problemas."""
    # Lista de arquivos específicos para corrigir
    specific_files = [
        glob.glob(os.path.join(ros2_root, '**/launch_service.py'), recursive=True),
        glob.glob(os.path.join(ros2_root, '**/launch_description.py'), recursive=True),
        glob.glob(os.path.join(ros2_root, '**/launch/**/parser.py'), recursive=True)
    ]
    
    # Aplanar a lista
    all_specific_files = []
    for file_list in specific_files:
        all_specific_files.extend(file_list)
    
    for filepath in all_specific_files:
        if os.path.isfile(filepath):
            print_info(f"Corrigindo arquivo específico: {filepath}")
            
            # Ler conteúdo atual
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Modificar o código para usar UTF-8 explicitamente
            content = content.replace('.decode(', '.decode("utf-8", ')
            content = content.replace('.encode(', '.encode("utf-8", ')
            
            # Se o arquivo não tiver indicação de codificação, adicionar
            if '# -*- coding: utf-8 -*-' not in content:
                if content.startswith('#!'):
                    lines = content.split('\n')
                    content = lines[0] + '\n# -*- coding: utf-8 -*-\n' + '\n'.join(lines[1:])
                else:
                    content = '# -*- coding: utf-8 -*-\n' + content
            
            # Gravar conteúdo modificado
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print_success(f"Arquivo específico corrigido: {filepath}")

def apply_immediate_environment():
    """Aplica as configurações de ambiente para a sessão atual."""
    os.environ['PYTHONIOENCODING'] = 'utf8'
    os.environ['LANG'] = 'C.UTF-8'
    os.environ['LC_ALL'] = 'C.UTF-8'
    print_success("Variáveis de ambiente aplicadas para a sessão atual")

def main():
    """Função principal para aplicar os patches."""
    print_header("Aplicando patches para corrigir o problema de codificação UTF-8 no ROS2")
    
    # Verificar permissões
    check_root()
    
    # Obter caminhos do ROS2
    paths = get_ros2_paths()
    
    # Criar wrapper para o executável ros2
    create_executable_wrapper(paths['ros2_executable'])
    
    # Corrigir arquivos Python
    patch_python_files(paths['ros2_root'], paths['python_site_packages'])
    
    # Corrigir arquivos específicos
    patch_specific_files(paths['ros2_root'])
    
    # Configurar variáveis de ambiente do sistema
    setup_system_environment()
    
    # Aplicar configurações para sessão atual
    apply_immediate_environment()
    
    print_header("Correção concluída com sucesso!")
    print_info("As configurações de codificação UTF-8 foram aplicadas permanentemente ao ROS2.")
    print_info("Reinicie o sistema para garantir que todas as alterações sejam aplicadas.")
    print_info("Para testar, execute: ros2 launch perception perception.launch.py camera_src:=csi")

if __name__ == "__main__":
    main() 