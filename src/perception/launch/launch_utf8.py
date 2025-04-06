#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper para o comando ros2 launch com codificação UTF-8.
Este script resolve o problema de 'ascii' codec can't decode byte no ROS2.

Uso:
    python3 launch_utf8.py <caminho_do_launch_file> [argumentos]
    
Exemplo:
    python3 launch_utf8.py perception/perception.launch.py camera_src:=csi
"""

import os
import sys
import rclpy
import subprocess
from pathlib import Path

# Configurar variáveis de ambiente
os.environ["PYTHONIOENCODING"] = "utf8"
os.environ["LANG"] = "C.UTF-8"
os.environ["LC_ALL"] = "C.UTF-8"

def main():
    """Função principal que executa o comando ros2 launch com UTF-8."""
    if len(sys.argv) < 2:
        print("Uso: python3 launch_utf8.py <caminho_do_launch_file> [argumentos]")
        print("Exemplo: python3 launch_utf8.py perception/perception.launch.py camera_src:=csi")
        return 1
    
    # Preparar argumentos para o comando launch
    launch_file = sys.argv[1]
    launch_args = sys.argv[2:]
    
    # Caminho completo do executável ros2
    ros2_cmd = "ros2"
    
    # Construir o comando
    cmd = [ros2_cmd, "launch"]
    cmd.append(launch_file)
    cmd.extend(launch_args)
    
    print(f"Executando com UTF-8: {' '.join(cmd)}")
    
    try:
        # Executar o comando com UTF-8
        proc = subprocess.Popen(
            cmd,
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8"
        )
        
        # Processar saída em tempo real
        while True:
            stdout_line = proc.stdout.readline()
            stderr_line = proc.stderr.readline()
            
            if stdout_line:
                print(stdout_line.strip())
            
            if stderr_line:
                print(stderr_line.strip(), file=sys.stderr)
            
            if not stdout_line and not stderr_line and proc.poll() is not None:
                break
        
        return proc.returncode
    
    except Exception as e:
        print(f"Erro ao executar o comando: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 