#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scripts para o sistema de percepção YOEO.

Este diretório contém scripts executáveis para várias tarefas relacionadas
ao sistema de percepção YOEO, incluindo:

- train_yoeo.py: Script para treinamento do modelo YOEO
- test_yoeo.py: Script para testar o modelo em imagens ou vídeos
- convert_model.py: Script para converter o modelo para formatos otimizados
- evaluate_model.py: Script para avaliar o desempenho do modelo
- collect_data.py: Script para coletar e preparar dados de treinamento
- visualize_results.py: Script para visualizar resultados de detecção e segmentação

Scripts para nós ROS:
- yoeo_detector_node.py: Nó detector YOEO
- yoeo_visualizer_node.py: Nó visualizador YOEO

Estes scripts podem ser executados diretamente do terminal e fornecem
interfaces de linha de comando para configurar parâmetros.
"""

# Versão do módulo de scripts
__version__ = '0.1.0' 