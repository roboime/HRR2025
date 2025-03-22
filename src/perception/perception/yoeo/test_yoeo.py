#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para testar o modelo YOEO com uma imagem ou vídeo.

Este script carrega um modelo YOEO treinado e o utiliza para processar
uma imagem ou vídeo, exibindo os resultados de detecção e segmentação.
"""

import os
import cv2
import numpy as np
import argparse
import yaml
import time

from yoeo_handler import YOEOHandler
from components.ball_component import BallDetectionComponent
from components.field_component import FieldSegmentationComponent
from components.line_component import LineSegmentationComponent
from components.goal_component import GoalDetectionComponent
from components.robot_component import RobotDetectionComponent
from components.referee_component import RefereeDetectionComponent


def load_config(config_path):
    """
    Carrega a configuração do arquivo YAML.
    
    Args:
        config_path: Caminho para o arquivo de configuração
        
    Returns:
        Dicionário com a configuração
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def process_image(image, yoeo_handler, components, show_fps=True):
    """
    Processa uma imagem com o modelo YOEO e seus componentes.
    
    Args:
        image: Imagem BGR
        yoeo_handler: Manipulador do modelo YOEO
        components: Dicionário com componentes de detecção/segmentação
        show_fps: Se deve mostrar informações de FPS
        
    Returns:
        Imagem com visualizações
    """
    # Medir tempo de processamento
    start_time = time.time()
    
    # Obter detecções e segmentações
    results = yoeo_handler.get_detections(image)
    
    # Processar campo primeiro (para usar como máscara)
    field_results = None
    if 'field' in components:
        field_results = components['field'].process(image)
        field_mask = field_results['mask'] if field_results else None
    else:
        field_mask = None
    
    # Processar outros componentes
    component_results = {}
    
    # Processar linhas (usando máscara de campo)
    if 'line' in components:
        component_results['line'] = components['line'].process(image)
    
    # Processar detecções de objetos
    if 'ball' in components:
        component_results['ball'] = components['ball'].process(image, field_mask)
    
    if 'goal' in components:
        component_results['goal'] = components['goal'].process(image)
    
    if 'robot' in components:
        component_results['robot'] = components['robot'].process(image, field_mask)
    
    if 'referee' in components:
        component_results['referee'] = components['referee'].process(image)
    
    # Calcular FPS
    processing_time = time.time() - start_time
    fps = 1.0 / processing_time if processing_time > 0 else 0
    
    # Criar visualização
    vis_image = image.copy()
    
    # Desenhar segmentações primeiro (para não cobrir detecções)
    if field_results:
        vis_image = components['field'].draw_segmentation(vis_image, field_results)
    
    if 'line' in component_results and component_results['line']:
        vis_image = components['line'].draw_segmentation(vis_image, component_results['line'])
    
    # Desenhar detecções
    if 'ball' in component_results and component_results['ball']:
        vis_image = components['ball'].draw_detections(vis_image, component_results['ball'])
    
    if 'goal' in component_results and component_results['goal']:
        vis_image = components['goal'].draw_detections(vis_image, component_results['goal'])
    
    if 'robot' in component_results and component_results['robot']:
        vis_image = components['robot'].draw_detections(vis_image, component_results['robot'])
    
    if 'referee' in component_results and component_results['referee']:
        vis_image = components['referee'].draw_detections(vis_image, component_results['referee'])
    
    # Adicionar informações de FPS
    if show_fps:
        cv2.putText(vis_image, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(vis_image, f"Tempo: {processing_time*1000:.1f} ms", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return vis_image


def process_video(video_path, output_path, yoeo_handler, components, show_fps=True):
    """
    Processa um vídeo com o modelo YOEO.
    
    Args:
        video_path: Caminho para o vídeo de entrada
        output_path: Caminho para o vídeo de saída
        yoeo_handler: Manipulador do modelo YOEO
        components: Dicionário com componentes de detecção/segmentação
        show_fps: Se deve mostrar informações de FPS
    """
    # Abrir vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return
    
    # Obter informações do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Configurar gravador de vídeo
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Processar cada quadro
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Processar quadro
        processed_frame = process_image(frame, yoeo_handler, components, show_fps)
        
        # Gravar quadro processado
        out.write(processed_frame)
        
        # Exibir quadro
        cv2.imshow('YOEO Test', processed_frame)
        
        # Atualizar contador
        frame_count += 1
        print(f"Processando quadro {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)", end='\r')
        
        # Verificar tecla pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nVídeo processado e salvo em: {output_path}")


def main():
    """Função principal."""
    # Configurar argumentos da linha de comando
    parser = argparse.ArgumentParser(description='Testar modelo YOEO em imagem ou vídeo')
    parser.add_argument('--model', type=str, required=True, help='Caminho para o modelo YOEO')
    parser.add_argument('--config', type=str, required=True, help='Caminho para o arquivo de configuração')
    parser.add_argument('--input', type=str, required=True, help='Caminho para imagem ou vídeo de entrada')
    parser.add_argument('--output', type=str, help='Caminho para imagem ou vídeo de saída')
    parser.add_argument('--confidence', type=float, default=0.5, help='Limiar de confiança para detecções')
    parser.add_argument('--no-ball', action='store_true', help='Desabilitar detecção de bola')
    parser.add_argument('--no-goal', action='store_true', help='Desabilitar detecção de gol')
    parser.add_argument('--no-robot', action='store_true', help='Desabilitar detecção de robô')
    parser.add_argument('--no-referee', action='store_true', help='Desabilitar detecção de árbitro')
    parser.add_argument('--no-field', action='store_true', help='Desabilitar segmentação de campo')
    parser.add_argument('--no-line', action='store_true', help='Desabilitar segmentação de linha')
    
    args = parser.parse_args()
    
    # Verificar se o arquivo de entrada existe
    if not os.path.exists(args.input):
        print(f"Erro: Arquivo de entrada não encontrado: {args.input}")
        return
    
    # Verificar se o modelo existe
    if not os.path.exists(args.model):
        print(f"Erro: Arquivo de modelo não encontrado: {args.model}")
        return
    
    # Verificar se o arquivo de configuração existe
    if not os.path.exists(args.config):
        print(f"Erro: Arquivo de configuração não encontrado: {args.config}")
        return
    
    # Carregar configuração
    config = load_config(args.config)
    
    # Criar manipulador YOEO
    print(f"Carregando modelo YOEO de {args.model}...")
    yoeo_handler = YOEOHandler(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    # Criar componentes
    components = {}
    
    if not args.no_field:
        components['field'] = FieldSegmentationComponent(yoeo_handler)
    
    if not args.no_line:
        components['line'] = LineSegmentationComponent(yoeo_handler)
    
    if not args.no_ball:
        ball_diameter = config.get('ball_diameter', 0.043)  # Padrão: 4.3 cm
        components['ball'] = BallDetectionComponent(yoeo_handler, ball_diameter, args.confidence)
    
    if not args.no_goal:
        goal_height = config.get('goal_height', 0.18)  # Padrão: 18 cm
        components['goal'] = GoalDetectionComponent(yoeo_handler, goal_height, args.confidence)
    
    if not args.no_robot:
        robot_height = config.get('robot_height', 0.15)  # Padrão: 15 cm
        components['robot'] = RobotDetectionComponent(yoeo_handler, robot_height, args.confidence)
    
    if not args.no_referee:
        referee_height = config.get('referee_height', 1.75)  # Padrão: 1.75 m
        components['referee'] = RefereeDetectionComponent(yoeo_handler, referee_height, args.confidence)
    
    # Verificar se é imagem ou vídeo
    is_video = args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    if is_video:
        # Definir caminho de saída para vídeo
        output_path = args.output if args.output else os.path.splitext(args.input)[0] + '_processed.avi'
        
        # Processar vídeo
        print(f"Processando vídeo: {args.input}")
        process_video(args.input, output_path, yoeo_handler, components)
    else:
        # Definir caminho de saída para imagem
        output_path = args.output if args.output else os.path.splitext(args.input)[0] + '_processed.jpg'
        
        # Carregar e processar imagem
        print(f"Processando imagem: {args.input}")
        image = cv2.imread(args.input)
        if image is None:
            print(f"Erro ao carregar a imagem: {args.input}")
            return
        
        # Processar imagem
        processed_image = process_image(image, yoeo_handler, components)
        
        # Salvar imagem processada
        cv2.imwrite(output_path, processed_image)
        print(f"Imagem processada salva em: {output_path}")
        
        # Exibir imagem
        cv2.imshow('YOEO Test', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main() 