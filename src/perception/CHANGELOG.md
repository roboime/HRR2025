# Changelog - Sistema de Percepção da RoboIME

## [3.0.0] - 2023-10-01 - Simplificação e Reorganização

### Modificado
- Reorganização completa da estrutura do sistema para maior clareza
- Unificação dos arquivos de configuração em um único arquivo `perception_config.yaml`
- Substituição dos múltiplos arquivos de lançamento por um único arquivo `perception.launch.py`
- Simplificação da interface de linha de comando com opções mais intuitivas

### Adicionado
- Novo script de teste interativo `test_perception.sh` com interface de menu
- Documentação mais didática no README.md
- Diagrama de arquitetura simplificado

### Removido
- Arquivos de configuração redundantes
- Parâmetros de lançamento desnecessários

## [2.2.0] - 2023-09-15 - Simplificação do Sistema

### Modificado
- Simplificação significativa da arquitetura do sistema de percepção
- Remoção do sistema complexo de fusão e substituição por escolhas diretas de detector
- Eliminação do modo de alternância, mantendo apenas um pipeline unificado e simplificado
- Redução substancial do número de parâmetros configuráveis

### Otimizado
- Código mais eficiente e fácil de manter
- Menor sobrecarga cognitiva para configuração do sistema
- Processo de seleção de detector mais direto e intuitivo 
- Desempenho melhorado com menos ramificações condicionais

## [2.1.0] - 2023-08-30 - Sistema Verdadeiramente Complementar

### Adicionado
- Implementação de um sistema verdadeiramente complementar onde YOEO e detectores tradicionais trabalham juntos simultaneamente
- Mecanismo de fusão inteligente para combinar resultados dos dois sistemas
- Configurações de pesos e regras para cada tipo de detecção
- Parâmetro `complementary_mode` para alternar entre modo complementar e modo alternância

### Modificado
- Refatoração do pipeline de visão para executar ambos os sistemas em paralelo
- Estratégias específicas para cada tipo de objeto (bola, campo, linhas, gols, robôs)
- Documentação atualizada para refletir o novo modo complementar

### Otimizado
- Melhor aproveitamento dos pontos fortes de cada sistema para cada tipo de detecção
- Renderização de informações de debug para mostrar o modo de operação ativo

## [2.0.0] - 2023-07-15 - Sistema Unificado

### Adicionado
- Sistema unificado que combina YOEO e detectores tradicionais
- Pipeline inteligente com estratégias de fallback automático
- Novo arquivo de lançamento `unified_vision.launch.py`
- Arquivo de configuração `unified_vision_params.yaml`
- Documentação ampliada no README e arquivos de lançamento
- Testes automatizados para o sistema unificado

### Modificado
- Reestruturação do pipeline de visão para suportar múltiplos sistemas
- Atualização de todos os arquivos README.md
- Melhorias no script de teste

### Otimizado
- Sistema inteligente para escolher automaticamente entre YOEO e detectores tradicionais
- Desempenho melhorado através da unificação

## [1.1.0] - 2023-03-01 - Implementação do YOEO

### Adicionado
- Sistema YOEO para detecção de objetos e segmentação semântica
- Suporte para TensorRT na Jetson Nano
- Componentes modulares para o YOEO

### Modificado
- Melhorias no pipeline de visão
- Otimizações para Jetson Nano

## [1.0.0] - 2023-01-15 - Lançamento Inicial

### Adicionado
- Detectores tradicionais baseados em visão computacional
- Detector de bola usando segmentação por cor e transformada de Hough
- Detector de campo usando segmentação por cor
- Detector de linhas usando detecção de bordas e transformada de Hough
- Detector de gols usando segmentação por cor
- Detector de obstáculos
- Suporte para câmera CSI e USB na Jetson Nano 