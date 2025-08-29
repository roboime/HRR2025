# Exemplos de Detecção YOLOv8

Esta pasta contém imagens de exemplo com detecções do sistema YOLOv8.

## 📁 Conteúdo

### **Resultados de Detecção**
- `resultado_*.jpg`: Imagens com bounding boxes das detecções
- `before_after/`: Comparação antes/depois do treinamento
- `test_cases/`: Casos de teste específicos

## 🎯 Tipos de Exemplos

### **Detecções Típicas**
- **Bola**: Detecção da bola em diferentes condições
- **Robôs**: Identificação de robôs azuis e vermelhos
- **Gols**: Detecção de estruturas de gol
- **Landmarks**: Círculo central, marcas de penalty, cantos

### **Cenários Desafiadores**
- Múltiplos objetos na cena
- Condições de iluminação variadas
- Objetos parcialmente oclusos
- Distâncias diferentes da câmera

## 🚀 Gerar Novos Exemplos

### **Com Modelo Treinado**
```python
from ultralytics import YOLO

# Carregar modelo
model = YOLO('../yolov8/robocup_yolov8.pt')

# Executar detecção
results = model('test_image.jpg')

# Salvar resultado com detecções
results[0].save('resultado_test.jpg')
```

### **Via Sistema ROS2**
```bash
# Executar sistema com debug
ros2 launch perception perception.launch.py debug:=true

# Salvar imagem de debug
ros2 topic echo /perception/debug_image > debug_output.jpg
```

## 📊 Análise de Performance

### **Métricas por Classe**
```python
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO('../yolov8/robocup_yolov8.pt')

# Validar com dataset
results = model.val()

# Plotar métricas
results.plot()
```

### **Benchmark com Imagens**
```bash
# Testar velocidade
cd ../../training
python train_model.py --check-only

# Benchmark específico
python benchmark_model.py --images examples/
```

---

**💡 Dica**: Use esta pasta para documentar visualmente o progresso do modelo e identificar casos onde melhorias são necessárias. 