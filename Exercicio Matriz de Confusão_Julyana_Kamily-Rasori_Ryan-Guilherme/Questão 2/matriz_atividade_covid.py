import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# Definir número total de pessoas
total = 20

# Calcular número de pessoas em cada categoria mantendo as porcentagens
# TN: 35.0% (7 pessoas)
# FP: 20.0% (4 pessoas)
# FN: 20.0% (4 pessoas)
# TP: 25.0% (5 pessoas)

# Criar a matriz de confusão manualmente
conf_matrix = np.array([
    [7, 4],  # [TN, FP]
    [4, 5]   # [FN, TP]
])

# Extrair valores para uso posterior
tn, fp, fn, tp = conf_matrix.ravel()

# Calcular as métricas
accuracy = (tp + tn) / total
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Configuração para replicar o estilo da imagem
plt.rcParams.update({'font.size': 12})

# 1. HEATMAP DA MATRIZ DE CONFUSÃO
# --------------------------------
plt.figure(figsize=(10, 8))

# Definir as cores exatas da imagem
colors_heatmap = ['#8B0000', '#00008B', '#00BFFF', '#FFD700']  # Vermelho escuro, Azul escuro, Azul claro, Amarelo
cmap_custom = ListedColormap([colors_heatmap[0], colors_heatmap[1], colors_heatmap[2], colors_heatmap[3]])

# Criar um array 2x2 para as cores
color_array = np.array([[0, 1], [2, 3]])

# Valores percentuais para anotações
annot_matrix = np.array([
    [f"{tn/total:.2%}", f"{fp/total:.2%}"],
    [f"{fn/total:.2%}", f"{tp/total:.2%}"]
])

# Criar um heatmap personalizado
ax = plt.subplot(111)
ax.imshow(color_array, cmap=cmap_custom)

# Adicionar as anotações
for i in range(2):
    for j in range(2):
        plt.text(j, i, annot_matrix[i, j], 
                 ha="center", va="center", color="white", fontsize=14, fontweight='bold')

# Adicionar rótulos
labels = [['TN', 'FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j, i-0.2, labels[i][j], 
                 ha="center", va="center", color="white", fontsize=16, fontweight='bold')

# Configurar eixos
plt.xticks([0, 1], ['0', '1'], fontsize=12)
plt.yticks([0, 1], ['0', '1'], fontsize=12)
plt.xlabel('Preditos', fontsize=12)
plt.ylabel('Reais', fontsize=12)
plt.title('Covid-19', fontsize=14)

plt.tight_layout()
plt.savefig('matriz_confusao_covid_exact.png', bbox_inches='tight', dpi=300)
plt.show()  # Adicionado para mostrar o gráfico interativamente

# 2. GRÁFICO DE PIZZA
# -------------------
plt.figure(figsize=(8, 8))

# Cores exatas da imagem para o gráfico de pizza
colors_pie = ['#FF9999', '#99FF99', '#66B3FF', '#FFCC99']  # Vermelho claro, Verde claro, Azul claro, Laranja claro

# Valores em percentual
sizes = [35.0, 20.0, 20.0, 25.0]  # TN, FN, FP, TP (na ordem mostrada na imagem)

# Rótulos exatos
labels = ['TN', 'FN', 'FP', 'TP']

# Criar o gráfico de pizza
plt.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 12})

plt.axis('equal')
plt.title('Grafico de pizza', fontsize=14)

plt.tight_layout()
plt.savefig('pizza_covid_exact.png', bbox_inches='tight', dpi=300)
plt.show()  # Adicionado para mostrar o gráfico interativamente

# Imprimir os resultados no console para referência
print("Matriz de Confusão:")
print(conf_matrix)
print("\nMétricas de Avaliação:")
print(f"Acurácia: {accuracy:.2%}")
print(f"Precisão: {precision:.2%}")
print(f"Sensibilidade (Recall): {recall:.2%}")
print(f"F1 Score: {f1:.2%}")

print("\nExplicação da Matriz de Confusão:")
print(f"Verdadeiros Negativos (TN): {tn} ({tn/total:.2%}) - Pessoas sem COVID corretamente identificadas como negativas")
print(f"Falsos Positivos (FP): {fp} ({fp/total:.2%}) - Pessoas sem COVID incorretamente identificadas como positivas")
print(f"Falsos Negativos (FN): {fn} ({fn/total:.2%}) - Pessoas com COVID incorretamente identificadas como negativas")
print(f"Verdadeiros Positivos (TP): {tp} ({tp/total:.2%}) - Pessoas com COVID corretamente identificadas como positivas")