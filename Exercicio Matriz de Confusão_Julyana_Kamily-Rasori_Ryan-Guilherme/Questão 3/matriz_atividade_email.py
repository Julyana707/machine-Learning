import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Definindo os valores da matriz de confusão
tn = 140  # Verdadeiro Negativo (Não Spam corretamente classificado)
fp = 10   # Falso Positivo (Não Spam classificado como Spam)
fn = 20   # Falso Negativo (Spam classificado como Não Spam)
tp = 30   # Verdadeiro Positivo (Spam corretamente classificado)

# Criando a matriz de confusão
conf_matrix = np.array([[tn, fp], 
                        [fn, tp]])

# Calculando as métricas
total = tn + fp + fn + tp

# Acurácia: proporção de previsões corretas
accuracy = (tn + tp) / total

# Precisão: proporção de previsões positivas corretas
precision = tp / (tp + fp)

# Sensibilidade (Recall): proporção de positivos reais identificados
sensitivity = tp / (tp + fn)

# Especificidade: proporção de negativos reais identificados
specificity = tn / (tn + fp)

# F1-Score: média harmônica entre precisão e recall
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

# Exibindo os resultados
print("Matriz de Confusão:")
print(f"[[{tn}, {fp}]")
print(f" [{fn}, {tp}]]")
print("\nMétricas de Avaliação:")
print(f"Acurácia: {accuracy:.4f} ou {accuracy*100:.2f}%")
print(f"Precisão: {precision:.4f} ou {precision*100:.2f}%")
print(f"Sensibilidade/Recall: {sensitivity:.4f} ou {sensitivity*100:.2f}%")
print(f"Especificidade: {specificity:.4f} ou {specificity*100:.2f}%")
print(f"F1-Score: {f1_score:.4f} ou {f1_score*100:.2f}%")

# --- VISUALIZAÇÕES ---

# 1. Matriz de Confusão como Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Não Spam', 'Spam'],
            yticklabels=['Não Spam', 'Spam'])
plt.ylabel('Classe Real')
plt.xlabel('Classe Prevista')
plt.title('Matriz de Confusão - Classificação de Spam')
plt.savefig('matriz_confusao.png')
plt.show()

# 2. Gráfico de barras das métricas
plt.figure(figsize=(12, 6))
metricas = ['Acurácia', 'Precisão', 'Sensibilidade', 'Especificidade', 'F1-Score']
valores = [accuracy, precision, sensitivity, specificity, f1_score]
cores = ['blue', 'green', 'red', 'purple', 'orange']

plt.bar(metricas, valores, color=cores)
plt.ylim(0, 1)
plt.title('Métricas de Avaliação do Classificador')
plt.ylabel('Valor')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adiciona os valores em cima de cada barra
for i, v in enumerate(valores):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')

plt.savefig('metricas_avaliacao.png')
plt.show()

# 3. Gráfico pizza mostrando a distribuição dos resultados
plt.figure(figsize=(10, 8))
labels = ['Verdadeiro Negativo', 'Falso Positivo', 'Falso Negativo', 'Verdadeiro Positivo']
sizes = [tn, fp, fn, tp]
colors = ['#66b3ff', '#ff9999', '#99ff99', '#ffcc99']
explode = (0.1, 0, 0, 0.1)  # explode os acertos para destacar

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')  # Para garantir que o gráfico de pizza seja circular
plt.title('Distribuição dos Resultados da Classificação')
plt.savefig('distribuicao_resultados.png')
plt.show()