import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Dados da matriz de confusão fornecida
vp = 70  # Verdadeiro Positivo
fn = 5   # Falso Negativo
fp = 10  # Falso Positivo
vn = 65  # Verdadeiro Negativo

# Cálculo das métricas solicitadas
total = vp + fn + fp + vn

# Taxa de erro
taxa_erro = (fp + fn) / total * 100

# Taxa de acerto
taxa_acerto = (vp + vn) / total * 100

# Precisão
precisao = vp / (vp + fp) * 100

# Acurácia
acuracia = (vp + vn) / total * 100

# Sensibilidade (Recall)
sensibilidade = vp / (vp + fn) * 100

# Especificidade
especificidade = vn / (vn + fp) * 100

# F1-Score
f1_score = 2 * (precisao/100 * sensibilidade/100) / (precisao/100 + sensibilidade/100) * 100

# Imprimir os resultados
print("Resultados do teste de gravidez:")
print(f"Taxa de Erro: {taxa_erro:.2f}%")
print(f"Taxa de Acerto: {taxa_acerto:.2f}%")
print(f"Precisão: {precisao:.2f}%")
print(f"Acurácia: {acuracia:.2f}%")
print(f"Sensibilidade: {sensibilidade:.2f}%")
print(f"Especificidade: {especificidade:.2f}%")
print(f"F1-Score: {f1_score:.2f}%")

# Criar e mostrar a matriz de confusão
matriz_confusao = np.array([[vp, fn], [fp, vn]])
print("\nMatriz de Confusão:")
print(f"VP: {vp}, FN: {fn}")
print(f"FP: {fp}, VN: {vn}")

# Criar a matriz binária (VP, VN, FP, FN)
matriz_binaria = np.zeros((2, 2))
matriz_binaria[0, 0] = 1  # VP (sempre presente neste caso)
matriz_binaria[0, 1] = 1  # FN (sempre presente neste caso)
matriz_binaria[1, 0] = 1  # FP (sempre presente neste caso)
matriz_binaria[1, 1] = 1  # VN (sempre presente neste caso)

print("\nMatriz Binária:")
print(matriz_binaria)

# Configuração da visualização
plt.figure(figsize=(15, 10))

# 1. Visualização da Matriz de Confusão
plt.subplot(2, 2, 1)
df_cm = pd.DataFrame(matriz_confusao, 
                    index=["Positivo (Real)", "Negativo (Real)"],
                    columns=["Positivo (Previsto)", "Negativo (Previsto)"])
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')

# 2. Visualização da Matriz Binária
plt.subplot(2, 2, 2)
df_bin = pd.DataFrame(matriz_binaria, 
                     index=["Positivo (Real)", "Negativo (Real)"],
                     columns=["Positivo (Previsto)", "Negativo (Previsto)"])
sns.heatmap(df_bin, annot=True, fmt='.0f', cmap='Greens')
plt.title('Matriz Binária')

# 3. Gráfico de barras para as métricas
plt.subplot(2, 2, 3)
metricas = ['Taxa de Erro', 'Taxa de Acerto', 'Precisão', 
            'Acurácia', 'Sensibilidade', 'Especificidade', 'F1-Score']
valores = [taxa_erro, taxa_acerto, precisao, acuracia, sensibilidade, especificidade, f1_score]

cores = ['red', 'green', 'blue', 'green', 'purple', 'orange', 'cyan']
plt.bar(metricas, valores, color=cores)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.ylabel('Porcentagem (%)')
plt.title('Métricas de Desempenho')

for i, v in enumerate(valores):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center')

# 4. Gráfico de pizza para visualizar proporções
plt.subplot(2, 2, 4)
labels = ['Verdadeiro Positivo', 'Falso Negativo', 'Falso Positivo', 'Verdadeiro Negativo']
sizes = [vp, fn, fp, vn]
colors = ['#66b3ff', '#ff9999', '#99ff99', '#ffcc99']
explode = (0.1, 0, 0, 0)  # Destacar VP

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.title('Distribuição dos Resultados')

plt.tight_layout()
plt.savefig('teste_gravidez_resultados.png')
plt.show()

# Figura adicional para visualizar VP, FP, FN, VN
plt.figure(figsize=(10, 6))
categorias = ['Positivos Reais', 'Negativos Reais']
valores_corretos = [vp, vn]
valores_incorretos = [fn, fp]

plt.bar(categorias, valores_corretos, label='Predições Corretas', color='green')
plt.bar(categorias, valores_incorretos, bottom=valores_corretos, label='Predições Incorretas', color='red')

plt.xlabel('Categoria')
plt.ylabel('Quantidade')
plt.title('Acertos e Erros do Teste de Gravidez')
plt.legend()

# Adicionar anotações
plt.text(0, vp/2, f'VP: {vp}', ha='center', color='white', fontweight='bold')
plt.text(1, vn/2, f'VN: {vn}', ha='center', color='white', fontweight='bold')
plt.text(0, vp + fn/2, f'FN: {fn}', ha='center', color='white', fontweight='bold')
plt.text(1, vn + fp/2, f'FP: {fp}', ha='center', color='white', fontweight='bold')

plt.savefig('teste_gravidez_acertos_erros.png')
plt.show()