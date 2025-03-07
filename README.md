from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Dados fictícios
potencia = [100, 150, 200, 250, 300]
peso = [1000, 1200, 1500, 1800, 2000]
economico = [1, 1, 0, 0, 0]  # 1 = Econômico, 0 = Não Econômico

# Treinando o modelo
X = list(zip(potencia, peso))
y = economico
modelo = DecisionTreeClassifier()
modelo.fit(X, y)

# Visualização da árvore
plt.figure(figsize=(10, 6))
plot_tree(modelo, filled=True, feature_names=["Potência", "Peso"])
plt.show()
