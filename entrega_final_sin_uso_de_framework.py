# Librerias utilizadas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Funciones para calcular predicciones utilizando Regresión Linear

def regresionlinear(X, b0, b1):
    return [b0+b1*x for x in X]

# b0 Variable independiente
def b0(X, Y, b1): 
    x_ = np.mean(X)
    y_ = np.mean(Y)
    return y_-b1*x_

# b1 Variable dependiente
def b1(X, Y):
    x_ = np.mean(X)
    y_ = np.mean(Y)
    rise = sum([(x-x_) * (y-y_) for x,y in zip(X,Y)])
    run = sum([(x-x_)**2 for x,y in zip(X,Y)])
    return rise / run

# Lectura de datos
data= pd.read_csv("winequality-red (1).csv")

#Normalización
for column in data.columns:
    data[column] = (data[column] -
                           data[column].mean()) / data[column].std()   

# Mapa de Correlaciones

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True)
plt.show()


# Separación de datos para realizar predicciones de nivel de alcohol según el pH

datapred = data[['pH', 'alcohol']]
predictora = data['pH']
objetivo = data['alcohol']

# Graficando distribución de los datos

plt.figure(figsize=(8,5))
plt.title("Alcohol vs pH")
plt.scatter(predictora, objetivo, color = "#247ba0")
plt.xlabel('pH')
plt.ylabel('Alcohol')
plt.show()

# Aplicando las predicciones a nuestros datos 

b1_pred = b1(predictora, objetivo)
b0_pred = b0(predictora, objetivo, b1_pred)
predicted = regresionlinear(predictora, b0_pred, b1_pred)

# Graficando las predicciones obtenidas

plt.figure(figsize = (8, 5))
plt.plot(predictora, predicted, color = '#f25f5c')
plt.scatter(predictora, predicted, color = '#f25f5c')
plt.title('Valores Predecidos usando Regresión Linear', fontsize = 15)
plt.xlabel('pH')
plt.ylabel('Alcohol')
plt.scatter(predictora, objetivo, color = "#247ba0")
plt.show()