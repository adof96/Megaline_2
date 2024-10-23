

# %% [markdown]
# # Introduccion
# La compañía móvil Megaline no está satisfecha al ver que muchos de sus clientes utilizan planes heredados. Quieren desarrollar un modelo que pueda analizar el comportamiento de los clientes y recomendar uno de los nuevos planes de Megaline: Smart o Ultra.
# 
# Tenemos acceso a los datos de comportamiento de los suscriptores que ya se han cambiado a los planes nuevos (del proyecto megaline 1). Para esta tarea de clasificación debemos crear un modelo que escoja el plan correcto. Como ya hicimos el paso de procesar los datos, podemos lanzarnos directo a crear el modelo.

# %% [markdown]
# ## Inicializacion y carga de datos

# %%
#importar librerias y funciones
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %%
#leer el data frame
df = pd.read_csv('datasets/users_behavior.csv')

# %%
# visualizar la estructura del data frame
df.info()
df.head()


# %% [markdown]
# ## Segmenta los datos

# %%
# Divide el DataFrame en conjunto de entrenamiento y conjunto restante
df_train, df_temp = train_test_split(df, test_size=0.3, random_state=12345)

# Divide el conjunto restante en conjunto de validación y conjunto de prueba
df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=12345)

# Verifica el tamaño de cada conjunto
print(f'Tamaño del conjunto de entrenamiento: {df_train.shape}')
print(f'Tamaño del conjunto de validación: {df_valid.shape}')
print(f'Tamaño del conjunto de prueba: {df_test.shape}')

# %%
features_train = df_train.drop(['is_ultra'], axis=1)
target_train = df_train['is_ultra']
features_valid = df_valid.drop(['is_ultra'], axis=1)
target_valid = df_valid['is_ultra']


# %% [markdown]
# ## Modelos
# ### Arbol de decision

# %%
#creamos un bucle para ver la calidad cambiando hiperparametro depth
for depth in range(1,10):
    model= DecisionTreeClassifier(random_state=12345,max_depth=depth)
    model.fit(features_train,target_train)
    predictions_valid= model.predict(features_valid)
    print('max_depth =', depth, ': ', end='')
    print(accuracy_score(target_valid, predictions_valid))


# %% [markdown]
# ### Bosque aleatorio

# %%
#tambien crearemos un bucle para poder encontrar el mejor modelo cambiando los estimadores
best_score = 0
best_est = 0
for est in range(1, 15): 
    model = RandomForestClassifier(random_state=54321, n_estimators= est )
    model.fit(features_train,target_train) 
    score = model.score(features_valid,target_valid) 
    if score > best_score:
        best_score = score
        best_est = est

print("La exactitud del mejor modelo en el conjunto de validación (n_estimators = {}): {}".format(best_est, best_score))

# %% [markdown]
# ### Regresion logistica

# %%
 # iniciamos el constructor de regresión logística con los parámetros random_state=12345 y solver='liblinear'
model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train,target_train) 
score_train = model.score(features_train,target_train)
score_valid = model.score(features_valid,target_valid) 

print("Accuracy del modelo de regresión logística en el conjunto de entrenamiento:", score_train)
print("Accuracy del modelo de regresión logística en el conjunto de validación:", score_valid)

# %% [markdown]
# Podemos observar que el modelo que tiene mayor precision es el bosque aleatorio con 0.7946 teniendo 12 estimadores, seguido del arbol de decision con 0.7883 con una profundidad de 4 y por ultimo la regresion logistica con 0.6950. Por lo que para este ejercicio en particular lo mejor seria utilizar un bosque aleatorio.


# %% [markdown]
# ## Comprobar la calidad del modelo

# %%
test_features = df_test.drop(['is_ultra'], axis=1)
test_target = df_test['is_ultra']
test_predictions = model.predict(test_features)
def error_count(answers, predictions):
    count = 0
    for i in range(len(answers)):
        if answers[i] != predictions[i]:
            count += 1
    return count

# Convertir test_target a una lista antes de pasarla a error_count
print('Errores:', error_count(test_target.tolist(), test_predictions))



# %% [markdown]
# ## Conclusion

# %% [markdown]
# Ya que el objetivo era desarrolla un modelo con la mayor exactitud posible. Y para este proyecto, el umbral de exactitud es 0.75. Podemos concluir que la mejor manera de superar ese umbral es utilizando el bosque aleatorio con 12 estimadores el cual fue el mejor resultado posible con 0.7946 de exactitud. 





