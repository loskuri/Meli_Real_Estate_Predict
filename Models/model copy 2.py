import pandas as pd
import numpy as np
import os
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, QuantileRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import joblib  # Para guardar los modelos
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

banos = 'full_bathrooms'
dormitorios = 'bedrooms'
ambientes = "rooms"
superficie_total = "superficie_total" 
superficie_cubierta = "superficie_cubierta"
precio = 'price'
ln_precio = 'ln_precio_USD'
precio_metro = 'precio_m2'
tipo_propiedad = 'property_type'
operacion = 'operation_type'
estado = 'location_state'
barrio = 'location_city'

df = pd.read_csv(r'D:\Tesis\base_limpia_tesis\data_clean_v3.csv')

output_dir = r'D:\Tesis\output_modelos_v5'

vars = [banos, dormitorios, ambientes, superficie_total, superficie_cubierta]
target = "ln_precio_USD"

def calcula_metricas(y_true, y_pred):
    """
    Calcula diversas métricas de evaluación para un modelo de regresión.
    
    Parámetros:
    - y_true: Valores reales.
    - y_pred: Valores predichos por el modelo.
    
    Retorna:
    - metrics: Diccionario con las métricas calculadas.
    """
    # Calcular métricas estándar
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calcular el Error Relativo Promedio
    with np.errstate(divide='ignore', invalid='ignore'):
        error_relativo = (y_pred - y_true) / y_true
        error_relativo = np.where(y_true == 0, np.nan, error_relativo)  # Asignar NaN donde Precio Real es 0
    
    # Calcular el promedio del Error Relativo, ignorando NaN
    error_relativo_promedio = np.nanmean(error_relativo)
    
    # Agregar todas las métricas a un diccionario
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Error Relativo Promedio': error_relativo_promedio
    }
    
    return metrics

def guardar_resultados(model_name, params, metrics, iteration, output_dir):
    # Crear directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Excluir 'model_instance' de 'params' al crear el DataFrame
    params_without_model = {k: v for k, v in params.items() if k != 'model_instance'}
    df_resultados = pd.DataFrame([{'Modelo': model_name, 'Iteración': iteration, **params_without_model, **metrics}])
    resultados_path = os.path.join(output_dir, f'resultados_{model_name}.xlsx')
    
    # Si el archivo Excel existe, leerlo y agregar nuevos resultados
    if not os.path.exists(resultados_path):
        # Guardar DataFrame en un nuevo archivo Excel
        df_resultados.to_excel(resultados_path, index=False)
    else:
        # Leer el archivo Excel existente
        existing_df = pd.read_excel(resultados_path)
        # Concatenar el nuevo DataFrame con el existente
        updated_df = pd.concat([existing_df, df_resultados], ignore_index=True)
        # Guardar el DataFrame actualizado en el archivo Excel
        updated_df.to_excel(resultados_path, index=False)
    
    # Guardar el modelo
    model_path = os.path.join(output_dir, f'{model_name}_iter_{iteration}.joblib')
    joblib.dump(params['model_instance'], model_path)

def obtener_conjuntos_nulos(df, target, thresholds=[0.05, 0.10]):
    conjuntos_nulos = {}
    for thresh in thresholds:
        missing = df.isnull().mean()
        vars_incluidas = missing[missing < thresh].index.tolist()
        if target in vars_incluidas:
            vars_incluidas.remove(target)
        conjuntos_nulos[f'conjunto_nulos_{int(thresh*100)}'] = vars_incluidas
    return conjuntos_nulos

def preprocesar_datos(X):
    # Identificar variables numéricas y categóricas
    num_vars = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_vars = X.select_dtypes(include=['object', 'category']).columns.tolist()
    # Definir transformaciones
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_vars),
            ('cat', categorical_transformer, cat_vars)
        ])
    # Aplicar transformaciones
    X_processed = preprocessor.fit_transform(X)
    return X_processed, preprocessor

# A continuación, las funciones 'modelo_' y 'correr_' para cada modelo permanecen sin cambios significativos, excepto por el uso de 'guardar_resultados' actualizado.

# Ejemplo de modelo Lasso:
def modelo_lasso(X_train, y_train, X_test, y_test, params, iteration, output_dir):
    model = Lasso(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calcula_metricas(y_test, y_pred)
    # Guardar resultados
    guardar_resultados('Lasso', {'alpha': params.get('alpha'), 'model_instance': model}, metrics, iteration, output_dir)
    return model, metrics

def correr_lasso(df, vars, target, output_dir):
    X = df[vars]
    y = df[target]
    # Manejo de nulos
    X = X.dropna()
    y = y.loc[X.index]
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Grilla de hiperparámetros
    param_distributions = {'alpha': np.logspace(-4, 0, 50)}
    lasso = Lasso()
    random_search = RandomizedSearchCV(lasso, param_distributions, n_iter=10, scoring='neg_mean_squared_error', cv=3, random_state=42)
    random_search.fit(X_train, y_train)
    # Iterar sobre los resultados y guardar
    for i, params in enumerate(random_search.cv_results_['params']):
        model_params = {'alpha': params['alpha']}
        model, metrics = modelo_lasso(X_train, y_train, X_test, y_test, model_params, i, output_dir)
    # Guardar el mejor modelo
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    metrics = calcula_metricas(y_test, best_model.predict(X_test))
    guardar_resultados('Lasso_Best', {'alpha': best_params['alpha'], 'model_instance': best_model}, metrics, 'best', output_dir)
    return best_model, metrics

# Repite esto para las demás funciones de los modelos.

def main():
    correr_lasso(df, vars, target, output_dir)
    correr_decision_tree(df, vars, target, output_dir)
    correr_bagging(df, vars, target, output_dir)
    correr_xgboost(df, vars, target, output_dir)
    correr_random_forest(df, vars, target, output_dir)
    correr_regresion_lineal(df, vars, target, output_dir)
    correr_lightgbm(df, vars, target, output_dir)
    correr_mlp(df, vars, target, output_dir)
    correr_knn(df, vars, target, output_dir)
    correr_xgboost_quantile(df, vars, target, output_dir, quantile=0.5)
    correr_lasso_quantile(df, vars, target, output_dir, quantile=0.5)
    # Asegúrate de que no hay llamadas duplicadas a las funciones de los modelos.

if __name__ == "__main__":
    main()
