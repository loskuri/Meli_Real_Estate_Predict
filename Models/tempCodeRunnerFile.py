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

df = pd.read_csv(r'D:\Tesis\base_limpia_tesis\data_clean.csv')

output_dir = r'D:\Tesis\output_modelos'

scoring = 'neg_mean_squared_error' ## QUE VERGA ES ESTO
## FALTA LOCATION NORTE
vars = [banos,dormitorios,ambientes,superficie_total,superficie_cubierta]
 # "dias_desde_fecha"
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
    
    # Guardar parámetros y métricas en un archivo CSV
    df_resultados = pd.DataFrame([{'Modelo': model_name, 'Iteración': iteration, **params, **metrics}])
    resultados_path = os.path.join(output_dir, f'resultados_{model_name}.csv')
    if not os.path.exists(resultados_path):
        df_resultados.to_csv(resultados_path, index=False)
    else:
        df_resultados.to_csv(resultados_path, mode='a', header=False, index=False)
    
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
def modelo_random_forest(X_train, y_train, X_test, y_test, params, iteration, output_dir):
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calcula_metricas(y_test, y_pred)
    # Guardar resultados
    guardar_resultados('RandomForest', {**params, 'model_instance': model}, metrics, iteration, output_dir)
    return model, metrics

def correr_random_forest(df, vars, target, output_dir):
    X = df[vars]
    y = df[target]
    X = X.dropna()
    y = y.loc[X.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    param_distributions = {
        'n_estimators': [int(x) for x in np.linspace(100, 1000, 10)],
        'max_depth': [int(x) for x in np.linspace(10, 100, 10)],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor()
    random_search = RandomizedSearchCV(rf, param_distributions, n_iter=10, scoring='neg_mean_squared_error', cv=3, random_state=42)
    random_search.fit(X_train, y_train)
    for i, params in enumerate(random_search.cv_results_['params']):
        model, metrics = modelo_random_forest(X_train, y_train, X_test, y_test, params, i, output_dir)
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    metrics = calcula_metricas(y_test, best_model.predict(X_test))
    guardar_resultados('RandomForest_Best', {**best_params, 'model_instance': best_model}, metrics, 'best', output_dir)
    return best_model, metrics
def modelo_regresion_lineal(X_train, y_train, X_test, y_test, iteration, output_dir):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calcula_metricas(y_test, y_pred)
    guardar_resultados('LinearRegression', {'model_instance': model}, metrics, iteration, output_dir)
    return model, metrics

def correr_regresion_lineal(df, vars, target, output_dir):
    X = df[vars]
    y = df[target]
    X = X.dropna()
    y = y.loc[X.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train_processed, preprocessor = preprocesar_datos(X_train)
    X_test_processed = preprocessor.transform(X_test)
    model, metrics = modelo_regresion_lineal(X_train_processed, y_train, X_test_processed, y_test, 0, output_dir)
    return model, metrics


def modelo_decision_tree(X_train, y_train, X_test, y_test, params, iteration, output_dir):
    model = DecisionTreeRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calcula_metricas(y_test, y_pred)
    # Guardar resultados
    guardar_resultados('DecisionTree', {**params, 'model_instance': model}, metrics, iteration, output_dir)
    return model, metrics

def correr_decision_tree(df, vars, target, output_dir):
    X = df[vars]
    y = df[target]
    X = X.dropna()
    y = y.loc[X.index]
    # Preprocesamiento
    X_processed, preprocessor = preprocesar_datos(X)
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2)
    # Grilla de hiperparámetros
    param_distributions = {
        'max_depth': [int(x) for x in np.linspace(5, 50, 10)],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }
    dt = DecisionTreeRegressor()
    random_search = RandomizedSearchCV(dt, param_distributions, n_iter=10, scoring='neg_mean_squared_error', cv=3, random_state=42)
    random_search.fit(X_train, y_train)
    # Iterar sobre los resultados y guardar
    for i, params in enumerate(random_search.cv_results_['params']):
        model, metrics = modelo_decision_tree(X_train, y_train, X_test, y_test, params, i, output_dir)
    # Guardar el mejor modelo
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    metrics = calcula_metricas(y_test, best_model.predict(X_test))
    guardar_resultados('DecisionTree_Best', {**best_params, 'model_instance': best_model}, metrics, 'best', output_dir)
    return best_model, metrics


def modelo_bagging(X_train, y_train, X_test, y_test, params, iteration, output_dir):
    model = BaggingRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calcula_metricas(y_test, y_pred)
    # Guardar resultados
    guardar_resultados('Bagging', {**params, 'model_instance': model}, metrics, iteration, output_dir)
    return model, metrics

def correr_bagging(df, vars, target, output_dir):
    X = df[vars]
    y = df[target]
    X = X.dropna()
    y = y.loc[X.index]
    # Preprocesamiento
    X_processed, preprocessor = preprocesar_datos(X)
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2)
    # Grilla de hiperparámetros
    param_distributions = {
        'n_estimators': [10, 50, 100, 200],
        'max_samples': [0.5, 0.7, 1.0],
        'max_features': [0.5, 0.7, 1.0],
        'bootstrap': [True, False],
        'bootstrap_features': [True, False]
    }
    bagging = BaggingRegressor()
    random_search = RandomizedSearchCV(bagging, param_distributions, n_iter=10, scoring='neg_mean_squared_error', cv=3, random_state=42)
    random_search.fit(X_train, y_train)
    # Iterar sobre los resultados y guardar
    for i, params in enumerate(random_search.cv_results_['params']):
        model, metrics = modelo_bagging(X_train, y_train, X_test, y_test, params, i, output_dir)
    # Guardar el mejor modelo
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    metrics = calcula_metricas(y_test, best_model.predict(X_test))
    guardar_resultados('Bagging_Best', {**best_params, 'model_instance': best_model}, metrics, 'best', output_dir)
    return best_model, metrics

def modelo_xgboost(X_train, y_train, X_test, y_test, params, iteration, output_dir):
    model = XGBRegressor(**params, objective='reg:squarederror')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calcula_metricas(y_test, y_pred)
    # Guardar resultados
    guardar_resultados('XGBoost', {**params, 'model_instance': model}, metrics, iteration, output_dir)
    return model, metrics

def correr_xgboost(df, vars, target, output_dir):
    X = df[vars]
    y = df[target]
    X = X.dropna()
    y = y.loc[X.index]
    # Preprocesamiento
    X_processed, preprocessor = preprocesar_datos(X)
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2)
    # Grilla de hiperparámetros
    param_distributions = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }
    xgb = XGBRegressor(objective='reg:squarederror')
    random_search = RandomizedSearchCV(xgb, param_distributions, n_iter=10, scoring='neg_mean_squared_error', cv=3, random_state=42)
    random_search.fit(X_train, y_train)
    # Iterar sobre los resultados y guardar
    for i, params in enumerate(random_search.cv_results_['params']):
        model, metrics = modelo_xgboost(X_train, y_train, X_test, y_test, params, i, output_dir)
    # Guardar el mejor modelo
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    metrics = calcula_metricas(y_test, best_model.predict(X_test))
    guardar_resultados('XGBoost_Best', {**best_params, 'model_instance': best_model}, metrics, 'best', output_dir)
    return best_model, metrics

def modelo_lightgbm(X_train, y_train, X_test, y_test, params, iteration, output_dir):
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calcula_metricas(y_test, y_pred)
    # Guardar resultados
    guardar_resultados('LightGBM', {**params, 'model_instance': model}, metrics, iteration, output_dir)
    return model, metrics

def correr_lightgbm(df, vars, target, output_dir):
    X = df[vars]
    y = df[target]
    X = X.dropna()
    y = y.loc[X.index]
    # Preprocesamiento
    X_processed, preprocessor = preprocesar_datos(X)
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2)
    # Grilla de hiperparámetros
    param_distributions = {
        'n_estimators': [100, 200, 500],
        'max_depth': [-1, 5, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 100],
        'subsample': [0.7, 0.8, 1.0]
    }
    lgbm = LGBMRegressor()
    random_search = RandomizedSearchCV(lgbm, param_distributions, n_iter=10, scoring='neg_mean_squared_error', cv=3, random_state=42)
    random_search.fit(X_train, y_train)
    # Iterar sobre los resultados y guardar
    for i, params in enumerate(random_search.cv_results_['params']):
        model, metrics = modelo_lightgbm(X_train, y_train, X_test, y_test, params, i, output_dir)
    # Guardar el mejor modelo
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    metrics = calcula_metricas(y_test, best_model.predict(X_test))
    guardar_resultados('LightGBM_Best', {**best_params, 'model_instance': best_model}, metrics, 'best', output_dir)
    return best_model, metrics


def modelo_mlp(X_train, y_train, X_test, y_test, params, iteration, output_dir):
    model = MLPRegressor(**params, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calcula_metricas(y_test, y_pred)
    # Guardar resultados
    guardar_resultados('MLPRegressor', {**params, 'model_instance': model}, metrics, iteration, output_dir)
    return model, metrics
def correr_mlp(df, vars, target, output_dir):
    X = df[vars]
    y = df[target]
    X = X.dropna()
    y = y.loc[X.index]
    # Preprocesamiento
    X_processed, preprocessor = preprocesar_datos(X)
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2)
    # Grilla de hiperparámetros
    param_distributions = {
        'hidden_layer_sizes': [(50,), (100,), (50,50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
    mlp = MLPRegressor(max_iter=1000)
    random_search = RandomizedSearchCV(mlp, param_distributions, n_iter=10, scoring='neg_mean_squared_error', cv=3, random_state=42)
    random_search.fit(X_train, y_train)
    # Iterar sobre los resultados y guardar
    for i, params in enumerate(random_search.cv_results_['params']):
        model, metrics = modelo_mlp(X_train, y_train, X_test, y_test, params, i, output_dir)
    # Guardar el mejor modelo
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    metrics = calcula_metricas(y_test, best_model.predict(X_test))
    guardar_resultados('MLPRegressor_Best', {**best_params, 'model_instance': best_model}, metrics, 'best', output_dir)
    return best_model, metrics

def modelo_knn(X_train, y_train, X_test, y_test, params, iteration, output_dir):
    model = KNeighborsRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calcula_metricas(y_test, y_pred)
    # Guardar resultados
    guardar_resultados('KNN', {**params, 'model_instance': model}, metrics, iteration, output_dir)
    return model, metrics

def correr_knn(df, vars, target, output_dir):
    X = df[vars]
    y = df[target]
    X = X.dropna()
    y = y.loc[X.index]
    # Preprocesamiento
    X_processed, preprocessor = preprocesar_datos(X)
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2)
    # Grilla de hiperparámetros
    param_distributions = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # Distancia Manhattan (p=1) y Euclídea (p=2)
    }
    knn = KNeighborsRegressor()
    random_search = RandomizedSearchCV(knn, param_distributions, n_iter=10, scoring='neg_mean_squared_error', cv=3, random_state=42)
    random_search.fit(X_train, y_train)
    # Iterar sobre los resultados y guardar
    for i, params in enumerate(random_search.cv_results_['params']):
        model, metrics = modelo_knn(X_train, y_train, X_test, y_test, params, i, output_dir)
    # Guardar el mejor modelo
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    metrics = calcula_metricas(y_test, best_model.predict(X_test))
    guardar_resultados('KNN_Best', {**best_params, 'model_instance': best_model}, metrics, 'best', output_dir)
    return best_model, metrics

def modelo_xgboost_quantile(X_train, y_train, X_test, y_test, params, iteration, output_dir, quantile):
    # Definir la función de pérdida cuantil
    def quantile_loss(y_true, y_pred):
        errors = y_true - y_pred
        return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))
    
    model = XGBRegressor(**params, objective='reg:squarederror')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {'QuantileLoss': quantile_loss(y_test, y_pred)}
    # Guardar resultados
    guardar_resultados('XGBoostQuantile', {**params, 'model_instance': model, 'quantile': quantile}, metrics, iteration, output_dir)
    return model, metrics
def correr_xgboost_quantile(df, vars, target, output_dir, quantile=0.5):
    X = df[vars]
    y = df[target]
    X = X.dropna()
    y = y.loc[X.index]
    # Preprocesamiento
    X_processed, preprocessor = preprocesar_datos(X)
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2)
    # Grilla de hiperparámetros
    param_distributions = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    xgb = XGBRegressor(objective='reg:squarederror')
    random_search = RandomizedSearchCV(xgb, param_distributions, n_iter=5, scoring='neg_mean_squared_error', cv=3, random_state=42)
    random_search.fit(X_train, y_train)
    # Iterar sobre los resultados y guardar
    for i, params in enumerate(random_search.cv_results_['params']):
        model, metrics = modelo_xgboost_quantile(X_train, y_train, X_test, y_test, params, i, output_dir, quantile)
    # Guardar el mejor modelo
    best_model = random_search.best_estimator_
    metrics = {'QuantileLoss': None}  # Puedes calcular la pérdida cuantil para el mejor modelo si lo deseas
    guardar_resultados('XGBoostQuantile_Best', {**best_model.get_params(), 'model_instance': best_model, 'quantile': quantile}, metrics, 'best', output_dir)
    return best_model, metrics


def modelo_lasso_quantile(X_train, y_train, X_test, y_test, params, iteration, output_dir, quantile):
    model = QuantileRegressor(quantile=quantile, **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calcula_metricas(y_test, y_pred)
    # Guardar resultados
    guardar_resultados('LassoQuantile', {**params, 'model_instance': model, 'quantile': quantile}, metrics, iteration, output_dir)
    return model, metrics

def correr_lasso_quantile(df, vars, target, output_dir, quantile=0.5):
    X = df[vars]
    y = df[target]
    X = X.dropna()
    y = y.loc[X.index]
    # Preprocesamiento
    X_processed, preprocessor = preprocesar_datos(X)
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2)
    # Grilla de hiperparámetros
    param_distributions = {
        'alpha': np.logspace(-4, 0, 10),
        'solver': ['highs']
    }
    quantile_reg = QuantileRegressor(quantile=quantile)
    random_search = RandomizedSearchCV(quantile_reg, param_distributions, n_iter=5, scoring='neg_mean_squared_error', cv=3, random_state=42)
    random_search.fit(X_train, y_train)
    # Iterar sobre los resultados y guardar
    for i, params in enumerate(random_search.cv_results_['params']):
        model, metrics = modelo_lasso_quantile(X_train, y_train, X_test, y_test, params, i, output_dir, quantile)
    # Guardar el mejor modelo
    best_model = random_search.best_estimator_
    metrics = calcula_metricas(y_test, best_model.predict(X_test))
    guardar_resultados('LassoQuantile_Best', {**best_model.get_params(), 'model_instance': best_model, 'quantile': quantile}, metrics, 'best', output_dir)
    return best_model, metrics


def modelo_lasso(X_train, y_train, X_test, y_test, params, iteration, output_dir):
    model = Lasso(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calcula_metricas(y_test, y_pred)
    # Guardar resultados
    guardar_resultados('Lasso', {'alpha': params.get('alpha'), 'model_instance': model}, metrics, iteration, output_dir)
    return model, metrics
def correr_random_forest(df, vars, target, output_dir):
    X = df[vars]
    y = df[target]
    X = X.dropna()
    y = y.loc[X.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    param_distributions = {
        'n_estimators': [int(x) for x in np.linspace(100, 1000, 10)],
        'max_depth': [int(x) for x in np.linspace(10, 100, 10)],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor()
    random_search = RandomizedSearchCV(rf, param_distributions, n_iter=10, scoring='neg_mean_squared_error', cv=3, random_state=42)
    random_search.fit(X_train, y_train)
    for i, params in enumerate(random_search.cv_results_['params']):
        model, metrics = modelo_random_forest(X_train, y_train, X_test, y_test, params, i, output_dir)
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    metrics = calcula_metricas(y_test, best_model.predict(X_test))
    guardar_resultados('RandomForest_Best', {**best_params, 'model_instance': best_model}, metrics, 'best', output_dir)
    return best_model, metrics

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
    correr_random_forest(df, vars, target, output_dir)


if __name__ == "__main__":
    main()