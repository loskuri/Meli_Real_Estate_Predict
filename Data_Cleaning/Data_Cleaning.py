import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import numpy as np
import time
import re
import pytz

tipo_propiedad = 'property_type'
operacion = 'operation'
precio = 'price'
moneda = 'currency_id'
precio_usd = 'price_USD'
ln_precio_usd = 'ln_precio_USD'
fecha_comparacion = '2024-11-10'
fecha_creacion = 'date_created'
descripcion = 'description'
titulo = 'title'
path_categorica_numerica = 'categoricas_numericas' + ".csv"
superficie_cubierta = 'covered_area'
superficie_total = 'total_area'
estado = 'location_state'
tasa_conversion=1100
fecha_referencia = '2024-10-17'
path_output_csv = r'D:\Tesis\base_limpia_tesis\data_clean_v3.csv'

df = pd.read_csv(r'D:\Tesis\base_acumulada\data.csv')
df = df.drop_duplicates()
df = df[df[estado].isin(["Capital Federal", "Bs.As. G.B.A. Norte", "Bs.As. G.B.A. Sur", "Bs.As. G.B.A. Oeste"])]

df = df[df[tipo_propiedad].isin(["Departamento","Casa","Ph"])]
df = df[df[operacion].isin(["Venta","venta","Ventas","Ventas"])]

keywords = ["contruccion", "construcción", "pozo","entrega"]


def convertir_precio_a_usd(df, tasa_conversion=1150, eliminar_columna_precio=False, columna_precio=precio):
    """
    Función que convierte los precios en ARS a USD en base a una tasa de conversión.
    
    Parámetros:
        - df: DataFrame con las columnas 'currency_id' y la columna de precios.
        - tasa_conversion: Tasa de conversión de ARS a USD (por defecto 1100).
        - eliminar_columna_precio: Booleano para eliminar la columna original de precios si es True (por defecto False).
        - columna_precio: Nombre de la columna que contiene los precios (por defecto 'price').
    
    Retorna:
        - DataFrame con la columna 'price_USD'.
    """
    # Verificar si las columnas existen en el DataFrame
    columnas_necesarias = [moneda, columna_precio]
    for col in columnas_necesarias:
        if col not in df.columns:
            print(f"La columna '{col}' no existe en el DataFrame.")
            return df
    
    # Verificar si hay valores nulos en 'currency_id' y en la columna de precios
    print(df[[moneda, columna_precio]].isnull().sum())
    
    # Convertir la columna de precios a numérico
    df[columna_precio] = pd.to_numeric(df[columna_precio], errors='coerce')
    
    # Crear la columna 'price_USD' basada en 'currency_id'
    df[precio_usd] = np.where(
        df[moneda] == 'ARS',
        df[columna_precio] / tasa_conversion,
        df[columna_precio])
    
    if eliminar_columna_precio:
        df = df.drop(columna_precio, axis=1)
    
    return df

def agregar_columna_ln_precio(df):
    """
    Agrega una columna al DataFrame con el logaritmo natural del precio en USD.

    Retorna:
    - DataFrame con la nueva columna agregada.
    """
    # Nombres de las columnas
    columna_precio = precio_usd
    nueva_columna = ln_precio_usd

    # Verificar si la columna de precio existe en el DataFrame
    if columna_precio not in df.columns:
        print(f"Error: La columna '{columna_precio}' no existe en el DataFrame.")
        return df

    # Convertir la columna de precios a numérico
    df[columna_precio] = pd.to_numeric(df[columna_precio], errors='coerce')

    # Reemplazar valores no positivos (<=0) o nulos por NaN
    df[columna_precio] = df[columna_precio].replace([np.inf, -np.inf], np.nan)
    df[columna_precio] = df[columna_precio].where(df[columna_precio] > 0)

    # Calcular el logaritmo natural de los precios positivos
    df[nueva_columna] = np.log(df[columna_precio])

    # Verificar si la nueva columna se ha creado correctamente
    if nueva_columna in df.columns:
        print(f"La columna '{nueva_columna}' se ha creado correctamente.")
    else:
        print(f"Error: La columna '{nueva_columna}' no se pudo crear.")

    return df
def eliminar_columnas_con_valor_dominante(df, porcentaje_dominante=0.95):
    """
    Elimina columnas donde un solo valor (excluyendo nulos) representa un porcentaje igual o superior al especificado.

    Parámetros:
    - df (DataFrame): El DataFrame original.
    - porcentaje_dominante (float): Porcentaje mínimo para considerar que un valor domina la columna (default = 0.95).

    Retorna:
    - DataFrame modificado sin las columnas donde un valor domina.
    """
    columnas_a_eliminar = []

    for col in df.columns:
        # Excluir valores nulos
        serie_sin_nulos = df[col].dropna()
        total_sin_nulos = len(serie_sin_nulos)
        
        if total_sin_nulos == 0:
            continue  # O puedes decidir eliminar columnas que solo tienen nulos

        # Calcular la frecuencia del valor más común
        conteo_valores = serie_sin_nulos.value_counts()
        valor_mas_frecuente = conteo_valores.index[0]
        frecuencia_mas_frecuente = conteo_valores.iloc[0]
        porcentaje_mas_frecuente = frecuencia_mas_frecuente / total_sin_nulos

        if porcentaje_mas_frecuente >= porcentaje_dominante:
            columnas_a_eliminar.append(col)
            print(f"Columna '{col}' eliminada: el valor '{valor_mas_frecuente}' representa el {porcentaje_mas_frecuente:.2%} de los datos no nulos.")

    df = df.drop(columns=columnas_a_eliminar)
    print(f"\nSe han eliminado las siguientes columnas por tener un valor dominante:\n{columnas_a_eliminar}")

    return df


def eliminar_columnas_con_nulos(df, umbral_nulos=0.9):
    """
    Elimina columnas con un porcentaje de valores nulos superior al umbral especificado.

    Parámetros:
    - df (DataFrame): El DataFrame a procesar.
    - umbral_nulos (float): Umbral de porcentaje de valores nulos para eliminar una columna (por defecto = 0.9).

    Retorna:
    - DataFrame modificado sin las columnas con más del umbral de valores nulos.
    """
    # Calcular el porcentaje de valores nulos por columna
    porcentaje_nulos = df.isnull().mean()  # Esto da el porcentaje de nulos en cada columna.

    # Identificar columnas a eliminar con base en el umbral definido
    columnas_a_eliminar = porcentaje_nulos[porcentaje_nulos > umbral_nulos].index

    # Eliminar las columnas identificadas
    df = df.drop(columns=columnas_a_eliminar)

    return df


def eliminar_filas_con_palabras_clave(df,keywords):
    """
    Elimina filas del DataFrame donde se encuentren las palabras clave en las columnas especificadas.

    Parámetros:
    - df: DataFrame de pandas del cual se eliminarán las filas.
    - keywords: Lista de palabras clave a buscar.
    - columnas_busqueda: Lista de columnas donde buscar las palabras clave.

    Retorna:
    - DataFrame sin las filas que contienen las palabras clave en las columnas especificadas.
    """

    columnas_busqueda = [descripcion, titulo]

    # Crear un patrón de expresión regular que incluya todas las palabras clave
    pattern = r'(' + '|'.join(keywords) + r')'

    # Crear una máscara booleana que es True si la fila contiene alguna palabra clave en cualquiera de las columnas
    mask = pd.Series(False, index=df.index)
    for columna in columnas_busqueda:
        if columna in df.columns:
            # Actualizar la máscara
            mask |= df[columna].astype(str).str.contains(pattern, flags=re.IGNORECASE, na=False)
        else:
            print(f"La columna '{columna}' no existe en el DataFrame.")

    # Invertir la máscara para seleccionar las filas que NO contienen las palabras clave
    df_sin_filas = df[~mask]

    return df_sin_filas


def convertir_superficie(valor):
    """
    Convierte una cadena de texto que representa la superficie con unidad a un valor numérico en m².
    
    Parámetros:
    - valor (str): La cadena que contiene el valor y la unidad (ej. "300 m²", "329400 ha").
    
    Retorna:
    - float: El valor numérico en m².
    - np.nan: Si el valor no se puede convertir.
    """
    try:
        # Asegurarse de que el valor sea una cadena y eliminar espacios
        valor = str(valor).strip().lower()
        
        # Expresión regular para extraer el número y la unidad (permitiendo algunos formatos adicionales)
        match = re.match(r"([\d.,]+)\s*(m2|m²|ha|metros|hectareas|hectáreas)", valor)
        
        if match:
            numero = match.group(1)
            unidad = match.group(2)
            
            # Reemplazar comas por puntos y convertir a float
            numero = float(numero.replace(',', '.'))
            
            if unidad in ['ha', 'hectareas', 'hectáreas']:
                # 1 hectárea = 10,000 m²
                return numero * 10000
            elif unidad in ['m2', 'm²', 'metros']:
                return numero
        else:
            # Si no coincide con el patrón, mostrar advertencia y retornar NaN
            print(f"Advertencia: No se pudo convertir el valor '{valor}'")
            return np.nan
    except Exception as e:
        # En caso de cualquier error, mostrar el error y retornar NaN
        print(f"Error al convertir el valor '{valor}': {e}")
        return np.nan

def calcular_dias_desde_fecha(df, columna_fecha=fecha_creacion, fecha_referencia=fecha_comparacion):
    """
    Calcula la cantidad de días desde la fecha en `columna_fecha` hasta `fecha_referencia` 
    y agrega una nueva columna `dias_desde_fecha`.

    Parámetros:
    - df (DataFrame): El DataFrame a procesar.
    - columna_fecha (str): El nombre de la columna con las fechas a calcular (por defecto 'date_created').
    - fecha_referencia (str): La fecha de referencia en formato 'YYYY-MM-DD' (por defecto '2024-10-17').

    Retorna:
    - DataFrame con una nueva columna `dias_desde_fecha` que contiene los días transcurridos.
    """
    # Convertir la columna de fechas al formato datetime, removiendo las zonas horarias si existen
    df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors='coerce').dt.tz_localize(None)
    
    # Convertir la fecha de referencia al formato datetime sin zona horaria
    fecha_referencia = pd.to_datetime(fecha_referencia).tz_localize(None)
    
    # Calcular la diferencia en días y agregar como nueva columna
    df['dias_desde_fecha'] = (fecha_referencia - df[columna_fecha]).dt.days

    return df
def detectar_variables(df, max_categories=10, drop_original=True, output_path=path_categorica_numerica, excel_output_path='variables_numericas_categoricas.xlsx'):
    """
    Identifica variables numéricas y categóricas, aplica One-Hot Encoding a las categóricas filtradas,
    y exporta el DataFrame resultante a un archivo CSV. Además, exporta los nombres de las columnas
    numéricas y categóricas a un archivo Excel en columnas separadas.

    Parámetros:
    - df: DataFrame de pandas.
    - max_categories: Número máximo de categorías permitidas para una variable categórica (default=10).
    - drop_original: Booleano que indica si se deben eliminar las columnas categóricas originales después de la codificación.
    - output_path: Ruta donde se guardará el archivo CSV exportado.
    - excel_output_path: Ruta donde se guardará el archivo Excel con los nombres de las variables.

    Retorna:
    - df_mod: DataFrame modificado con las columnas numéricas y One-Hot Encoding aplicadas.
    - numericas: Lista de nombres de variables numéricas.
    - categoricas_filtradas: Lista de nombres de variables categóricas con <= max_categories categorías.
    - categoricas_excluidas: Lista de variables categóricas que fueron excluidas por exceder el límite de categorías.
    """
    import pandas as pd

    # Trabajar sobre una copia para no modificar el DataFrame original
    df_mod = df.copy()

    # Paso 1: Identificar variables numéricas y categóricas
    numericas = df_mod.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categoricas = df_mod.select_dtypes(include=['object', 'category']).columns.tolist()

    categoricas_filtradas = []
    categoricas_excluidas = []

    for col in categoricas:
        num_categorias = df_mod[col].nunique()
        if num_categorias <= max_categories:
            categoricas_filtradas.append(col)
        else:
            categoricas_excluidas.append(col)

    # Imprimir las variables identificadas
    print("Variables numéricas:", numericas)
    print(f"Variables categóricas con <= {max_categories} categorías:", categoricas_filtradas)
    if categoricas_excluidas:
        print(f"Variables categóricas excluidas por exceder {max_categories} categorías:", categoricas_excluidas)

    # Paso 2: Aplicar One-Hot Encoding a las categóricas filtradas
    if categoricas_filtradas:
        dummies = pd.get_dummies(df_mod[categoricas_filtradas], prefix=categoricas_filtradas, drop_first=False)
        df_mod = pd.concat([df_mod, dummies], axis=1)
        print(f"One-Hot Encoding realizado en las columnas: {categoricas_filtradas}")

        if drop_original:
            df_mod.drop(columns=categoricas_filtradas, inplace=True)
            print(f"Columnas categóricas originales eliminadas: {categoricas_filtradas}")
    else:
        print("No hay columnas categóricas para realizar One-Hot Encoding.")

    # Paso 3: Eliminar columnas categóricas excluidas (opcional)
    if categoricas_excluidas:
        df_mod.drop(columns=categoricas_excluidas, inplace=True)
        print(f"Columnas categóricas excluidas eliminadas del DataFrame: {categoricas_excluidas}")

    # Paso 4: Exportar el DataFrame modificado a CSV
    df_mod.to_csv(output_path, index=False)
    print(f"DataFrame exportado a {output_path}")

    # **Nuevo**: Exportar nombres de variables numéricas y categóricas a Excel
    # Crear un DataFrame con dos columnas
    variables_df = pd.DataFrame({
        'Variables Numéricas': pd.Series(numericas),
        'Variables Categóricas': pd.Series(categoricas_filtradas)
    })

    # Exportar a Excel
    variables_df.to_excel(excel_output_path, index=False)
    print(f"Nombres de variables exportados a {excel_output_path}")

    # Retornar el DataFrame modificado y las listas
    return df_mod, numericas, categoricas_filtradas, categoricas_excluidas

def eliminar_anomalias_columna(df, columna):
    """
    Modifica una columna en el DataFrame, eliminando valores atípicos usando el método del rango intercuartílico (IQR).
    
    Parámetros:
    df (DataFrame): DataFrame en el cual se eliminarán los valores atípicos.
    columna (str): Nombre de la columna en la que se identificarán y eliminarán los valores atípicos.
    
    Retorna:
    DataFrame: DataFrame con la columna modificada y los valores atípicos reemplazados con NaN.
    """
    # Verificar que la columna exista en el DataFrame
    if columna not in df.columns:
        raise KeyError(f"La columna '{columna}' no existe en el DataFrame.")
    
    # Convertir la columna a tipo numérico, reemplazando valores no convertibles con NaN
    df[columna] = pd.to_numeric(df[columna], errors='coerce')
    
    # Calcular el primer y tercer cuartil
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    
    # Calcular el rango intercuartílico (IQR)
    IQR = Q3 - Q1
    
    # Definir límites para valores no atípicos
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Reemplazar valores atípicos con NaN en la columna especificada
    df[columna] = df[columna].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)
    
    return df

# Ejemplo de uso

#---------------------------------------------------------------

def main(df):
    df['superficie_cubierta'] = df[superficie_cubierta].apply(convertir_superficie)
    df['superficie_total'] = df[superficie_total].apply(convertir_superficie)
    df = eliminar_anomalias_columna(df, 'superficie_total')
    df = eliminar_anomalias_columna(df, 'superficie_cubierta')
    df = eliminar_anomalias_columna(df, 'rooms')

    df = calcular_dias_desde_fecha(df, columna_fecha='date_created', fecha_referencia=fecha_referencia)
    df = convertir_precio_a_usd(df,tasa_conversion=tasa_conversion,eliminar_columna_precio=False,columna_precio='price')
    df = agregar_columna_ln_precio(df)
    df = eliminar_columnas_con_valor_dominante(df, porcentaje_dominante=0.95)
    df = eliminar_filas_con_palabras_clave(df, keywords)
    df = eliminar_columnas_con_nulos(df, umbral_nulos=0.9)
    df, numericas, categoricas_filtradas, categoricas_excluidas = detectar_variables(
    df,max_categories=10,output_path='dataframe_modificado.csv',excel_output_path='variables_numericas_categoricas.xlsx') 
    df.to_csv(path_output_csv, index=False)


if __name__ == "__main__":
    main(df)