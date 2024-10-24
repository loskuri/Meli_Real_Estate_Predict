import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import numpy as np
import time
import re
import pytz


df = pd.read_csv(r'D:\Tesis\bases_datos\real_estate.csv')
df = df.drop_duplicates()
df = df[df["location_state_name"].isin(["Capital Federal", "Bs.As. G.B.A. Norte", "Bs.As. G.B.A. Sur", "Bs.As. G.B.A. Oeste"])]

df = df[df["attribute_property_type"].isin(["Departamento","Casa","Ph"])]

keywords = ["contruccion", "construcción", "pozo","entrega"]

def convertir_precio_a_usd(df, tasa_conversion=1150, eliminar_columna_precio=False, columna_precio='price'):
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
    columnas_necesarias = ['currency_id', columna_precio]
    for col in columnas_necesarias:
        if col not in df.columns:
            print(f"La columna '{col}' no existe en el DataFrame.")
            return df
    
    # Verificar si hay valores nulos en 'currency_id' y en la columna de precios
    print(df[['currency_id', columna_precio]].isnull().sum())
    
    # Convertir la columna de precios a numérico
    df[columna_precio] = pd.to_numeric(df[columna_precio], errors='coerce')
    
    # Crear la columna 'price_USD' basada en 'currency_id'
    df['price_USD'] = np.where(
        df['currency_id'] == 'ARS',
        df[columna_precio] / tasa_conversion,
        df[columna_precio]
    )
    
    # Verificar los cambios
    print("\nPrimeras filas con la columna 'price_USD':")
    print(df[['currency_id', columna_precio, 'price_USD']].head())
    
    # Eliminar la columna de precios original si se indica
    if eliminar_columna_precio:
        df = df.drop(columna_precio, axis=1)
    
    return df

import numpy as np
import pandas as pd  # Asegúrate de tener pandas importado

def agregar_columna_ln_precio(df):
    """
    Agrega una columna al DataFrame con el logaritmo natural del precio en USD.

    Retorna:
    - DataFrame con la nueva columna agregada.
    """
    # Nombres de las columnas
    columna_precio = 'price_USD'
    nueva_columna = 'ln_precio_USD'

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

def reemplazar_valores_porcentuales(df, umbral=5.0, porcentaje_minimo=10.0):
    """
    Reemplaza valores en cada columna de un DataFrame que representan menos del umbral especificado,
    pero solo en aquellas columnas donde al menos un valor representa el `porcentaje_minimo` o más.

    Parámetros:
    - df (DataFrame): El DataFrame original.
    - umbral (float): Umbral de porcentaje mínimo para mantener un valor (default = 5.0%).
    - porcentaje_minimo (float): Mínimo porcentaje para que una columna sea considerada para el reemplazo (default = 10.0%).

    Retorna:
    - DataFrame modificado con valores de baja representación convertidos a NaN.
    """
    # Recorrer cada columna del DataFrame
    for col in df.columns:
        # Calcular la distribución porcentual de los valores
        distribucion_porcentual = df[col].value_counts(normalize=True).mul(100)
        
        # Verificar si algún valor cumple con el porcentaje mínimo requerido
        if (distribucion_porcentual >= porcentaje_minimo).any():
            # Identificar los valores que representan menos del umbral especificado
            valores_a_convertir = distribucion_porcentual[distribucion_porcentual < umbral].index
            
            # Reemplazar esos valores con NaN en el DataFrame
            df[col] = df[col].apply(lambda x: np.nan if x in valores_a_convertir else x)

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

import re
import pandas as pd  # Asegúrate de tener pandas importado
keywords = ["contruccion", "construcción", "pozo","entrega"]

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
    columnas_busqueda = ['description', 'title']

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


def calcular_dias_desde_fecha(df, columna_fecha='date_created', fecha_referencia='2024-10-17'):
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


#---------------------------------------------------------------


def main(df):
    df['superficie_total'] = df['attribute_superficie_total'].apply(convertir_superficie)
    df = calcular_dias_desde_fecha(df, columna_fecha='date_created', fecha_referencia='2024-10-17')
    df = convertir_precio_a_usd(df,tasa_conversion=1100,eliminar_columna_precio=False,columna_precio='price')
    df = agregar_columna_ln_precio(df)
    df = reemplazar_valores_porcentuales(df, umbral=5.0, porcentaje_minimo=10.0)
    df = eliminar_filas_con_palabras_clave(df, keywords)
    df = eliminar_columnas_con_nulos(df, umbral_nulos=0.9)
    df.to_csv(r'D:\Tesis\bases_datos\db_real_estate_cleaned.csv', index=False)


if __name__ == "__main__":
    main(df)