import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import kaleido

# Leer el DataFrame limpio
df = pd.read_csv(r'D:\Tesis\base_limpia_tesis\data_clean_v3.csv')

# Variables predefinidas
banos = 'full_bathrooms'
dormitorios = 'bedrooms'
ambientes = "rooms"
superficie_total = "superficie_total"
superficie_cubierta = "superficie_cubierta"
precio = 'price'
ln_precio = 'ln_precio_USD'
precio_metro = 'precio_m2'
tipo_propiedad = 'property_type'
operacion = 'operation'
estado = 'location_state'
barrio = 'location_city'
variable_objetivo = ln_precio  # O 'precio_m2'

# Ruta del archivo Excel con las listas de variables
ruta_excel_variables = 'variables_numericas_categoricas.xlsx'

# Leer las listas de variables desde el archivo Excel
variables_df = pd.read_excel(ruta_excel_variables)

# Obtener las listas de variables numéricas y categóricas
numericas = variables_df['Variables Numéricas'].dropna().tolist()
categoricas = variables_df['Variables Categóricas'].dropna().tolist()

# Filtrar las variables que existen en el DataFrame
numericas = [var for var in numericas if var in df.columns]
categoricas = [var for var in categoricas if var in df.columns]

def main():
    # Ruta donde se guardarán los gráficos
    ruta_graficos = r'D:\Tesis\estadistica_descriptiva_3'

    # Verificar si la ruta existe, si no, crearla
    if not os.path.exists(ruta_graficos):
        os.makedirs(ruta_graficos)
        
    # Calcular precio por m2 si es necesario
    if precio_metro not in df.columns:
        df[precio_metro] = df[precio] / df[superficie_total]
    
    # Funciones para generar y guardar gráficos
    def graficar_histograma(df, variable, nbins=50, nombre_archivo=''):
        if variable in df.columns:
            try:
                fig = px.histogram(df, x=variable, nbins=nbins, title=f'Distribución de {variable}')
                fig.update_layout(
                    xaxis_title=variable,
                    yaxis_title='Frecuencia'
                )
                # Guardar gráfico
                fig.write_html(os.path.join(ruta_graficos, f'{nombre_archivo}.html'))
                fig.write_image(os.path.join(ruta_graficos, f'{nombre_archivo}.png'))
                print(f"Histograma '{nombre_archivo}' guardado correctamente.")
            except Exception as e:
                print(f"Error al generar o guardar el histograma '{nombre_archivo}': {e}")
        else:
            print(f"La variable '{variable}' no existe en el DataFrame.")

    def graficar_barra(df, variable, nombre_archivo=''):
        if variable in df.columns:
            try:
                conteo = df[variable].value_counts().nlargest(20).reset_index()  # Top 20 categorías
                conteo.columns = [variable, 'Frecuencia']
                fig = px.bar(conteo, x=variable, y='Frecuencia', title=f'Distribución de {variable}')
                fig.update_layout(xaxis_title=variable, yaxis_title='Frecuencia')
                # Guardar gráfico
                fig.write_html(os.path.join(ruta_graficos, f'{nombre_archivo}.html'))
                fig.write_image(os.path.join(ruta_graficos, f'{nombre_archivo}.png'))
                print(f"Gráfico de barras '{nombre_archivo}' guardado correctamente.")
            except Exception as e:
                print(f"Error al generar o guardar el gráfico de barras '{nombre_archivo}': {e}")
        else:
            print(f"La variable '{variable}' no existe en el DataFrame.")

    def graficar_dispersion(df, x_var, y_var, titulo=None, nombre_archivo=''):
        if x_var in df.columns and y_var in df.columns:
            try:
                if not titulo:
                    titulo = f'{x_var} vs {y_var}'
                hover_data = [tipo_propiedad] if tipo_propiedad in df.columns else None

                fig = px.scatter(df, x=x_var, y=y_var, title=titulo, trendline='ols', hover_data=hover_data)
                fig.update_layout(
                    xaxis_title=x_var,
                    yaxis_title=y_var
                )
                # Guardar gráfico
                fig.write_html(os.path.join(ruta_graficos, f'{nombre_archivo}.html'))
                fig.write_image(os.path.join(ruta_graficos, f'{nombre_archivo}.png'))
                print(f"Gráfico de dispersión '{nombre_archivo}' guardado correctamente.")
            except Exception as e:
                print(f"Error al generar o guardar el gráfico de dispersión '{nombre_archivo}': {e}")
        else:
            print(f"Una o ambas variables '{x_var}' y '{y_var}' no existen en el DataFrame.")

    # Generar y guardar gráficos para variables numéricas
    for variable in numericas:
        print(f"Generando histograma para la variable numérica: {variable}")
        # Histograma
        graficar_histograma(df, variable, nombre_archivo=f'histograma_{variable}')
        
        # Gráfico de dispersión con la variable objetivo (si es diferente)
        if variable != variable_objetivo:
            print(f"Generando gráfico de dispersión para: {variable} vs {variable_objetivo}")
            graficar_dispersion(df, variable, variable_objetivo, nombre_archivo=f'dispersion_{variable}_vs_{variable_objetivo}')
    
    # Generar y guardar gráficos para variables categóricas
    for variable in categoricas:
        print(f"Generando gráfico de barras para la variable categórica: {variable}")
        graficar_barra(df, variable, nombre_archivo=f'barra_{variable}')
    
    # Generar tabla de estadísticas descriptivas para variables numéricas
    if numericas:
        print("Generando tabla de estadísticas descriptivas.")
        estadisticas_personalizadas = df[numericas].describe(percentiles=[0.1, 0.5, 0.9]).T
        estadisticas_personalizadas = estadisticas_personalizadas[['mean', 'std', 'min', '10%', '50%', '90%', 'max']]
        estadisticas_personalizadas.rename(columns={'10%': 'p10', '50%': 'median', '90%': 'p90'}, inplace=True)
        
        # Mostrar y guardar tabla interactiva
        try:
            fig = go.Figure(data=[go.Table(
                header=dict(values=['Variable'] + list(estadisticas_personalizadas.columns),
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(values=[estadisticas_personalizadas.index] + [estadisticas_personalizadas[col] for col in estadisticas_personalizadas.columns],
                           fill_color='lavender',
                           align='left'))
            ])
            
            fig.update_layout(title='Estadísticas Descriptivas')
            # Guardar tabla
            fig.write_html(os.path.join(ruta_graficos, 'tabla_estadisticas_descriptivas.html'))
            print("Tabla de estadísticas descriptivas guardada correctamente.")
            # fig.show()
        except Exception as e:
            print(f"Error al generar o guardar la tabla de estadísticas descriptivas: {e}")
        
        # También guardar la tabla en formato CSV
        try:
            estadisticas_personalizadas.to_csv(os.path.join(ruta_graficos, 'estadisticas_descriptivas.csv'), index=True)
            print("Tabla de estadísticas descriptivas guardada en CSV correctamente.")
        except Exception as e:
            print(f"Error al guardar la tabla de estadísticas descriptivas en CSV: {e}")
    else:
        print("No hay variables numéricas para generar estadísticas descriptivas.")

if __name__ == "__main__":
    main()
