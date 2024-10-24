import requests
import csv
import time
import os
import logging
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# CONFIGURACIÓN INICIAL
# =============================================================================

# Token de acceso a la API de Mercado Libre

# Nombre del archivo CSV donde se guardarán los datos
accumulate_filename = 'real_estate.csv'

# Nombre del archivo de control para filtros procesados
control_filename = 'real_estate_filters.xlsx'

# Provincias a procesar
provincias = [
    'AR-C',  # Ciudad Autónoma de Buenos Aires
    'AR-B',  # Buenos Aires (provincia)
]

# Categorías a procesar
categorias = ['MLA1459']  # Inmuebles

# Moneda a utilizar
currency = 'USD'

# Tamaño del límite de resultados por página (máximo permitido por la API es 50)
page_limit = 50

# Límite total de publicaciones a procesar por consulta
total_limit = 1000  # Máximo permitido por la API es 1000

# Número máximo de hilos para solicitudes concurrentes
max_workers = 10  # Ajusta este valor según tus necesidades y las políticas de la API

# Rango de precios
def generate_price_ranges(range_size=1000, num_ranges=500000):
    price_ranges = []
    for i in range(num_ranges):
        x = range_size * i
        y = range_size * (i + 1) - 1
        price_ranges.append((x, y))
    price_ranges.append((range_size * num_ranges, '*'))
    return price_ranges

price_ranges = generate_price_ranges()

# =============================================================================
# CONFIGURACIÓN DEL LOGGING
# =============================================================================

logging.basicConfig(
    filename='procesamiento.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def setup_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    return session

def flatten_item(item, description_text):
    flat_item = {}
    for key, value in item.items():
        if isinstance(value, dict):
            for subkey, subvalue in flatten_item(value, '').items():
                flat_item[f"{key}_{subkey}"] = subvalue
        elif isinstance(value, list):
            if key == 'attributes':
                for attribute in value:
                    attr_id = attribute.get('id', '').lower()
                    attr_name = attribute.get('name', '').lower().replace(' ', '_')
                    attr_value = attribute.get('value_name', 'null')
                    attr_value_id = attribute.get('value_id', '')

                    if attr_id == 'operation':
                        flat_item['attribute_operation'] = attr_value
                        flat_item['attribute_operation_id'] = attr_value_id
                    elif attr_id == 'property_type':
                        flat_item['attribute_property_type'] = attr_value
                    else:
                        flat_item[f"attribute_{attr_name}"] = attr_value
            elif key == 'pictures':
                flat_item['pictures_urls'] = ';'.join(pic.get('url', '') for pic in value)
            else:
                flat_item[key] = ';'.join(str(v) for v in value)
        else:
            flat_item[key] = value

    # Agregar descripción
    flat_item['description'] = description_text

    # Agregar información de financiamiento
    tags = item.get('tags', [])
    flat_item['financing'] = 'Sí' if 'financing' in tags else 'No'

    # Imprimir operación para depuración
    print(f"ID de propiedad: {item.get('id', '')} - Operación: {flat_item.get('attribute_operation', '')}")

    return flat_item

def extract_relevant_fields(item, description_text):
    return flatten_item(item, description_text)

def load_processed_filters():
    if os.path.exists(control_filename):
        try:
            df = pd.read_excel(control_filename)
            return set(tuple(x) for x in df[['category', 'province', 'price_range']].to_records(index=False))
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, pd.errors.ExcelFileError):
            print(f"El archivo '{control_filename}' no es válido. Será eliminado.")
            os.remove(control_filename)
    return set()

def save_processed_filter(category, province, price_range):
    if os.path.exists(control_filename):
        try:
            df = pd.read_excel(control_filename)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, pd.errors.ExcelFileError):
            df = pd.DataFrame(columns=['category', 'province', 'price_range'])
    else:
        df = pd.DataFrame(columns=['category', 'province', 'price_range'])

    new_entry = pd.DataFrame({
        'category': [category],
        'province': [province],
        'price_range': [price_range]
    })
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_excel(control_filename, index=False)

def obtener_cantidad_publicaciones(access_token, params):
    url = 'https://api.mercadolibre.com/sites/MLA/search'
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get('paging', {}).get('total', 0)
    else:
        logging.error(f"Error al obtener el total: {response.status_code} - {response.text}")
        print(f"Error al obtener el total de publicaciones: {response.status_code}")
        return None

def fetch_description(session, access_token, item_id):
    item_description_url = f'https://api.mercadolibre.com/items/{item_id}/description'
    try:
        description_response = session.get(
            item_description_url,
            headers={'Authorization': f'Bearer {access_token}'},
            timeout=10
        )
        if description_response.status_code == 200:
            description_data = description_response.json()
            return description_data.get('plain_text', '')
        else:
            return ''
    except requests.RequestException as e:
        logging.error(f"Error al obtener descripción del ítem {item_id}: {e}")
        return ''

def fetch_inmuebles(access_token, params, total_publicaciones, limit=1000, session=None):
    url = 'https://api.mercadolibre.com/sites/MLA/search'
    offset = 0
    total_fetched = 0
    records = []
    fieldnames = set()

    print("Iniciando extracción de inmuebles...")
    while total_fetched < min(total_publicaciones, limit):
        local_params = params.copy()
        local_params['offset'] = offset
        try:
            response = session.get(
                url,
                headers={'Authorization': f'Bearer {access_token}'},
                params=local_params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            results = data.get('results', [])
            if not results:
                print("No se encontraron más resultados.")
                break

            item_ids = [item['id'] for item in results]

            # Obtener detalles de los ítems en lotes de 20
            for i in range(0, len(item_ids), 20):
                batch_ids = item_ids[i:i+20]
                items_detail_url = f'https://api.mercadolibre.com/items?ids={",".join(batch_ids)}'
                try:
                    items_response = session.get(
                        items_detail_url,
                        headers={'Authorization': f'Bearer {access_token}'},
                        timeout=10
                    )
                    items_response.raise_for_status()
                    items_data = items_response.json()

                    # Obtener descripciones en paralelo
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_item = {
                            executor.submit(fetch_description, session, access_token, item_result['body']['id']): item_result
                            for item_result in items_data
                        }
                        for future in as_completed(future_to_item):
                            item_result = future_to_item[future]
                            item_data = item_result.get('body', {})
                            item_id = item_data.get('id', '')

                            description_text = future.result()

                            record = extract_relevant_fields(item_data, description_text)
                            # Verificar que la operación sea 'Venta' mediante el ID del valor
                            if record.get('attribute_operation_id', '') != '242075':
                                continue  # Omitir si no es venta
                            records.append(record)
                            fieldnames.update(record.keys())

                except requests.RequestException as e:
                    logging.error(f"Error al obtener detalles de los ítems: {e}")
                    print(f"Error al obtener detalles de los ítems: {e}")
                    continue

            total_fetched += len(results)
            offset += len(results)
            print(f"Total procesado: {total_fetched}/{min(total_publicaciones, limit)} publicaciones")

            # Verificar si hemos alcanzado el máximo offset permitido
            if offset >= 1000:
                print("Se ha alcanzado el máximo offset permitido por la API.")
                break

        except requests.RequestException as e:
            logging.error(f"Error en la solicitud a la API: {e}")
            print(f"Error en la solicitud a la API: {e}")
            break

    if records:
        fieldnames = sorted(fieldnames)
        # Verifica si el archivo existe y tiene encabezados
        file_exists = os.path.exists(accumulate_filename)
        if file_exists and os.path.getsize(accumulate_filename) > 0:
            with open(accumulate_filename, 'r', newline='', encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                existing_fieldnames = next(reader)
                fieldnames = list(set(fieldnames) | set(existing_fieldnames))
                fieldnames = sorted(fieldnames)
        with open(accumulate_filename, 'a', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
                extrasaction='ignore',
                restval='null'
            )
            if not file_exists or os.path.getsize(accumulate_filename) == 0:
                csv_writer.writeheader()
            else:
                # Reescribe el archivo con las nuevas columnas si es necesario
                if set(fieldnames) != set(existing_fieldnames):
                    df_existing = pd.read_csv(accumulate_filename)
                    df_new = pd.DataFrame(records)
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.to_csv(accumulate_filename, index=False, columns=fieldnames)
                    print(f"Archivo actualizado con nuevas columnas y guardado como '{accumulate_filename}'.")
                    return
            csv_writer.writerows(records)
        print(f"Archivo guardado como '{accumulate_filename}' con todas las columnas detectadas.")
    else:
        print("No hay registros para guardar.")

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main_once():
    session = setup_session()
    processed_filters = load_processed_filters()

    # El ID correcto para "Venta" es '242075' según tu información
    operation_id_venta = '242075'

    for categoria in categorias:
        for provincia in provincias:
            for min_price, max_price_range in price_ranges:
                price_param = f"{min_price}-*" if max_price_range == '*' else f"{min_price}-{max_price_range}"
                current_filter = (categoria, provincia, price_param)
                if current_filter in processed_filters:
                    print(f"Filtro ya procesado: {current_filter}. Se omite.")
                    continue

                params = {
                    'category': categoria,
                    'state': provincia,
                    'limit': page_limit,
                    'sort': 'price_asc',
                    'price': price_param,
                    'currency': currency,
                    'OPERATION': operation_id_venta,  # Filtrar solo propiedades en venta
                }
                print(f"Procesando: Categoría {categoria}, Provincia {provincia}, Rango de precio {price_param}")
                total_publicaciones = obtener_cantidad_publicaciones(access_token, params)

                if total_publicaciones:
                    # Ajustar el límite si total_publicaciones es menor que total_limit
                    limit = min(total_publicaciones, total_limit)
                    fetch_inmuebles(access_token, params, total_publicaciones, limit=limit, session=session)
                    save_processed_filter(*current_filter)
                else:
                    print(f"No hay publicaciones para: Categoría {categoria}, Provincia {provincia}, Rango de precio {price_param}")
                    save_processed_filter(*current_filter)

if __name__ == "__main__":
    main_once()
