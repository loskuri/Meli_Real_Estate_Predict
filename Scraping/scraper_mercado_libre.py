import requests
import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import asyncio
import aiohttp
import itertools
from collections import deque
import nest_asyncio


id = '370085680092051'
secret = "5KwGBJlsZoWBYdeypr9mH1N5BcCQojW4"
code = 'TG-'+ "6734bf28a572f800018aec21" "-242964583"
redirect_url  = 'https://borderline.mercadoshops.com.ar/'
#url = "https://auth.mercadolibre.com.ar/authorization?response_type=code&client_id=370085680092051&redirect_uri=https://borderline.mercadoshops.com.ar/"

# Archivo para guardar los IDs procesados
processed_ids_file = 'D:\Tesis\id_procesado.txt'
output_folder = 'D:\Tesis\Scraper_meli_5'

url = 'https://api.mercadolibre.com/oauth/token'

# Encabezados
headers = {
    'accept': 'application/json',
    
    'content-type': 'application/x-www-form-urlencoded',
}

# Datos que se envían en la solicitud
data = {
    'grant_type': 'authorization_code',
    'client_id': id,  # ID de tu aplicación
    'client_secret': secret,  # Secret Key
    'code': code,  # Authorization code
    'redirect_uri':redirect_url # URI de redireccionamiento
}

# Enviar la solicitud POST
response = requests.post(url, headers=headers, data=data)

# Imprimir la respuesta de la API
print(response.status_code)
print(response.json())
response_json = response.json()
access_token = response_json.get("access_token")  # Acceder al token
ACCESS_TOKEN = access_token


# Aplicar nest_asyncio para entornos de Jupyter Notebook
nest_asyncio.apply()

# Configuración de la API de MercadoLibre
api_url = 'https://api.mercadolibre.com/sites/MLA/search'

# Parámetros de configuración
os.makedirs(output_folder, exist_ok=True)
results_per_page = 50
max_results_per_query = 1000
max_price = 100000000000
min_price = 0

# IDs de categorías de interés
category_ids = ['MLA1459']  # Casas, Departamentos, PH

# IDs de ubicaciones para AMBA
location_ids = ['es_AR']


# Cargar IDs procesados previamente
def load_processed_ids(filename):
    if not os.path.exists(filename):
        return set()
    with open(filename, 'r') as f:
        return set(line.strip() for line in f)

# Guardar nuevos IDs procesados
def save_processed_ids(filename, processed_ids):
    with open(filename, 'w') as f:
        for item_id in processed_ids:
            f.write(f"{item_id}\n")

# Asegúrate de definir tu access_token aquí

# Función para extraer campos relevantes de un ítem
async def extract_relevant_fields(item, session):
    record = {}
    try:
        # Campos básicos
        record['id'] = item.get('id')
        record['title'] = item.get('title')
        record['price'] = item.get('price')
        record['currency_id'] = item.get('currency_id')
        record['condition'] = item.get('condition')
        record['permalink'] = item.get('permalink')
        record['available_quantity'] = item.get('available_quantity')
        record['sold_quantity'] = item.get('sold_quantity')
        record['category_id'] = item.get('category_id')
        record['domain_id'] = item.get('domain_id')
        record['date_created'] = item.get('date_created')
        record['last_updated'] = item.get('last_updated')
        record['listing_type_id'] = item.get('listing_type_id')  # Tipo de listado

        # Información del vendedor
        seller = item.get('seller', {})
        record['seller_id'] = seller.get('id')
        record['seller_power_seller_status'] = seller.get('power_seller_status')
        record['seller_car_dealer'] = seller.get('car_dealer')
        record['seller_real_estate_agency'] = seller.get('real_estate_agency')
        record['seller_tags'] = ';'.join(seller.get('tags', []))

        # Información adicional del vendedor
        seller_reputation = seller.get('seller_reputation', {})
        record['seller_reputation_level'] = seller_reputation.get('level_id')
        record['seller_transactions_completed'] = seller_reputation.get('transactions', {}).get('completed')
        record['seller_transactions_canceled'] = seller_reputation.get('transactions', {}).get('canceled')
        record['seller_transactions_total'] = seller_reputation.get('transactions', {}).get('total')
        record['seller_transactions_ratings_positive'] = seller_reputation.get('transactions', {}).get('ratings', {}).get('positive')
        record['seller_transactions_ratings_negative'] = seller_reputation.get('transactions', {}).get('ratings', {}).get('negative')
        record['seller_transactions_ratings_neutral'] = seller_reputation.get('transactions', {}).get('ratings', {}).get('neutral')
        record['seller_metrics_claims_rate'] = seller_reputation.get('metrics', {}).get('claims', {}).get('rate')
        record['seller_metrics_cancellations_rate'] = seller_reputation.get('metrics', {}).get('cancellations', {}).get('rate')
        record['seller_metrics_delayed_handling_time_rate'] = seller_reputation.get('metrics', {}).get('delayed_handling_time', {}).get('rate')

        # Ubicación de la propiedad
        location = item.get('location', {})
        record['location_address_line'] = location.get('address_line')
        record['location_city'] = location.get('city', {}).get('name')
        record['location_state'] = location.get('state', {}).get('name')
        record['location_country'] = location.get('country', {}).get('name')
        record['location_neighborhood'] = location.get('neighborhood', {}).get('name')
        record['location_municipality'] = location.get('municipality', {}).get('name')
        record['location_latitude'] = location.get('latitude')
        record['location_longitude'] = location.get('longitude')

        # Información de envío (si aplica)
        shipping = item.get('shipping', {})
        record['shipping_mode'] = shipping.get('mode')
        record['shipping_local_pick_up'] = shipping.get('local_pick_up')
        record['shipping_free_shipping'] = shipping.get('free_shipping')
        record['shipping_methods'] = ';'.join(shipping.get('methods', []))

        # Información de promociones
        record['original_price'] = item.get('original_price')
        record['discount'] = item.get('original_price') - item.get('price') if item.get('original_price') else 0

        # Etiquetas del artículo
        record['tags'] = ';'.join(item.get('tags', []))

        # Garantía
        warranty = item.get('warranty')
        record['warranty'] = warranty

        # Atributos de la propiedad
        attributes = item.get('attributes', [])
        for attr in attributes:
            attr_id = attr.get('id', '').lower()
            attr_value = attr.get('value_name', '')
            record[attr_id] = attr_value

        # Financiamiento
        record['financiamiento'] = 'Sí' if any('financing' in tag for tag in item.get('tags', [])) else 'No'

        # Descripción adicional (requiere solicitud adicional)
        description = await fetch_item_description(item.get('id'), session)
        if description:
            record['description'] = description.get('plain_text')

    except Exception as e:
        print(f"Error al extraer campos del ítem {item.get('id')}: {e}")
        return None

    return record

# Función asíncrona para obtener la descripción de un ítem
async def fetch_item_description(item_id, session):
    url = f'https://api.mercadolibre.com/items/{item_id}/description'
    headers = {'Authorization': f'Bearer {access_token}'}
    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                # Manejar errores específicos
                if response.status == 404:
                    print(f'Descripción no encontrada para el ítem {item_id}')
                else:
                    print(f'Error {response.status} al obtener la descripción del ítem {item_id}')
                return None
    except Exception as e:
        print(f'Error al obtener la descripción del ítem {item_id}: {e}')
        return None

# Función asíncrona para obtener detalles de múltiples ítems
async def fetch_items_details(session, item_ids):
    items_details = []
    tasks = []
    for i in range(0, len(item_ids), 20):
        ids_chunk = item_ids[i:i+20]
        url = f'https://api.mercadolibre.com/items?ids={",".join(ids_chunk)}'
        tasks.append(fetch(session, url))
    responses = await asyncio.gather(*tasks)
    for response in responses:
        if response:
            for item in response:
                if 'body' in item:
                    items_details.append(item['body'])
    return items_details

async def fetch(session, url):
    headers = {'Authorization': f'Bearer {access_token}'}
    retries = 5
    backoff_factor = 0.5
    for attempt in range(retries):
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                elif response.status == 429:
                    print(f'Error 429: Too Many Requests al obtener datos de {url}. Reintentando...')
                    await asyncio.sleep(backoff_factor * (2 ** attempt))
                elif response.status == 401:
                    print(f'Error 401: No autorizado al obtener datos de {url}. Verifica tu access_token.')
                    return None
                else:
                    print(f'Error {response.status} al obtener datos de {url}')
                    return None
        except Exception as e:
            print(f'Excepción al obtener datos de {url}: {e}')
            await asyncio.sleep(backoff_factor * (2 ** attempt))
    print(f'No se pudo obtener datos de {url} después de {retries} intentos.')
    return None

# Función asíncrona para procesar un rango de precios
async def process_price_range(category_id, location_id, price_min, price_max, processed_ids, session):
    queue = deque([(price_min, price_max)])
    results_obtained = 0

    while queue:
        price_min, price_max = queue.popleft()
        success, total_results, records = await fetch_data(category_id, location_id, price_min, price_max, processed_ids, session)

        if not success:
            if price_min < price_max:
                N = 10
                price_step = max((price_max - price_min + 1) // N, 1)
                new_ranges = [(price_min + i * price_step, min(price_min + (i + 1) * price_step - 1, price_max)) for i in range(N)]
                queue.extend(new_ranges)
            else:
                print(f"No se puede dividir más el rango {price_min}-{price_max}.")
        else:
            results_obtained += len(records)
            print(f"Procesado rango {price_min}-{price_max} con {total_results} resultados obtenidos: {results_obtained} acumulados.")

# Función asíncrona para obtener datos
async def fetch_data(category_id, location_id, price_min, price_max, processed_ids, session):
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {
        'category': category_id,
        'price': f'{price_min}-{price_max}',
        'limit': results_per_page,
        'sort': 'price_asc',
        'offset': 0,
    }
    if location_id:
        params['item_location.state'] = location_id

    offset = 0
    all_records = []
    total_results = None

    while True:
        params['offset'] = offset
        try:
            async with session.get(api_url, params=params, headers=headers) as response:
                if response.status != 200:
                    print(f'Error {response.status} en la solicitud: {response.url}')
                    data = await response.json()
                    print(f'Detalles del error: {data}')
                    break
                data = await response.json()
        except Exception as e:
            print(f'Error al realizar la solicitud: {e}')
            break

        if total_results is None:
            total_results = data.get('paging', {}).get('total', 0)
            print(f'Total resultados esperados para el rango {price_min}-{price_max}: {total_results}')
            if total_results > max_results_per_query:
                return False, total_results, []

        results = data.get('results', [])
        if not results:
            break

        item_ids = [item['id'] for item in results if item['id'] not in processed_ids]

        if item_ids:
            items_details = await fetch_items_details(session, item_ids)
            for item_detail in items_details:
                item_id = item_detail.get('id')
                if item_id in processed_ids:
                    continue
                # Extraer campos relevantes
                record = await extract_relevant_fields(item_detail, session)
                if record:
                    all_records.append(record)
                    processed_ids.add(item_id)
                    print(f"Agregado ítem {item_id} a all_records")
                else:
                    print(f"No se pudo extraer información del ítem {item_id}")

        offset += results_per_page
        if offset >= total_results:
            break

    if all_records:
        filename = f'{output_folder}/data_{category_id}_{location_id or "all"}_{price_min}_{price_max}.json'
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(all_records, f, ensure_ascii=False, indent=4)
        print(f'Datos guardados en {filename}')

    return True, total_results, all_records

# Función principal asíncrona
async def main():
    processed_ids = load_processed_ids(processed_ids_file)
    async with aiohttp.ClientSession() as session:
        for category_id, location_id in itertools.product(category_ids, location_ids):
            print(f'Procesando categoría {category_id}, ubicación {location_id}')
            await process_price_range(category_id, location_id, min_price, max_price, processed_ids, session)
    save_processed_ids(processed_ids_file, processed_ids)

# Ejecutar el script
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except RuntimeError:
        nest_asyncio.apply()
        asyncio.run(main())
