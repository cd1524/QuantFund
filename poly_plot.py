import pandas as pd
import glob
import matplotlib.pyplot as plt

# Ruta donde se encuentran los archivos de Polymarket
ruta_polymarket = ""  # Ajusta según sea necesario

# Leer todos los archivos que empiezan con "polymarket"
archivos_polymarket = glob.glob(ruta_polymarket + "polymarket*.csv")

# Concatenar todos los archivos de Polymarket
datos_polymarket = pd.concat([pd.read_csv(archivo) for archivo in archivos_polymarket])

datos_polymarket = pd.read_csv('polymarket-price-data-03-11-2024-01-12-2024-1738593213639.csv') 
# Leer el archivo de BTC
archivo_btc = ruta_polymarket + "btc.csv"
datos_btc = pd.read_csv(archivo_btc)


# Asegúrate de que las fechas estén en el mismo formato
datos_polymarket['date'] = pd.to_datetime(datos_polymarket['Date (UTC)'])
datos_btc['date'] = pd.to_datetime(datos_btc['date'])

# Combinar datos basados en la fecha
datos_combinados = pd.merge(datos_polymarket, datos_btc, on='date', how='inner')
import pdb;pdb.set_trace()

# Mostrar las primeras filas de los datos combinados
print(datos_combinados.head())
import pdb;pdb.set_trace()

datos_combinados['Close'] = datos_combinados['Close'].fillna(method='ffill')







price_btc= '$90,000'
price_btc= ' $95,000'

datos_combinados[price_btc] = datos_combinados[price_btc].fillna(method='ffill')


# Crear figura y ejes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Gráfico para la columna 'Close'
ax1.plot(datos_combinados['date'], datos_combinados['Close'], label='Close (BTC)', color='blue')
ax1.set_ylabel('Close (BTC)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xlabel('Fecha')

# Crear un segundo eje para price_btc
ax2 = ax1.twinx()
ax2.plot(datos_combinados['date'], datos_combinados[price_btc], label='$90,000 (Polymarket)', color='green')
ax2.set_ylabel('$90,000 (Polymarket)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Títulos y leyendas
fig.suptitle('Comparación de Close (BTC) y $90,000 (Polymarket)')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Mostrar el gráfico
plt.show()



'''

obtener a traves de la red de eth los blockes de cada dia o cosa asi y analizar los subidas de gas los precios los volumenes y esas cosas '''





















datos_combinados['$90,000'] = datos_combinados['$90,000'].fillna(method='ffill')

import matplotlib.pyplot as plt

# Crear figura y ejes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Gráfico para la columna 'Close'
ax1.plot(datos_combinados['date'], datos_combinados['Close'], label='Close (BTC)', color='blue')
ax1.set_ylabel('Close (BTC)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xlabel('Fecha')

# Crear un segundo eje para '$90,000'
ax2 = ax1.twinx()
ax2.plot(datos_combinados['date'], datos_combinados['$90,000'], label='$90,000 (Polymarket)', color='green')
ax2.set_ylabel('$90,000 (Polymarket)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Títulos y leyendas
fig.suptitle('Comparación de Close (BTC) y $90,000 (Polymarket)')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Mostrar el gráfico
plt.show()


# Si quieres guardar el resultado
datos_combinados.to_csv("datos_combinados.csv", index=False)