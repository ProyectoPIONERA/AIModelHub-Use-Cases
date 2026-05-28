# AIModelHub Component Use Cases: FLARES & Mobility                                                                                                                   
## Descripción General                                                                                                                                                           
Este repositorio contiene los recursos necesarios para construir dos casos de uso para el componente AIModelHub de PIONERA: **FLARES** (análisis lingüístico de fiabilidad y contenido) y **Mobility** (predicción en transporte público). 
                                                                                                                                                            
> **Nota importante**: Este repositorio es una **demostración simplificada**. Los modelos y el preprocesamiento de los datos de entrada están diseñados de manera simplificada para exponerlos como ejemplos en AIModelLHub dentro de un espacio de datos. El objetivo de este repositorio no es profundizar en ambos dominios.                                                                                    
                                                                                                                                                            
## Tabla de Contenidos                                                                                                                                      
- [Descripción General](#descripción-general)                                                                                                               
- [FLARES: Análisis Lingüístico](#flares-análisis-lingüístico)                                                                                              
- [Mobility: Predicción de Movilidad](#mobility-predicción-de-tiempos-de-viaje)                                                                      
- [Requisitos](#requisitos)                                                                                                                                 
- [Inicio Rápido](#inicio-rápido)                                                                                                                                                                                                                                                 
- [Estructura del Repositorio](#estructura-del-repositorio)                                                                                                                                                                                                                                                                                                                            
## FLARES: Análisis Lingüístico                                                                                                                             
                                                                                                                                                            
### Descripción                                                                                                                                             
FLARES es un desafío de procesamiento de lenguaje natural (NLP) que se centra en dos tareas subordinadas:                                                   
                                                                                                                                                            
*   **Subtarea 1: Identificación de 5W1H**                                                                                                                  
    Dado un texto, el modelo debe identificar y anotar las respuestas a las preguntas de los 5W1H. Por ejemplo, en la frase *"El arresto del científico     
italiano tuvo lugar por la fuerza ayer en Milán por vender una vacuna no autorizada"*, el modelo debe extraer:                                              
    *   **Qué**: El arresto                                                                                                                                 
    *   **Quién**: El científico italiano                                                                                                                   
    *   **Cómo**: Por la fuerza                                                                                                                             
    *   **Cuándo**: Ayer                                                                                                                                    
    *   **Dónde**: En Milán                                                                                                                                 
    *   **Por qué**: Por vender una vacuna no autorizada                                                                                                    
                                                                                                                                                            
*   **Subtarea 2: Clasificación de Fiabilidad**                                                                                                             
    Para cada elemento 5W1H identificado, el modelo debe clasificar su fiabilidad en una de tres categorías:                                                
    *   `confiable`: Información precisa y objetiva.                                                                                                        
    *   `semiconfiable`: Información que puede ser precisa pero carece de evidencia o es ambigua.                                                           
    *   `no confiable`: Información inexacta, subjetiva o que carece de fuentes verificables.                                                               
                                                                                                                                                            
### Métricas de Evaluación                                                                                                                                  
*   **Subtarea 1 (Identificación)**: Se utilizan métricas de *span classification*:                                                                         
    *   **Correcto**: Coincidencia exacta de inicio, fin y etiqueta.                                                                                        
    *   **Parcial**: Superposición parcial de intervalos con la misma etiqueta.                                                                             
    *   **Faltante**: Elemento presente en el texto de referencia pero no detectado.                                                                        
    *   **Espurio**: Elemento detectado que no existe en el texto de referencia.                                                                            
    *   Se calcula la precisión, el recall y el F1-Score a partir de estas categorías.                                                                      
                                                                                                                                                            
*   **Subtarea 2 (Clasificación)**: Se utilizan métricas de clasificación estándar:                                                                         
    *   **Precisión, Recall y F1-Score** para cada una de las tres clases de fiabilidad.                                                                    
                                                                                                                                                            
### Referencias                                                                                                                                             
*   [Descripción completa de FLARES](https://sites.google.com/gcloud.ua.es/flares/description)                                                              
*   [Métricas de evaluación de FLARES](https://sites.google.com/gcloud.ua.es/flares/evaluation-metrics)                                                     
                                                                                                                                                            
## Mobility: Predicción de Tiempos de Viaje                                                                                                                 
                                                                                                                                                            
### Descripción                                                                                                                                             
Este caso de uso utiliza el estándar **GTFS** para entrenar modelos de aprendizaje automático que predicen métricas relacionadas con el tiempo transcurrido entre paradas y los delays en el transporte público.                                                                                                                      
                                                                                                                                                            
*   **GTFS Schedule**: El repositorio incluye datos de ejemplo basados en el feed de la *Empresa Municipal de Transportes de Fuenlabrada (EMTF)*            
([mdb-2400](https://mobilitydatabase.org/feeds/gtfs/mdb-2400)). Este feed contiene información estática como rutas, paradas, horarios y formas geográficas. 
                                                                                                                                                            
*   **GTFS Realtime**: Para complementar los datos estáticos, se utilizan feeds en tiempo real de la misma agencia:                                         
    *   [Actualizaciones de viaje (mdb-2401)](https://mobilitydatabase.org/feeds/gtfs_rt/mdb-2401)                                                          
    *   [Posiciones de vehículos (mdb-2402)](https://mobilitydatabase.org/feeds/gtfs_rt/mdb-2402)                                                           
    *   [Alertas de servicio (mdb-2403)](https://mobilitydatabase.org/feeds/gtfs_rt/mdb-2403)                                                               
                                                                                                                                                            
*   **Modelo de Entrenamiento**: El script `src/utils/mobility/train_models.py` entrena tres modelos (LightGBM, Random Forest y MLP) para predecir el       
`actual_travel_time`, `delay` o `previous_delay`). El preprocesamiento incluye la codificación de paradas y la creación de configuraciones específicas por `target`.                                                                                        
                                                                                                                                                            
### Referencias                                                                                                                                             
*   [Documentación GTFS (Español)](https://gtfs.org/es/documentation/overview/)                                                                             
*   [Referencia GTFS Schedule (Español)](https://gtfs.org/es/documentation/schedule/reference/)                                                             
*   [Referencia GTFS Realtime (Español)](https://gtfs.org/es/documentation/realtime/reference/)                                                             
*   [Base de datos de feeds GTFS (Mobility Database)](https://mobilitydatabase.org/feeds/gtfs_rt/mdb-2402)                                                  
                                                                                                                                                            
## Requisitos                                                                                                                                               
                                                                                                                                                            
*   **Python**: 3.9 o superior.                                                                                                                             
*   **Dependencias**: Se listan en `requirements.txt`. Las principales son:                                                                                 
    *   `scikit-learn`: Para el preprocesamiento y los modelos de ML.                                                                                       
    *   `pandas`: Para la manipulación de datos.                                                                                                            
    *   `fastapi`: Para el servidor API.                                                                                                                    
    *   `lightgbm`: Para el modelo LightGBM.                                                                                                                
    *   `transformers` y `torch`: Para los modelos de FLARES (aunque no se usan en la demo de Mobility, son necesarios para la estructura completa).        

```
pip install -r requirements.txt
```
## Inicio Rápido                                                                                                                                            
                                                                                                                                                            
### 1. Preparar los datos                                                                                                                                   
Asegúrate de que los datos de ejemplo estén en la ubicación esperada:                                                                                       
#### Datos de FLARES                                                                                   

El directorio data/flares debe contener los siguientes archivos:
- 5w1h_subtarea_1_train.json
- 5w1h_subtarea_1_test.json
- 5w1h_subtarea_2_train.json
- 5w1h_subtarea_2_test.json

Los ficheros se pueden obtener en el siguiente enlace: 
* [Flares Data](https://sites.google.com/gcloud.ua.es/flares/data)                                                 
#### Datos de Mobility                                                                                                                                         
El directorio data/mobility-datasets debe contener los siguientes recursos:
- Subdirectorio GTFS_Schedule con los ficheros que definen la planificación.
- Subdirectorio GTFS_RT con los ficheros .pb recolectados
- Dataset Generados: segments_train.csv y segments_test.csv

Los siguientes comandos se pueden utilizar para ejecutar script que recolecta datos de GTFS-RT para Fuenlabrada cada 10 segundos:
```
chmod u+x ./scripts/collector.sh
./scripts/collector.sh 10
```

Mediante el siguiente comando se generan los ficheros segments_train.csv y segments_test.csv usando los datos en formato GTFS:

```
python -m src.utils.mobility.preprocessor.py
```

### 2. Entrenar y evaluar los modelos                                                                                                                                     

El entrenamiento y la generación de los modelos está automatizada en el script **create_models.sh**. Si se requiere modificar algún parámetro o entrenar otros modelos se pueden ejecutar de manera individual mediante los siguientes comandos:                                                       

FLARES:                                                                                                         
```                                                            
python -m src.utils.flares.train_models.py --model model_name                                                                        
```                                                  
                                                                                        
Mobility:                                                                                                                                             
```                                                                                               
python -m src.utils.mobility.train_models.py \     
--data-path "./data/mobility-datasets/segments_train.csv" \                                                                                        --test-path "./data/mobility-datasets/segments_test.csv" \                      \
--output-dir "./models/mobility" \                                                                                       --target "actual_travel_time"                                                                 
```                                                        
                                                                                                                                                                                                                                                                                  

```
python -m src.utils.flares.evaluate_models --task reliability --model-name dccuchile-bert-base-spanish-wwm-uncased-reliability
```

**Revisar los scripts para ver parámetros adicionales**

### 3. Iniciar el servidor API                                                                                                                                  

El servidor expondrá dinámicamente los endpoints para todos los modelos entrenados que se encuentren en el directorio ./models.                                                                         

```                                                               
uvicorn src.server:app --reload                                                                                 
```                                            
                                                                                                                                                            
### 4. Probar los endpoints                                                                                                                                     

Una vez el servidor esté en ejecución, puedes probar los endpoints con curl o una herramienta como Postman.                                                 

Ejemplo para Mobility:                                                                              
```                                                                  
ccurl -X 'POST' \
  'http://127.0.0.1:8000/mobility/catboost_actual_travel_time' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[
  {
    "trip_id": "L13_1_05:45_LxI",
    "from_stop_id": "7716",
    "to_stop_id": "19219",
    "route_id": "13",
    "scheduled_travel_time": 120,
    "shape_distance": 681.1956848810403,
    "is_peak": 0,
    "hour_sin": 0.7071067811865475,
    "hour_cos": 0.7071067811865476,
    "weekday_sin": 0.9749279121818236,
    "weekday_cos": -0.22252093395631434,
    "previous_delay_ratio": 0.2499999979166667,
    "previous_delay_delta": 0.0
  }
]'                                                                                   
```                                                                         

Ejemplo para FLARES (5W1H):                                                                                                                                 
```                                                                   
curl -X 'POST' \
  'http://127.0.0.1:8000/flares/dccuchile-distilbert-base-spanish-uncased-5w1h' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[
{"Id":840,"Text":"El comit\u00e9 de medicamentos humanos (CHMP) espera poder concluir el análisis de todo el paquete de datos de este ant\u00eddoto \"a mediados de marzo\", concretamente esperan dar luz verde el 8 de marzo.\u00a0 La condici\u00f3n es que la informaci\u00f3n presentada por la empresa sea siempre \"lo suficientemente completa y s\u00f3lida\" como para aceptar su uso en todos los pa\u00edses europeos al mismo tiempo y en las mismas condiciones."}]'                                                                                 
      
```

Ejemplo para FLARES (Reliability):                                                                                        

```
curl -X 'POST' \                                                                                                                                            
  'http://localhost:8000/flares/bert-base-uncased-reliability' \                                                                                                   
  -H 'Content-Type: application/json' \                                                                                                                     
  -d '[
    {"Id":840,"Text":"El comit\u00e9 de medicamentos humanos (CHMP) espera poder concluir el análisis de todo el paquete de datos de este ant\u00eddoto \"a mediados de marzo\", concretamente esperan dar luz verde el 8 de marzo.\u00a0 La condici\u00f3n es que la informaci\u00f3n presentada por la empresa sea siempre \"lo suficientemente completa y s\u00f3lida\" como para aceptar su uso en todos los pa\u00edses europeos al mismo tiempo y en las mismas condiciones.",
      "Tag_Start": 0,
      "Tag_End": 40,
      "5W1H_Label": "WHO",
      "Tag_Text": "El comité de medicamentos humanos (CHMP)"
    }]'                                                                                     
```              

## Estructura del Repositorio                                                                              

```
.                                                                                                                                                           
├── src/                                                                                                                                                    
│   ├── schemas/          # Definición de los esquemas de datos (Pydantic)                                                                                  
│   ├── services/         # Servicios de inferencia (flares_service.py, mobility_service.py)                                                                
│   ├── utils/                                                                                                                                              
│   │   ├── flares/       # Scripts de entrenamiento y utilidades de FLARES                                                                                 
│   │   └── mobility/     # Scripts de entrenamiento y utilidades de Mobility                                                                               
│   └── server.py         # Punto de entrada del servidor FastAPI                                                                                           
├── data/                 # Datos de entrada (entrenamiento y pruebas)                                                                                      
├── models/               # Directorio donde se guardan los modelos entrenados                                                                              
│   ├── flares/           # Modelos de FLARES                                                                                                               
│   └── mobility/         # Modelos de Mobility y configuraciones de preprocesamiento                                                                       
├── requirements.txt      # Dependencias del proyecto                                                                                                       
└── README.md             # Este archivo                                                                                                                    
```                                                                                                                                                            
## Financiación

This work has received funding from the **PIONERA project** (Enhancing interoperability in data spaces through artificial intelligence), a project funded in the context of the call for Technological Products and Services for Data Spaces of the Ministry for Digital Transformation and Public Administration within the framework of the PRTR funded by the European Union (NextGenerationEU).

<div align="center">
  <img src="funding_label.png" alt="Logos financiación" width="900" />
</div>