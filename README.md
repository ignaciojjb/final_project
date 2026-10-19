# Pipeline de clasificación de galaxias

Python 3.12. Ejecutar desde `/workspace/final_project`. No hay que compilar para reentrenar. Cada etapa guarda sus resultados y se ejecuta por separado.

```bash
bash setup_environment.sh
source .venv/bin/activate
```

## 1. Obtener el catálogo y, opcionalmente, el ZIP

Los adjuntos `ZooSpecPhotoDR19_stamps.csv` y `ZooStamps.zip` superan el límite de transferencia de 32 MiB del chat. No están en este entorno. Colócalos en `datasets/raw/` o descárgalos desde enlaces HTTPS directos (con su checksum esperado):

```bash
python -m scripts.download_data --url 'https://HOST/catalog.csv' --sha256 CHECKSUM --output datasets/raw/catalog.csv
python -m scripts.download_data --url 'https://HOST/stamps.zip' --sha256 CHECKSUM --output datasets/raw/stamps.zip
```

Los archivos existentes se reutilizan solo si su checksum coincide. No se imprimen ni guardan URLs con credenciales. Para destinos nuevos, añadir su dominio en la configuración de red. Esta ruta usa un catálogo exportado: no vuelve a ejecutar consultas destructivas de SciServer ni solicita credenciales.

## 2. Limpiar, seleccionar y fijar las particiones

```bash
python -m scripts.prepare_data --catalog datasets/raw/catalog.csv
```

Aplica los cortes de `data_processing.py`, elige 1.000 objetos por clase, fija 70/15/15 y crea el catálogo preparado y el manifiesto SQLite. Usa `--already-clean` si el CSV ya está limpio. `--per-class 0` selecciona todo, pero no es recomendable para un SVC con cientos de miles de filas. Es necesario disponer del tamaño solicitado de cada clase. Los IDs se leen como texto, sin redondearlos. No se sobrescriben manifiestos ni catálogos preparados: para otra muestra utiliza otras rutas.

## 3. Obtener imágenes (elegir una ruta)

Descarga de recortes SDSS mediante hips2fits, reanudable y con TLS verificado:

```bash
python -m scripts.download_images --workers 8
```

O importar los stamps existentes sin extraer el ZIP al sistema de archivos:

```bash
python -m scripts.import_stamps --archive datasets/raw/stamps.zip --mapping datasets/raw/image_mapping.csv
```

El mapping requiere `specobjid,image_path`, donde `image_path` es el nombre exacto dentro del ZIP. Debe construirse después de inspeccionar el catálogo y el ZIP; no se adivina la correspondencia. Las filas fuera de la muestra se omiten. Ambas rutas mantienen los splits y omiten imágenes ya descargadas.

## 4. Calcular y guardar características una vez

```bash
python -m scripts.extract_features --components 64
```

Convierte a gris 64×64 → 4096 píxeles → PCA de 64 componentes, escala e incorpora fotometría. PCA, imputación y escalado se ajustan únicamente con entrenamiento. Las banderas y probabilidades de morfología no entran como predictores. Guarda `artifacts/features.joblib` con matriz, IDs, etiquetas, splits, transformador y firmas SHA256 de los inputs/código. Reutiliza una caché válida; rechaza una caché desactualizada. Si cambian datos, splits o PCA, elige otro `--output` y recalcula esta etapa.

## 5. Entrenar y reentrenar sin repetir etapas anteriores

```bash
python -m scripts.train_model --output artifacts/runs/svc-001
python -m scripts.train_model --C 10 --gamma scale --output artifacts/runs/svc-002
```

Solo carga la caché y entrena el SVC. Cada ejecución conserva su propio `model.joblib`, `metrics.json`, `predictions.csv` y `merged_features.csv`. El modelo incluye el transformador para inferencia con los píxeles y columnas originales. No cargar archivos joblib de fuentes no confiables. El directorio de resultados no se sobrescribe.

Usa validación para comparar configuraciones; reserva test para la evaluación final y no lo uses para decidir hiperparámetros. Las métricas de ambas particiones se guardan por separado; los scripts no seleccionan automáticamente el mejor modelo. Las filas de una misma galaxia deben compartir split si existen observaciones con diferentes IDs.

## Verificación y estado

```bash
python -m unittest discover -s tests -v
```

Pruebas sintéticas de extremo a extremo: entrenamiento, 64 componentes, ausencia de ajuste con test, serialización, caché, invalidación, ZIP, preparación y descarga con checksum simulada. La descarga real del catálogo/ZIP y el entrenamiento astronómico no se han validado porque faltan datos accesibles. La prueba de descarga simulada no acredita conectividad a un proveedor. Los datos y artefactos se conservan en disco y están excluidos de Git.

Para archivos grandes, la matriz de 4096 píxeles requiere memoria considerable. La implementación está dimensionada para la muestra de 2.000 objetos de los notebooks originales. No descargar ni procesar el catálogo completo sin valorar memoria y coste de SVC.

## Entorno en la nube

`setup_environment.sh` instala/verifica dependencias y ejecuta las pruebas; no descarga datos ni inicia entrenamientos automáticamente. No requiere procesos de servidor persistentes. Las instrucciones de inicio del entorno deben activar `.venv`, reutilizar el checkout existente y conservar `datasets/` y `artifacts/`. Publicar el entorno es una operación de la interfaz del producto; guardar su borrador no lo publica.

Estado de red comprobado durante la preparación: una petición real de un recorte a hips2fits devolvió `Tunnel connection failed: 403 Forbidden` del proxy. Se guardaron `alasky.cds.unistra.fr` y `skyserver.sdss.org` en la lista de dominios del borrador, conservando los presets. Guardar ese borrador no aplica la política al runtime; tras guardar/publicar desde la interfaz, repetir `python -m scripts.download_images`. No se ha acreditado todavía una descarga real de recortes. Se verificó el instalador y pasaron las 6 pruebas sintéticas.
