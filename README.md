# modelo_marea_estuario

Modelo y análisis de proyección de marea / altura de agua (Río de la Plata → Paraná).

- `backtest_residuo.py`: backtest con reanálisis ERA5 (techo con meteorología perfecta).
- `analisis/`: scripts de exactitud de pronósticos contra lo observado, la capa combinada INA + modelo propio y su emisión por hora (actualización 2026-10-18).
- `.github/workflows/`: emisión por hora y evaluación diaria de la capa combinada (San Fernando).

## Emisión por hora (San Fernando, 1-66 h)

`emitir_pronostico.yml` corre cada hora (`:35`, después de que el captador del SMN levanta el ciclo nuevo):

1. Baja la altura observada en vivo de hidro.gov.ar (10 días horarios). Hace falta porque el captador del SHN corre solo 2 veces por día y el modelo usa el residuo observado de hasta 6 h antes de la emisión (a las 16:56 ART del 18/10 el último dato capturado era de las 09:45 y no se podía emitir). Si la descarga falla, sigue con lo capturado en el repo; si tampoco hay residuo reciente, no emite (mensaje `SIN EMISION` en el log, el job no falla).
2. Clona de forma parcial los repos de datos, ajusta el modelo con todo el archivo y emite desde la hora actual con el ciclo SMN más reciente y la última corrida INA capturada.
3. Agrega una línea a `data/pronosticos_capa.jsonl` (listas alineadas por `h`, horas desde la emisión: `ours`, `ina`, `bw`) y reescribe `data/pronostico_actual.json` (serie completa con `astro`, `stack` y `corr`). Si ya hay una emisión de esa hora con las mismas entradas, no repite.

`evaluar_emitidos.yml` corre una vez por día (14:50 UTC) y escribe `data/evaluacion_emitidos.json`: MAE de `ours`, `ina` y `bw` por horizonte y por día de emisión, contra el observado de San Fernando.

Qué es cada serie: `ours` = modelo propio (armónico + Ridge con residuo reciente y viento SMN); `ina` = pronóstico del INA; `bw` = mezcla con un peso por franja de horizonte (peso del modelo propio 0,56 a 1-6 h y 0,27-0,38 más allá; se reajusta en cada emisión). Los hechos que la sostienen están en `analisis/capa_combinada.py`.

## analisis/

Todos leen los repos de datos (públicos, cuenta `ytodojunto`) clonados en una carpeta `repos/`:
`SHN_captura_pronosticos`, `INA_captura_delta_2`, `SMN_captura_viento`.

```
mkdir trabajo && cd trabajo && mkdir repos && cd repos
for r in SHN_captura_pronosticos INA_captura_delta_2 SMN_captura_viento; do
  git clone --depth 1 https://github.com/ytodojunto/$r
done
cd ..
python3 /ruta/analisis/shn_exactitud.py "$PWD/repos"    # genera shn_eventos.json (en trabajo/)
python3 /ruta/analisis/ina_exactitud.py "$PWD/repos"    # necesita shn_eventos.json; genera sf_extremos.json
python3 /ruta/analisis/modelo_sf.py     "$PWD/repos"    # necesita sf_extremos.json
python3 /ruta/analisis/verif.py         "$PWD/repos"    # reproducción, diagnóstico y mezcla con INA
python3 /ruta/analisis/capa_combinada.py "$PWD/repos"   # capa INA + modelo propio; escribe pronostico_sf_capa.json en trabajo/
python3 /ruta/analisis/emision_rodante.py "$PWD/repos"  # sensibilidad a la hora de emisión (3-12 h desde el init)
python3 /ruta/analisis/emitir_pronostico.py "$PWD/repos" "$PWD/data" [--sin-vivo] [--ahora 2026-10-18T13:00:00]
python3 /ruta/analisis/evaluar_emitidos.py "$PWD/repos" "$PWD/data/pronosticos_capa.jsonl"
```

Hace falta `numpy` y `scikit-learn` (`requests` y `beautifulsoup4` para la descarga en vivo). Los primeros cuatro scripts se encadenan: cada uno usa el JSON que deja el anterior. Los otros solo necesitan `repos/`.

| Script | Qué hace |
|---|---|
| `shn_exactitud.py` | Extremos pronosticados por el SHN (último publicado antes de cada pleamar/bajamar) contra el extremo observado en ±3 h. Por estación, lead, semana. |
| `ina_exactitud.py` | INA horario en San Fernando (pares por corrida, horizonte desde `corrida_forecastdate`), extremos SHN vs INA y corredor Corrientes → Campana (misma hora, sesgo, MAE sin sesgo). |
| `sf_datos.py` | Datos y modelo base de San Fernando: observado, armónico de 17 constituyentes, grilla horaria, ciclos SMN, muestras de entrenamiento, corridas INA. Acepta `SF_EXTRA_JSON` (observado extra), `AHORA` (solo usa lo disponible a ese instante) y `EMIS_OFFSET_H`. |
| `modelo_sf.py` | Evaluación del modelo propio: variante A entrenada hasta el 7/10 y B hasta el 10/10; prueba fuera de muestra desde el 11/10 contra INA y mezcla 50/50. |
| `verif.py` | Reproduce la ventana del 8-10/10, diagnostica 13-14/10 y mide la sensibilidad de la mezcla. |
| `capa_lib.py` | Biblioteca de la capa combinada: variantes (`ours`, `ina`, `b50`, `bw`, `stack`, `corr`), ajuste, pesos y emisión de la serie horaria. |
| `capa_combinada.py` | Evaluación de la capa: validación cruzada por bloques de 3 días con mezcla anidada, prueba fuera de muestra desde el 11/10 e IC por bootstrap sobre días. |
| `emision_rodante.py` | Entrena con la convención validada (emisión = init + 6 h UTC) y prueba con emisiones de 3 a 12 h desde el init, que es lo que ocurre al emitir cada hora. |
| `emitir_pronostico.py` | Emite el pronóstico de las próximas 66 h con lo disponible ahora (ver arriba). Con `--ahora` reproduce lo que había disponible en ese instante, para pruebas y emisiones retroactivas. |
| `evaluar_emitidos.py` | Evalúa las líneas de `pronosticos_capa.jsonl` contra el observado. |

Convenciones: error = observado − pronosticado; horas ART (UTC-3); observado de San Fernando muestreado en HH:45 e interpolado a HH:00; en la validación, emisión = init del ciclo SMN + 6 h UTC. Las fechas de corte (7/10, 10/10, 11/10) están fijas en el código.
