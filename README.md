# modelo_marea_estuario

Modelo y análisis de proyección de marea / altura de agua (Río de la Plata → Paraná).

- `backtest_residuo.py`: backtest con reanálisis ERA5 (techo con meteorología perfecta).
- `analisis/`: scripts de exactitud de pronósticos contra lo observado (actualización 2026-10-18).

## analisis/

Todos leen los repos de datos (públicos, cuenta `ytodojunto`) clonados en una carpeta `repos/`:
`SHN_captura_pronosticos`, `INA_captura_delta_2`, `SMN_captura_viento`.

```
mkdir trabajo && cd trabajo && mkdir repos && cd repos
for r in SHN_captura_pronosticos INA_captura_delta_2 SMN_captura_viento; do
  git clone --depth 1 https://github.com/ytodojunto/$r
done
cd ..
python3 -I /ruta/analisis/shn_exactitud.py "$PWD/repos"   # genera shn_eventos.json (en trabajo/)
python3 -I /ruta/analisis/ina_exactitud.py "$PWD/repos"   # necesita shn_eventos.json; genera sf_extremos.json
python3 -I /ruta/analisis/modelo_sf.py     "$PWD/repos"   # necesita sf_extremos.json
python3 -I /ruta/analisis/verif.py         "$PWD/repos"   # reproducción, diagnóstico y mezcla con INA
```

Hace falta `numpy` y `scikit-learn`. El orden importa: cada script usa el JSON que deja el anterior.

| Script | Qué hace |
|---|---|
| `shn_exactitud.py` | Extremos pronosticados por el SHN (último publicado antes de cada pleamar/bajamar) contra el extremo observado en ±3 h. Por estación, lead, semana. |
| `ina_exactitud.py` | INA horario en San Fernando (pares por corrida, horizonte desde `corrida_forecastdate`), extremos SHN vs INA y corredor Corrientes → Campana (misma hora, sesgo, MAE sin sesgo). |
| `modelo_sf.py` | Modelo propio en San Fernando: armónico de 17 constituyentes + Ridge por franja de horizonte (residuo reciente + viento SMN). Variante A entrenada hasta el 7/10 y B hasta el 10/10; prueba fuera de muestra desde el 11/10 contra INA y mezcla 50/50. |
| `verif.py` | Reproduce la ventana del 8-10/10, diagnostica 13-14/10 y mide la sensibilidad de la mezcla. |

Convenciones: error = observado − pronosticado; horas ART (UTC-3); observado de San Fernando muestreado en HH:45 e interpolado a HH:00; emisión del modelo = init del ciclo SMN + 6 h. Las fechas de corte (7/10, 10/10, 11/10) están fijas en el código.
