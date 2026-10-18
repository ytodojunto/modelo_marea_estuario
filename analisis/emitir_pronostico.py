"""Emite el pronostico horario de la capa combinada (San Fernando, 1-66 h) con lo que hay disponible AHORA.

Cada corrida:
  1. baja la altura observada en vivo del SHN (hidro.gov.ar, 10 dias de datos horarios) para que el residuo
     reciente (r0) tenga ~1-2 h de edad; sin esto solo habria observado de las capturas de 2 veces por dia;
  2. carga los repos de datos (SHN alturas, INA, viento SMN), ajusta el modelo con todo el archivo y emite
     desde la hora actual con el ciclo SMN mas reciente y la ultima corrida del INA;
  3. agrega una linea a <data>/pronosticos_capa.jsonl (salvo que ya exista una emision para esa hora y esas
     mismas entradas) y reescribe <data>/pronostico_actual.json con la serie completa (ours/ina/bw/stack/corr).

La linea del jsonl es compacta: listas alineadas por h (horas desde la emision); T = emision + h horas.

Uso: python3 emitir_pronostico.py <carpeta repos> <carpeta data> [--ahora ISO_UTC] [--sin-vivo]
  --ahora     reproduce lo que habia disponible en ese instante (prueba / backfill); implica --sin-vivo
  --sin-vivo  no baja el observado de hidro.gov.ar
"""
import sys, os, json, datetime as dt, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = [a for a in sys.argv[1:] if a.startswith("--")]
R, DATA = args[0], args[1]
AHORA_ARG = None
if "--ahora" in sys.argv:
    AHORA_ARG = sys.argv[sys.argv.index("--ahora") + 1]
    args = [a for a in args if a != AHORA_ARG]
    R, DATA = args[0], args[1]
SIN_VIVO = "--sin-vivo" in flags or AHORA_ARG is not None
os.makedirs(DATA, exist_ok=True)

now_utc = dt.datetime.fromisoformat(AHORA_ARG.replace("Z", "")[:19]) if AHORA_ARG else dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
if AHORA_ARG:
    os.environ["AHORA"] = now_utc.isoformat()
emis = (now_utc - dt.timedelta(hours=3)).replace(minute=0, second=0, microsecond=0)  # hora ART, en punto

# ---------------- 1. observado en vivo
obs_vivo = False
if not SIN_VIVO:
    try:
        sys.path.insert(0, os.path.join(R, "SHN_captura_pronosticos"))
        import requests
        from capturar_alturas import URL_CSV, parsear_como_csv, parsear_como_html
        resp = requests.get(URL_CSV, timeout=40, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        txt = resp.text
        regs = parsear_como_html(txt) if txt.strip().lower().startswith(("<!doctype", "<html")) else parsear_como_csv(txt)
        extra = {}
        for m in regs:
            if m["estacion"].strip() == "San Fernando":
                t = dt.datetime.strptime(m["fecha_hora"], "%d/%m/%Y %H:%M")
                extra[t.strftime("%Y-%m-%dT%H:%M")] = float(m["altura_m"])
        if extra:
            p = os.path.join(tempfile.gettempdir(), "sf_obs_vivo.json")
            json.dump(extra, open(p, "w"))
            os.environ["SF_EXTRA_JSON"] = p
            obs_vivo = True
            print("observado en vivo: %d horas, hasta %s ART" % (len(extra), max(extra)))
        else:
            print("AVISO: la descarga en vivo no trajo datos de San Fernando; se sigue con las capturas del repo")
    except Exception as e:
        print("AVISO: no se pudo bajar el observado en vivo (%s: %s); se sigue con las capturas del repo" % (type(e).__name__, e))

import capa_lib as L

# ---------------- 2. cargar, ajustar, emitir
G = L.cargar(R, 3)
rows = L.enriquecer(G)
W, models = L.ajustar(rows)
cyc, P = G["cyc"], G["P"]
if not cyc:
    print("ERROR: no hay ciclos de viento completos")
    sys.exit(1)
init_k = max(cyc)  # ciclo mas reciente completo
d = cyc[init_k]
if emis < P(d["ciclo_init"]) - dt.timedelta(hours=3):
    print("ERROR: el ciclo mas reciente (%s) es posterior a la hora de emision" % d["ciclo_init"])
    sys.exit(1)
fc = L.serie_ciclo(G, d, emis, W, models)
if fc is None:
    print("SIN EMISION: no hay residuo observado entre %s y %s ART (ultimo observado %s ART)" % (emis - 6 * L.H, emis - L.H, G["last_obs"]))
    sys.exit(0)
fc.update(generado_en_utc=now_utc.isoformat(), obs_vivo=obs_vivo, ultimo_observado_art=str(G["last_obs"]),
          edad_residuo_h=round((emis - P(fc["ultimo_residuo_art"])).total_seconds() / 3600, 1),
          pesos_bw={"%d-%d h" % L.BANDS[i]: round(float(v), 3) for i, v in W.items()},
          cantidad_muestras=len(rows))
n_bw = sum(1 for p in fc["serie"] if "bw" in p)
print("emision %s ART | ciclo SMN %s | corrida INA %s | residuo de %s ART (edad %.1f h) | %d horas (%d con INA)" % (
    fc["emision_art"], fc["ciclo_init_utc"], fc["corrida_ina_t0"], fc["ultimo_residuo_art"], fc["edad_residuo_h"], len(fc["serie"]), n_bw))

# ---------------- 3. guardar
meta = {k: v for k, v in fc.items() if k != "serie"}
with open(os.path.join(DATA, "pronostico_actual.json"), "w") as f:  # un punto por linea (legible y chico)
    f.write("{\n" + ",\n".join(' "%s": %s' % (k, json.dumps(v, ensure_ascii=False)) for k, v in meta.items())
            + ',\n "serie": [\n' + ",\n".join("  " + json.dumps(p, ensure_ascii=False) for p in fc["serie"]) + "\n ]\n}\n")
jl = os.path.join(DATA, "pronosticos_capa.jsonl")
clave = lambda x: (x["emision_art"], x["ciclo_init_utc"], x["corrida_ina_t0"], x["ultimo_residuo_art"])
if os.path.exists(jl):
    last = None
    with open(jl) as f:
        for line in f:
            if line.strip():
                last = line
    if last and clave(json.loads(last)) == clave(fc):
        print("sin cambios respecto de la ultima emision; no se agrega linea")
        sys.exit(0)
ser = fc["serie"]
compacta = {k: v for k, v in fc.items() if k != "serie"}
compacta.update(
    h=[p["h"] for p in ser],
    ours=[p["ours"] for p in ser],
    ina=[p.get("ina") for p in ser],
    bw=[p.get("bw") for p in ser],
)
with open(jl, "a") as f:
    f.write(json.dumps(compacta, ensure_ascii=False, separators=(",", ":")) + "\n")
print("linea agregada a", jl)
