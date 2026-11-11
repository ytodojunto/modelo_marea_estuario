"""Emite la serie v2 de la capa (San Fernando, 1-66 h) en paralelo a la vigente, sin tocarla.

Igual que emitir_pronostico.py pero con capa_v2.py (viento acumulado 24/36 h + viento de Atalaya y San Clemente).
Escribe en <data>/pronosticos_capa_v2.jsonl con las MISMAS claves que la vigente (h, ours, ina, bw), de modo que
evaluar_emitidos.py sirve sin cambios, y reescribe <data>/pronostico_actual_v2.json.

Uso: python3 emitir_pronostico_v2.py <carpeta repos> <carpeta data> [--ahora ISO_UTC] [--sin-vivo]
"""
import sys, os, json, datetime as dt, tempfile

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
            p = os.path.join(tempfile.gettempdir(), "sf_obs_vivo_v2.json")
            json.dump(extra, open(p, "w"))
            os.environ["SF_EXTRA_JSON"] = p
            obs_vivo = True
            print("observado en vivo: %d horas, hasta %s ART" % (len(extra), max(extra)))
        else:
            print("AVISO: la descarga en vivo no trajo datos de San Fernando; se sigue con las capturas del repo")
    except Exception as e:
        print("AVISO: no se pudo bajar el observado en vivo (%s: %s); se sigue con las capturas del repo" % (type(e).__name__, e))

import capa_lib as L
import capa_v2 as V

GR = V.VARIANTES["v2"]
G = L.cargar(R, 3)
rows = V.muestras(G)
if len(rows) < 500:
    print("ERROR: pocas muestras de entrenamiento (%d)" % len(rows))
    sys.exit(1)
M, W = V.ajustar(rows, GR)
cyc, P = G["cyc"], G["P"]
if not cyc:
    print("ERROR: no hay ciclos de viento completos")
    sys.exit(1)
d = cyc[max(cyc)]
if emis < P(d["ciclo_init"]) - dt.timedelta(hours=3):
    print("ERROR: el ciclo mas reciente (%s) es posterior a la hora de emision" % d["ciclo_init"])
    sys.exit(1)
fc = V.emitir(G, GR, M, W, d, emis)
if fc is None or not fc["serie"]:
    print("SIN EMISION v2: falta residuo observado reciente o datos de viento (ultimo observado %s ART)" % G["last_obs"])
    sys.exit(0)
fc.update(generado_en_utc=now_utc.isoformat(), obs_vivo=obs_vivo, ultimo_observado_art=str(G["last_obs"]),
          edad_residuo_h=round((emis - P(fc["ultimo_residuo_art"])).total_seconds() / 3600, 1),
          pesos_bw={"%d-%d h" % L.BANDS[i]: round(float(v), 3) for i, v in W.items()},
          variables="v2", cantidad_muestras=len(rows))
ser = fc["serie"]
print("emision v2 %s ART | ciclo SMN %s | corrida INA %s | %d horas (%d con INA)" % (
    fc["emision_art"], fc["ciclo_init_utc"], fc["corrida_ina_t0"], len(ser), sum(1 for p in ser if "bw" in p)))

meta = {k: v for k, v in fc.items() if k != "serie"}
with open(os.path.join(DATA, "pronostico_actual_v2.json"), "w") as f:
    f.write("{\n" + ",\n".join(' "%s": %s' % (k, json.dumps(v, ensure_ascii=False)) for k, v in meta.items())
            + ',\n "serie": [\n' + ",\n".join("  " + json.dumps(p, ensure_ascii=False) for p in ser) + "\n ]\n}\n")
jl = os.path.join(DATA, "pronosticos_capa_v2.jsonl")
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
compacta = {k: v for k, v in fc.items() if k != "serie"}
compacta.update(h=[p["h"] for p in ser], ours=[p["ours"] for p in ser], ina=[p.get("ina") for p in ser], bw=[p.get("bw") for p in ser])
with open(jl, "a") as f:
    f.write(json.dumps(compacta, ensure_ascii=False, separators=(",", ":")) + "\n")
print("linea agregada a", jl)
