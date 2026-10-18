"""Evaluacion de la capa combinada INA + modelo propio, San Fernando (horario, 1-66 h).

Variantes y funciones en capa_lib.py. Evaluacion:
  (1) validacion cruzada por bloques de 3 dias de emision, con 3 dias de separacion a cada lado y
      la mezcla anidada (los pesos se ajustan solo con datos de entrenamiento);
  (2) prueba fuera de muestra desde el 11/10 entrenando solo con lo anterior;
  IC90 por bootstrap sobre dias para la diferencia de MAE contra el INA.
Salida: tablas por consola y pronostico de las proximas 66 h en <salida>/pronostico_sf_capa.json.

Uso: python3 capa_combinada.py <carpeta repos> [carpeta salida]
"""
import sys, os, json, datetime as dt, statistics as st, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capa_lib as L

R = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "..")
G = L.cargar(R)
rows = L.enriquecer(G)
has = [r for r in rows if r["has"]]
last_obs, runs, cyc = G["last_obs"], G["runs"], G["cyc"]
CUT = dt.datetime(2026, 10, 11)
DAY = L.DAY
METHODS, BANDS = L.METHODS, L.BANDS
print("muestras: %d (con INA: %d), emisiones con INA %s -> %s" % (len(rows), len(has), min(r["emis"] for r in has), max(r["emis"] for r in has)))


def report(title, recs):
    print("\n=== %s  (n=%d horas-objetivo)" % (title, len(recs)))
    print("%-8s" % "horiz" + "".join("%8s" % m for m in METHODS))
    for i, (a, b) in enumerate(BANDS):
        rr = [(r, o) for r, o in recs if a <= r["h"] <= b]
        print("%2d-%2d h  " % (a, b) + "".join("%8.3f" % st.mean(abs(r["obs"] - o[m]) for r, o in rr) for m in METHODS))
    print("%-8s" % "total" + "".join("%8.3f" % st.mean(abs(r["obs"] - o[m]) for r, o in recs) for m in METHODS))
    for m in METHODS:
        print("  %-6s %s" % (m, L.stats([r["obs"] - o[m] for r, o in recs])))


def daily_ci(recs, label, a="bw", b="ina"):
    """Diferencia de MAE por dia (a - b) con IC90 por bootstrap sobre dias (pesos iguales por dia)."""
    g = collections.defaultdict(list)
    for r, o in recs:
        g[r["T"].date()].append(abs(r["obs"] - o[a]) - abs(r["obs"] - o[b]))
    d = np.array([st.mean(v) for _, v in sorted(g.items())])
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
    print("%s: %s - %s por dia: media %+.3f, IC90 [%+.3f, %+.3f], %s mejor en %d de %d dias" % (
        label, a, b, d.mean(), np.percentile(bs, 5), np.percentile(bs, 95), a, int((d < 0).sum()), len(d)))


# ---------------- (1) validacion cruzada por bloques
cv = []
for bs, be, v in L.blocks(has):
    train = [r for r in rows if not (bs - 3 * DAY <= r["emis"] <= be + 3 * DAY)]
    out, W, _ = L.pipeline(train, v)
    cv += list(zip(v, out))
report("VALIDACION CRUZADA por bloques de 3 dias (todo 22/9 -> 18/10)", cv)
pk = [(r, o) for r, o in cv if dt.datetime(2026, 10, 9, 0) <= r["T"] <= dt.datetime(2026, 10, 9, 12)]
report("PICO del 9/10 (objetivos 00-12 h, dentro de la validacion cruzada)", pk)
by = collections.defaultdict(list)
for r, o in cv:
    by[r["T"].date()].append((r, o))
print("\nMAE por dia (ours / ina / bw / stack / corr):")
for d in sorted(by):
    v = by[d]
    print(d, " ".join("%.3f" % st.mean(abs(r["obs"] - o[m]) for r, o in v) for m in ["ours", "ina", "bw", "stack", "corr"]), "n=%d" % len(v))
for m in ["bw", "b50", "stack", "corr"]:
    daily_ci(cv, "VALIDACION CRUZADA", m, "ina")

# ---------------- (2) prueba fuera de muestra desde el 11/10
train = [r for r in rows if r["emis"] < CUT and r["T"] < CUT]
test = [r for r in has if r["emis"] >= CUT and r["T"] >= CUT]
out, W, _ = L.pipeline(train, test)
report("FUERA DE MUESTRA 11/10 -> 18/10 (entrenado solo con lo anterior)", list(zip(test, out)))
for m in ["bw", "b50", "stack", "corr"]:
    daily_ci(list(zip(test, out)), "FUERA DE MUESTRA", m, "ina")
print("pesos del modelo propio (bw) por franja, entrenados hasta el 10/10:", {"%d-%d h" % BANDS[i]: round(float(w), 2) for i, w in W.items()})

# ---------------- ajuste final y pronostico con el ultimo ciclo (convencion de validacion: emision = init + 6 h UTC)
Wf, models = L.ajustar(rows)
print("\npesos finales (todo el archivo):", {"%d-%d h" % BANDS[i]: round(float(w), 2) for i, w in Wf.items()})
fc = None
for k in sorted(cyc, reverse=True):
    emis = G["P"](cyc[k]["ciclo_init"]) + dt.timedelta(hours=3)
    fc = L.serie_ciclo(G, cyc[k], emis, Wf, models)
    if fc:
        break
if fc:
    fc["pesos_bw"] = {"%d-%d h" % BANDS[i]: round(float(v), 3) for i, v in Wf.items()}
    fc["ultimo_observado_art"] = str(last_obs)
    json.dump(fc, open(os.path.join(OUT, "pronostico_sf_capa.json"), "w"), ensure_ascii=False, indent=1)
    ser = [p for p in fc["serie"] if "bw" in p]
    print("\nPRONOSTICO San Fernando: ciclo SMN %s, emision %s ART, corrida INA %s, %d horas" % (fc["ciclo_init_utc"], fc["emision_art"], fc["corrida_ina_t0"], len(fc["serie"])))
    print("extremos (serie bw):")
    for i in range(1, len(ser) - 1):
        a, b, c_ = ser[i - 1]["bw"], ser[i]["bw"], ser[i + 1]["bw"]
        if (b > a and b >= c_) or (b < a and b <= c_):
            p = ser[i]
            print("  %s  %s  bw %.2f | INA %.2f nuestro %.2f stack %.2f corr %.2f" % (p["T"][5:16], "PLE" if b > a else "BAJ", p["bw"], p["ina"], p["ours"], p["stack"], p["corr"]))
else:
    print("no se pudo armar el pronostico (sin ciclo con residuo reciente)")
