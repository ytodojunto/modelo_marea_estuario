"""Datos y modelo base de San Fernando (sin evaluacion).

Carga observado, armonico, grilla horaria, ciclos SMN, muestras de entrenamiento y corridas INA.
Variables de entorno opcionales:
  SF_EXTRA_JSON  ruta a un JSON {"YYYY-mm-ddTHH:MM": altura_m} (hora ART) con observado mas reciente.
  AHORA          instante UTC (ISO) para reproducir lo que estaba disponible en ese momento: descarta
                 observado, ciclos SMN y capturas INA posteriores.
  EMIS_OFFSET_H  horas (hora ART) entre el init del ciclo y la emision usada para armar las muestras
                 (por defecto 3, o sea init + 6 h UTC, que es la convencion con la que se valido todo).
"""
import os, json, sys, glob, csv, math, datetime as dt, collections, statistics as st
import numpy as np
from sklearn.linear_model import Ridge

R = sys.argv[1]
OFF = float(os.environ.get("EMIS_OFFSET_H", "3"))
H = dt.timedelta(hours=1)
P = lambda s: dt.datetime.fromisoformat(s.replace("Z", "")[:19])

# ---------------- observado San Fernando (ART naive)
sf = {}
for r in csv.DictReader(open(f"{R}/INA_captura_delta_2/data/series/san_fernando.csv")):
    try:
        sf[P(r["timestart"])] = float(r["valor"])
    except Exception:
        pass
for line in open(f"{R}/SHN_captura_pronosticos/data/historico_alturas.jsonl"):
    for m in json.loads(line).get("mediciones", []):
        if m["estacion"] == "San Fernando":
            try:
                sf[dt.datetime.strptime(m["fecha_hora"], "%d/%m/%Y %H:%M")] = float(m["altura_m"])
            except Exception:
                pass
_extra = os.environ.get("SF_EXTRA_JSON")
if _extra and os.path.exists(_extra):
    for k, v in json.load(open(_extra)).items():
        sf[P(k)] = float(v)
AHORA = P(os.environ["AHORA"]) if os.environ.get("AHORA") else None  # UTC naive
if AHORA is not None:
    sf = {t: v for t, v in sf.items() if t <= AHORA - dt.timedelta(hours=3)}
last_obs = max(sf)

# ---------------- armonico (17 constituyentes) con datos 2021-01 -> 2026-09-09
DEG = dict(M2=28.9841042, S2=30.0, N2=28.4397295, K2=30.0821373, K1=15.0410686, O1=13.9430356, P1=14.9589314,
           Q1=13.3986609, M4=57.9682084, MS4=58.9841042, MN4=57.4238337, M6=86.9523127, **{"2N2": 27.8953548},
           MU2=27.9682084, NU2=28.5125831, L2=29.5284789, T2=29.9589333)
W = np.array([math.radians(v) for v in DEG.values()])
T0 = dt.datetime(2000, 1, 1)
th = lambda ts: np.array([(t - T0).total_seconds() / 3600 for t in ts])

def design(ts):
    x = th(ts)[:, None] * W[None, :]
    return np.hstack([np.ones((len(ts), 1)), np.cos(x), np.sin(x)])

fit_ts = sorted(t for t in sf if dt.datetime(2021, 1, 1) <= t < dt.datetime(2026, 9, 9))
y = np.array([sf[t] for t in fit_ts])
coef, *_ = np.linalg.lstsq(design(fit_ts), y, rcond=None)
pred = design(fit_ts) @ coef
print("armonico: %d h de ajuste, explica %.1f %% de la variacion" % (len(fit_ts), 100 * (1 - np.var(y - pred) / np.var(y))))
astro = lambda ts: design(ts) @ coef

# ---------------- grilla horaria HH:00 interpolada
def at(t):
    a, b = t - dt.timedelta(minutes=15), t + dt.timedelta(minutes=45)
    return 0.75 * sf[a] + 0.25 * sf[b] if a in sf and b in sf else None

g0 = dt.datetime(2026, 8, 31, 0, 0)
grid = {}
t = g0
while t <= last_obs:
    v = at(t)
    if v is not None:
        grid[t] = v
    t += H
gt = sorted(grid)
ast = dict(zip(gt, astro(gt)))
res = {t: grid[t] - ast[t] for t in gt}

# ---------------- ciclos de viento (San Fernando)
cyc = {}
for line in open(f"{R}/SMN_captura_viento/data/historico_viento.jsonl"):
    d = json.loads(line)
    if d["archivos_procesados"] < 73:
        continue
    if AHORA is not None and P(d["capturado_en"]) > AHORA:
        continue
    k = d["ciclo_init"]
    if k not in cyc or d["archivos_procesados"] > cyc[k]["archivos_procesados"]:
        cyc[k] = d
print("ciclos completos:", len(cyc))

def build_cycle(d):
    init = P(d["ciclo_init"])  # UTC
    emis = init + dt.timedelta(hours=OFF)  # emision en hora ART (3 h = init + 6 h UTC)
    w = {}
    for p in d["estaciones"]["San Fernando"]:
        tt = P(p["valido_para"]) - dt.timedelta(hours=3)
        s, dr = p["viento_10m_ms"], math.radians(p["direccion_10m_deg"])
        w[tt] = (s * math.cos(dr - math.radians(135)), s * math.cos(dr - math.radians(225)))
    # residuo observado reciente
    g = None
    tt = emis - H
    while tt >= emis - 6 * H:
        if tt in res:
            g = tt
            break
        tt -= H
    if g is None:
        return None
    r0 = res[g]
    last24 = [res[g - i * H] for i in range(24) if (g - i * H) in res]
    if len(last24) < 12:
        return None
    rm = st.mean(last24)
    rows = []
    for T in gt:
        h = (T - emis).total_seconds() / 3600
        if h < 1 or h > 66 or T not in w:
            continue
        acc = [w[T - i * H] for i in range(12) if (T - i * H) in w]
        uw = (st.mean(a[0] for a in acc), st.mean(a[1] for a in acc))
        rows.append(dict(c=d["ciclo_init"], emis=emis, T=T, h=h, x=[r0, rm, w[T][0], w[T][1], uw[0], uw[1]], y=res[T], ast=ast[T], obs=grid[T]))
    return rows

S = []
for d in cyc.values():
    r = build_cycle(d)
    if r:
        S += r
print("muestras:", len(S), " ultimo observado:", last_obs)

BANDS = [(1, 6), (7, 12), (13, 24), (25, 48), (49, 66)]
band = lambda h: next(i for i, (a, b) in enumerate(BANDS) if a <= h <= b)

def train(cut_emis, cut_T):
    models = {}
    tr = [s for s in S if s["emis"] < cut_emis and s["T"] < cut_T]
    for i in range(len(BANDS)):
        rr = [s for s in tr if band(s["h"]) == i]
        X = np.array([s["x"] for s in rr])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        m = Ridge(alpha=10).fit((X - mu) / sd, np.array([s["y"] for s in rr]))
        models[i] = (m, mu, sd, len(rr))
    return models

def predict(models, s):
    m, mu, sd, _ = models[band(s["h"])]
    return s["ast"] + float(m.predict(((np.array(s["x"]) - mu) / sd)[None, :])[0])

# ---------------- INA corridas SF
runs = {}
for f in sorted(glob.glob(f"{R}/INA_captura_delta_2/data/historico_prono/*.json")):
    d = json.load(open(f))
    if AHORA is not None and P(d.get("capturado_en") or d.get("generado_en")) > AHORA:
        continue
    e = d["estaciones"].get("san_fernando")
    if e and e["corid"] not in runs:
        runs[e["corid"]] = (P(e["corrida_forecastdate"]), {P(x["timestart"]): x["central"] for x in e["serie"] if x.get("central") is not None})
runs = sorted(runs.values(), key=lambda x: x[0])

def ina_for(emis, T):
    c = [r for r in runs if r[0] <= emis]
    return c[-1][1].get(T) if c else None

def stats(er):
    if not er:
        return "n=0"
    return "n=%3d MAE %.3f RMSE %.3f sesgo %+.3f" % (len(er), st.mean(abs(x) for x in er), st.mean(x * x for x in er) ** .5, st.mean(er))
