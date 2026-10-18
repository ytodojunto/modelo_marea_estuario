"""Capa combinada INA + modelo propio, San Fernando (horario, 1-66 h).

Variantes (por franja de horizonte):
  ours   modelo propio (armonico + Ridge con residuo reciente y viento SMN)
  ina    INA tal cual (corrida mas reciente con hora <= emision)
  b50    mezcla 50/50 ours + ina
  bw     mezcla con un peso por franja, ajustado en el entrenamiento con predicciones fuera de muestra del modelo propio
  stack  Ridge con las mismas variables del modelo propio + (INA - astronomia)
  corr   INA + correccion Ridge (error reciente del INA, media 24 h del error, viento SMN, INA - astronomia)

Evaluacion: (1) validacion cruzada por bloques de 3 dias de emision, con 3 dias de separacion
a cada lado, y la mezcla anidada (los pesos se ajustan solo con datos de entrenamiento);
(2) prueba fuera de muestra desde el 11/10 entrenando solo con lo anterior.
Salida: tablas por consola y pronostico de las proximas 66 h en <salida>/pronostico_sf_capa.json.

Uso: python3 -I capa_combinada.py <carpeta repos> [carpeta salida]
"""
import sys, os, io, contextlib, runpy, math, json, datetime as dt, statistics as st, collections
import numpy as np
from sklearn.linear_model import Ridge

R = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "..")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x", R]
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(os.path.join(HERE, "modelo_sf.py"))
S, res, ast, grid, gt, runs, last_obs, BANDS, band, cyc, P, astro = (
    G[k] for k in "S res ast grid gt runs last_obs BANDS band cyc P astro".split())
H = dt.timedelta(hours=1)
DAY = dt.timedelta(days=1)
CUT = dt.datetime(2026, 10, 11)
METHODS = ["ours", "ina", "b50", "bw", "stack", "corr"]

# ---------------- muestras enriquecidas con INA
rows = []
for s in S:
    r = dict(s)
    r["has"] = False
    em = s["emis"]
    c = [x for x in runs if x[0] <= em]
    if c:
        run = c[-1][1]
        if s["T"] in run:
            gg, t = None, em - H
            while t >= em - 6 * H:
                if t in res:
                    gg = t
                    break
                t -= H
            if gg is not None and gg in run:
                es = [grid[gg - i * H] - run[gg - i * H] for i in range(24) if (gg - i * H) in grid and (gg - i * H) in run]
                if len(es) >= 6:
                    r.update(has=True, ina=run[s["T"]], e0=grid[gg] - run[gg], e24=st.mean(es))
    rows.append(r)
has = [r for r in rows if r["has"]]
print("muestras: %d (con INA: %d), emisiones con INA %s -> %s" % (len(rows), len(has), min(r["emis"] for r in has), max(r["emis"] for r in has)))

f_ours = lambda r: r["x"]
f_stack = lambda r: r["x"] + [r["ina"] - r["ast"]]
f_corr = lambda r: [r["e0"], r["e24"]] + r["x"][2:] + [r["ina"] - r["ast"]]
y_ours = lambda r: r["obs"] - r["ast"]
y_corr = lambda r: r["obs"] - r["ina"]


def fit(rs, feat, ytar, alpha=10):
    M = {}
    for i in range(len(BANDS)):
        rr = [r for r in rs if band(r["h"]) == i]
        X = np.array([feat(r) for r in rr])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        M[i] = (Ridge(alpha=alpha).fit((X - mu) / sd, np.array([ytar(r) for r in rr])), mu, sd)
    return M


def pred(M, r, feat):
    m, mu, sd = M[band(r["h"])]
    return float(m.predict(((np.array(feat(r)) - mu) / sd)[None, :])[0])


def blocks(rs, days=3):
    d0 = min(r["emis"] for r in rs).replace(hour=0, minute=0)
    g = collections.defaultdict(list)
    for r in rs:
        g[int((r["emis"] - d0).total_seconds() // (days * 86400))].append(r)
    return [(min(r["emis"] for r in v), max(r["emis"] for r in v), v) for _, v in sorted(g.items())]


def pipeline(train, test):
    tr_has = [r for r in train if r["has"]]
    M_o, M_s, M_c = fit(train, f_ours, y_ours), fit(tr_has, f_stack, y_ours), fit(tr_has, f_corr, y_corr)
    oof = {}
    for bs, be, v in blocks(tr_has):
        inner = [r for r in train if not (bs - 3 * DAY <= r["emis"] <= be + 3 * DAY)]
        Mi = fit(inner, f_ours, y_ours)
        for r in v:
            oof[id(r)] = r["ast"] + pred(Mi, r, f_ours)
    W = {}
    for i in range(len(BANDS)):
        rr = [r for r in tr_has if band(r["h"]) == i]
        num = sum((r["obs"] - r["ina"]) * (oof[id(r)] - r["ina"]) for r in rr)
        den = sum((oof[id(r)] - r["ina"]) ** 2 for r in rr)
        W[i] = min(1.0, max(0.0, num / den)) if den > 0 else 0.5
    out = []
    for r in test:
        po, w = r["ast"] + pred(M_o, r, f_ours), W[band(r["h"])]
        out.append(dict(ours=po, ina=r["ina"], b50=0.5 * (po + r["ina"]), bw=w * po + (1 - w) * r["ina"],
                        stack=r["ast"] + pred(M_s, r, f_stack), corr=r["ina"] + pred(M_c, r, f_corr)))
    return out, W, (M_o, M_s, M_c)


def stats(er):
    return "MAE %.3f RMSE %.3f sesgo %+.3f" % (st.mean(abs(x) for x in er), st.mean(x * x for x in er) ** .5, st.mean(er))


def report(title, recs):
    print("\n=== %s  (n=%d horas-objetivo)" % (title, len(recs)))
    print("%-8s" % "horiz" + "".join("%8s" % m for m in METHODS))
    for i, (a, b) in enumerate(BANDS):
        rr = [(r, o) for r, o in recs if a <= r["h"] <= b]
        print("%2d-%2d h  " % (a, b) + "".join("%8.3f" % st.mean(abs(r["obs"] - o[m]) for r, o in rr) for m in METHODS))
    print("%-8s" % "total" + "".join("%8.3f" % st.mean(abs(r["obs"] - o[m]) for r, o in recs) for m in METHODS))
    for m in METHODS:
        print("  %-6s %s" % (m, stats([r["obs"] - o[m] for r, o in recs])))


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
for bs, be, v in blocks(has):
    train = [r for r in rows if not (bs - 3 * DAY <= r["emis"] <= be + 3 * DAY)]
    out, W, _ = pipeline(train, v)
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
out, W, _ = pipeline(train, test)
report("FUERA DE MUESTRA 11/10 -> 18/10 (entrenado solo con lo anterior)", list(zip(test, out)))
for m in ["bw", "b50", "stack", "corr"]:
    daily_ci(list(zip(test, out)), "FUERA DE MUESTRA", m, "ina")
print("pesos del modelo propio (bw) por franja, entrenados hasta el 10/10:", {"%d-%d h" % BANDS[i]: round(w, 2) for i, w in W.items()})

# ---------------- ajuste final y pronostico de las proximas 66 h
out_all, Wf, (M_o, M_s, M_c) = pipeline(rows, [])
print("\npesos finales (todo el archivo):", {"%d-%d h" % BANDS[i]: round(w, 2) for i, w in Wf.items()})
fc = None
for k in sorted(cyc, reverse=True):
    d = cyc[k]
    emis = P(d["ciclo_init"]) + dt.timedelta(hours=3)
    w = {}
    for p in d["estaciones"]["San Fernando"]:
        tt = P(p["valido_para"]) - dt.timedelta(hours=3)
        s_, dr = p["viento_10m_ms"], math.radians(p["direccion_10m_deg"])
        w[tt] = (s_ * math.cos(dr - math.radians(135)), s_ * math.cos(dr - math.radians(225)))
    gg, t = None, emis - H
    while t >= emis - 6 * H:
        if t in res:
            gg = t
            break
        t -= H
    c = [x for x in runs if x[0] <= emis]
    if gg is None or not c:
        continue
    run = c[-1][1]
    last24 = [res[gg - i * H] for i in range(24) if (gg - i * H) in res]
    es = [grid[gg - i * H] - run[gg - i * H] for i in range(24) if (gg - i * H) in grid and (gg - i * H) in run]
    if len(last24) < 12 or gg not in run or len(es) < 6:
        continue
    r0, rm, e0, e24 = res[gg], st.mean(last24), grid[gg] - run[gg], st.mean(es)
    pts = []
    for h in range(1, 67):
        T = emis + h * H
        if T not in w:
            continue
        acc = [w[T - i * H] for i in range(12) if (T - i * H) in w]
        x = [r0, rm, w[T][0], w[T][1], st.mean(a[0] for a in acc), st.mean(a[1] for a in acc)]
        a_ = float(astro([T])[0])
        r = dict(h=h, x=x, ast=a_, ina=run.get(T), e0=e0, e24=e24)
        po = a_ + pred(M_o, r, f_ours)
        row = dict(T=T.isoformat(), h=h, astro=round(a_, 3), ours=round(po, 3))
        if r["ina"] is not None:
            wb = Wf[band(h)]
            row.update(ina=round(r["ina"], 3), bw=round(wb * po + (1 - wb) * r["ina"], 3),
                       stack=round(a_ + pred(M_s, r, f_stack), 3), corr=round(r["ina"] + pred(M_c, r, f_corr), 3))
        pts.append(row)
    fc = dict(ciclo_init_utc=d["ciclo_init"], emision_art=emis.isoformat(), corrida_ina_t0=c[-1][0].isoformat(),
              ultimo_residuo_observado=round(r0, 3), ultimo_observado_art=str(last_obs), pesos_bw={"%d-%d h" % BANDS[i]: round(v, 3) for i, v in Wf.items()}, serie=pts)
    break
if fc:
    json.dump(fc, open(os.path.join(OUT, "pronostico_sf_capa.json"), "w"), ensure_ascii=False, indent=1)
    print("\nPRONOSTICO San Fernando: ciclo SMN %s, emision %s ART, corrida INA %s, %d horas" % (fc["ciclo_init_utc"], fc["emision_art"], fc["corrida_ina_t0"], len(fc["serie"])))
    ser = [p for p in fc["serie"] if "bw" in p]
    print("extremos (serie bw):")
    for i in range(1, len(ser) - 1):
        a, b, c_ = ser[i - 1]["bw"], ser[i]["bw"], ser[i + 1]["bw"]
        if (b > a and b >= c_) or (b < a and b <= c_):
            p = ser[i]
            print("  %s  %s  bw %.2f | INA %.2f nuestro %.2f stack %.2f corr %.2f" % (p["T"][5:16], "PLE" if b > a else "BAJ", p["bw"], p["ina"], p["ours"], p["stack"], p["corr"]))
else:
    print("no se pudo armar el pronostico (sin ciclo con residuo reciente)")
