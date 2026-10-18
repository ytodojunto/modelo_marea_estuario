"""Biblioteca de la capa combinada INA + modelo propio (San Fernando, 1-66 h).

Usada por capa_combinada.py (evaluacion), emision_rodante.py (sensibilidad a la hora de emision),
emitir_pronostico.py (emision en tiempo real) y evaluar_emitidos.py.

Variantes por franja de horizonte:
  ours   modelo propio (armonico + Ridge con residuo reciente y viento SMN)
  ina    INA tal cual (corrida mas reciente con hora <= emision)
  b50    mezcla 50/50 ours + ina
  bw     mezcla con un peso por franja, ajustado con predicciones fuera de muestra del modelo propio
  stack  Ridge con las variables del modelo propio + (INA - astronomia)
  corr   INA + correccion Ridge (error reciente del INA, viento, INA - astronomia)
"""
import os, sys, io, contextlib, runpy, math, datetime as dt, statistics as st, collections
import numpy as np
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
H = dt.timedelta(hours=1)
DAY = dt.timedelta(days=1)
METHODS = ["ours", "ina", "b50", "bw", "stack", "corr"]
BANDS = [(1, 6), (7, 12), (13, 24), (25, 48), (49, 66)]
band = lambda h: next(i for i, (a, b) in enumerate(BANDS) if a <= h <= b)

f_ours = lambda r: r["x"]
f_stack = lambda r: r["x"] + [r["ina"] - r["ast"]]
f_corr = lambda r: [r["e0"], r["e24"]] + r["x"][2:] + [r["ina"] - r["ast"]]
y_ours = lambda r: r["obs"] - r["ast"]
y_corr = lambda r: r["obs"] - r["ina"]


def cargar(R, offset_h=3, silencioso=True):
    """Corre sf_datos.py y devuelve su espacio de nombres (observado, armonico, ciclos SMN, muestras, corridas INA)."""
    old_argv, old_off = sys.argv, os.environ.get("EMIS_OFFSET_H")
    sys.argv = ["x", R]
    os.environ["EMIS_OFFSET_H"] = str(offset_h)
    try:
        if silencioso:
            with contextlib.redirect_stdout(io.StringIO()):
                G = runpy.run_path(os.path.join(HERE, "sf_datos.py"))
        else:
            G = runpy.run_path(os.path.join(HERE, "sf_datos.py"))
    finally:
        sys.argv = old_argv
        if old_off is None:
            os.environ.pop("EMIS_OFFSET_H", None)
        else:
            os.environ["EMIS_OFFSET_H"] = old_off
    return {k: v for k, v in G.items() if not k.startswith("__")}


def _ultimo_residuo(G, emis):
    """Ultima hora con residuo observado entre emis-6 h y emis-1 h, o None."""
    t = emis - H
    while t >= emis - 6 * H:
        if t in G["res"]:
            return t
        t -= H
    return None


def enriquecer(G):
    """Agrega a cada muestra el INA disponible a la emision (campo has/ina/e0/e24)."""
    res, grid, runs = G["res"], G["grid"], G["runs"]
    rows = []
    for s in G["S"]:
        r = dict(s)
        r["has"] = False
        em = s["emis"]
        c = [x for x in runs if x[0] <= em]
        if c:
            run = c[-1][1]
            if s["T"] in run:
                gg = _ultimo_residuo(G, em)
                if gg is not None and gg in run:
                    es = [grid[gg - i * H] - run[gg - i * H] for i in range(24) if (gg - i * H) in grid and (gg - i * H) in run]
                    if len(es) >= 6:
                        r.update(has=True, ina=run[s["T"]], e0=grid[gg] - run[gg], e24=st.mean(es))
        rows.append(r)
    return rows


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


def ajustar(train):
    """Ajusta modelo propio, stack, corr y los pesos de la mezcla (con predicciones fuera de muestra del modelo propio)."""
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
    return W, (M_o, M_s, M_c)


def predecir(test, W, models):
    M_o, M_s, M_c = models
    out = []
    for r in test:
        po, w = r["ast"] + pred(M_o, r, f_ours), W[band(r["h"])]
        out.append(dict(ours=po, ina=r["ina"], b50=0.5 * (po + r["ina"]), bw=w * po + (1 - w) * r["ina"],
                        stack=r["ast"] + pred(M_s, r, f_stack), corr=r["ina"] + pred(M_c, r, f_corr)))
    return out


def pipeline(train, test):
    W, models = ajustar(train)
    return predecir(test, W, models), W, models


def stats(er):
    return "MAE %.3f RMSE %.3f sesgo %+.3f" % (st.mean(abs(x) for x in er), st.mean(x * x for x in er) ** .5, st.mean(er))


def serie_ciclo(G, d, emis, W, models):
    """Pronostico horario (1-66 h desde emis) con el ciclo SMN d y la ultima corrida INA con hora <= emis.
    Devuelve None si no hay residuo observado reciente (entre emis-6 h y emis-1 h)."""
    res, grid, runs, astro, P = G["res"], G["grid"], G["runs"], G["astro"], G["P"]
    M_o, M_s, M_c = models
    w = {}
    for p in d["estaciones"]["San Fernando"]:
        tt = P(p["valido_para"]) - dt.timedelta(hours=3)
        s_, dr = p["viento_10m_ms"], math.radians(p["direccion_10m_deg"])
        w[tt] = (s_ * math.cos(dr - math.radians(135)), s_ * math.cos(dr - math.radians(225)))
    gg = _ultimo_residuo(G, emis)
    if gg is None:
        return None
    c = [x for x in runs if x[0] <= emis]
    run = c[-1][1] if c else {}
    last24 = [res[gg - i * H] for i in range(24) if (gg - i * H) in res]
    if len(last24) < 12:
        return None
    es = [grid[gg - i * H] - run[gg - i * H] for i in range(24) if (gg - i * H) in grid and (gg - i * H) in run] if run else []
    r0, rm = res[gg], st.mean(last24)
    e0 = grid[gg] - run[gg] if gg in run else None
    e24 = st.mean(es) if len(es) >= 6 else None
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
        if r["ina"] is not None and e0 is not None and e24 is not None:
            wb = W[band(h)]
            row.update(ina=round(r["ina"], 3), bw=round(wb * po + (1 - wb) * r["ina"], 3),
                       stack=round(a_ + pred(M_s, r, f_stack), 3), corr=round(r["ina"] + pred(M_c, r, f_corr), 3))
        pts.append(row)
    return dict(emision_art=emis.isoformat(), ciclo_init_utc=d["ciclo_init"],
                corrida_ina_t0=c[-1][0].isoformat() if c else None,
                ultimo_residuo_art=gg.isoformat(), residuo_r0=round(r0, 3), serie=pts)
