"""Serie v2 de la capa (San Fernando, 1-66 h): modelo propio con mas viento + mezcla con el INA.

Cambios respecto de capa_lib.py (que sigue emitiendo la serie vigente):
  - viento acumulado de 12, 24 y 36 h en San Fernando (el historial anterior al inicio del ciclo sale de los
    primeros leads 0-5 h de los ciclos anteriores, que es lo que hay disponible en el momento de emitir);
  - viento de Atalaya y San Clemente del Tuyu (puntos de plataforma): instantaneo y media de 12 h.
El resto (armonico, residuo reciente, Ridge por franja de horizonte, peso por franja con el INA) es igual.

Se usa desde emitir_pronostico_v2.py y analisis/prueba_v2.py (validacion).
"""
import math, datetime as dt, statistics as st, collections
import numpy as np
from sklearn.linear_model import Ridge
import capa_lib as L

H, DAY, BANDS, band = L.H, L.DAY, L.BANDS, L.band
SF, EXTRA = "San Fernando", ["Atalaya", "San Clemente del Tuyú"]
ESTS = [SF] + EXTRA
VARIANTES = {"base": ["sf12"], "acum": ["sf12", "sf_acum"], "v2": ["sf12", "sf_acum", "plat"]}


def _comp(p):
    s, dr = p["viento_10m_ms"], math.radians(p["direccion_10m_deg"])
    return (s * math.cos(dr - math.radians(135)), s * math.cos(dr - math.radians(225)))


def _ciclo_w(d, P):
    return {e: {P(p["valido_para"]) - dt.timedelta(hours=3): _comp(p) for p in d["estaciones"][e]}
            for e in ESTS if e in d["estaciones"]}


def preparar(G):
    """Viento por ciclo + 'analisis' (leads 0-5 h del ciclo mas reciente para cada hora)."""
    P = G["P"]
    cw, A = {}, {e: {} for e in ESTS}
    for k in sorted(G["cyc"]):
        w = _ciclo_w(G["cyc"][k], P)
        cw[k] = w
        for e, m in w.items():
            t0 = min(m) if m else None
            for t, v in m.items():
                if t0 is not None and t - t0 < 6 * H:
                    A[e][t] = v
    return cw, A


def _acc(m, T, n, minimo=1):
    a = [m[T - i * H] for i in range(n) if (T - i * H) in m]
    return (st.mean(x[0] for x in a), st.mean(x[1] for x in a)) if len(a) >= minimo else None


def _serie_completa(cw_e, A_e):
    """Historial (analisis) hasta el inicio del ciclo y el pronostico del ciclo desde alli."""
    t0 = min(cw_e) if cw_e else None
    full = {t: v for t, v in A_e.items() if t0 is None or t < t0}
    full.update(cw_e)
    return full


def features(G, k, cw, A, emis, T, r0, rm):
    """Diccionario de grupos de variables para la hora objetivo T (ART) del ciclo k; None si falta algo."""
    w = {e: _serie_completa(cw[k].get(e, {}), A[e]) for e in ESTS}
    sf = w[SF]
    if T not in sf:
        return None
    a12, a24, a36 = _acc(sf, T, 12, 3), _acc(sf, T, 24, 6), _acc(sf, T, 36, 9)
    if not (a12 and a24 and a36):
        return None
    u, v = sf[T]
    f = dict(sf12=[r0, rm, u, v, a12[0], a12[1]], sf_acum=[a24[0], a24[1], a36[0], a36[1]])
    pl = []
    for e in EXTRA:
        m = w[e]
        aa = _acc(m, T, 12, 3) if m else None
        if T not in m or not aa:
            return None
        pl += [m[T][0], m[T][1], aa[0], aa[1]]
    f["plat"] = pl
    return f


def residuo_reciente(G, emis):
    g = L._ultimo_residuo(G, emis)
    if g is None:
        return None
    l24 = [G["res"][g - i * H] for i in range(24) if (g - i * H) in G["res"]]
    if len(l24) < 12:
        return None
    return g, G["res"][g], st.mean(l24)


def muestras(G):
    """Muestras de entrenamiento: una por (ciclo, hora objetivo) con INA disponible a la emision cuando lo hay."""
    P = G["P"]
    cw, A = preparar(G)
    rows = []
    for k, d in G["cyc"].items():
        emis = P(d["ciclo_init"]) + dt.timedelta(hours=3)
        rr = residuo_reciente(G, emis)
        if rr is None:
            continue
        g, r0, rm = rr
        c = [x for x in G["runs"] if x[0] <= emis]
        run = c[-1][1] if c else {}
        for T in G["gt"]:
            h = (T - emis).total_seconds() / 3600
            if h < 1 or h > 66:
                continue
            f = features(G, k, cw, A, emis, T, r0, rm)
            if f is None:
                continue
            rows.append(dict(k=k, emis=emis, T=T, h=h, y=G["res"][T], ast=G["ast"][T], obs=G["grid"][T], f=f,
                             ina=run.get(T), has=run.get(T) is not None))
    return rows


def X_de(rows, grupos):
    return np.array([sum((r["f"][g] for g in grupos), []) for r in rows], float)


def ajustar(rows, grupos, alpha=10):
    """Ridge por franja + peso por franja con el INA (con predicciones fuera de bloque, como capa_lib.ajustar)."""
    M = {}
    for i in range(len(BANDS)):
        rr = [r for r in rows if band(r["h"]) == i]
        X = X_de(rr, grupos)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        M[i] = (Ridge(alpha=alpha).fit((X - mu) / sd, np.array([r["y"] for r in rr])), mu, sd)
    tr_has = [r for r in rows if r["has"]]
    oof = {}
    for bs, be, v in L.blocks(tr_has):
        inner = [r for r in rows if not (bs - 3 * DAY <= r["emis"] <= be + 3 * DAY)]
        Mi = ajustar_ridge(inner, grupos, alpha)
        for r in v:
            oof[id(r)] = r["ast"] + pred(Mi, r, grupos)
    W = {}
    for i in range(len(BANDS)):
        rr = [r for r in tr_has if band(r["h"]) == i]
        num = sum((r["obs"] - r["ina"]) * (oof[id(r)] - r["ina"]) for r in rr)
        den = sum((oof[id(r)] - r["ina"]) ** 2 for r in rr)
        W[i] = min(1.0, max(0.0, num / den)) if den > 0 else 0.5
    return M, W


def ajustar_ridge(rows, grupos, alpha=10):
    M = {}
    for i in range(len(BANDS)):
        rr = [r for r in rows if band(r["h"]) == i]
        X = X_de(rr, grupos)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        M[i] = (Ridge(alpha=alpha).fit((X - mu) / sd, np.array([r["y"] for r in rr])), mu, sd)
    return M


def pred(M, r, grupos):
    m, mu, sd = M[band(r["h"])]
    x = np.array(sum((r["f"][g] for g in grupos), []), float)
    return float(m.predict(((x - mu) / sd)[None, :])[0])


def emitir(G, grupos, M, W, d_ciclo, emis):
    """Serie 1-66 h (ours2, ina, bw2) para el ciclo d_ciclo y la hora de emision emis; None si falta residuo."""
    P = G["P"]
    k = d_ciclo["ciclo_init"]
    cw, A = preparar(G)
    rr = residuo_reciente(G, emis)
    if rr is None:
        return None
    g, r0, rm = rr
    c = [x for x in G["runs"] if x[0] <= emis]
    run = c[-1][1] if c else {}
    pts = []
    for h in range(1, 67):
        T = emis + h * H
        f = features(G, k, cw, A, emis, T, r0, rm)
        if f is None:
            continue
        r = dict(h=h, f=f)
        a_ = float(G["astro"]([T])[0])
        po = a_ + pred(M, r, grupos)
        row = dict(h=h, ours=round(po, 3), astro=round(a_, 3))
        ina = run.get(T)
        if ina is not None:
            wb = W[band(h)]
            row.update(ina=round(ina, 3), bw=round(wb * po + (1 - wb) * ina, 3))
        pts.append(row)
    return dict(emision_art=emis.isoformat(), ciclo_init_utc=d_ciclo["ciclo_init"],
                corrida_ina_t0=c[-1][0].isoformat() if c else None,
                ultimo_residuo_art=g.isoformat(), residuo_r0=round(r0, 3), serie=pts)
