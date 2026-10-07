"""
Backtest del residuo de marea en Pilote Norden con meteorología ERA5.

  residuo(t) = marea observada CARP(t) − marea astronómica ajustada (7 constituyentes)

Se modela residuo(t) con viento/presión ERA5 en el estuario y la plataforma,
con validación "leave-one-year-out" (se entrena con todos los años menos uno y
se evalúa en ese año), sobre ~10 años (incluye muchas sudestadas reales).

IMPORTANTE: ERA5 es reanálisis (meteorología "perfecta"), no pronóstico. Este
backtest mide el TECHO de lo que se puede explicar si el pronóstico de viento y
presión fuera exacto. Con pronóstico real el error va a ser mayor y crecer con
el horizonte.

Uso:
  python backtest_residuo.py --carp DIR_CARP --era5 DIR_ERA5 --out resultados.md

  DIR_CARP: carpeta con norden_tide.csv (columnas date_time [ART], tide_height)
  DIR_ERA5: carpeta con era5_<punto>.csv (salida de capturar_era5.py)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# velocidades angulares (grados/hora) de los 7 constituyentes
CONSTITUYENTES = {"M2": 28.9841042, "S2": 30.0, "N2": 28.4397295, "K1": 15.0410686,
                  "O1": 13.9430356, "P1": 14.9589314, "K2": 30.0821373}
UMBRAL_EVENTO = 0.8     # m de residuo positivo: "sudestada"
UMBRAL_ALERTA = 0.5     # m predichos para contar como detectado
LAGS_AR = (6, 24, 48)   # horas hacia atrás para el residuo observado (variante AR)


# ---------------- marea y residuo ----------------
def cargar_marea(carp_dir: Path) -> pd.Series:
    t = pd.read_csv(carp_dir / "norden_tide.csv", parse_dates=["date_time"])
    t["utc"] = t["date_time"] + pd.Timedelta(hours=3)          # CARP viene en ART
    t = t[(t["tide_height"] > -1.5) & (t["tide_height"] < 4.5)]  # fuera de rango = pico/dato inválido
    g = t.set_index("utc")["tide_height"].resample("1h").agg(["median", "count"])
    return g.loc[g["count"] >= 3, "median"].rename("marea")


def ajustar_armonico(h: pd.Series) -> pd.Series:
    horas = (h.index - pd.Timestamp("2000-01-01")).total_seconds().to_numpy() / 3600
    cols = [np.ones_like(horas)]
    for w in CONSTITUYENTES.values():
        a = np.deg2rad(w) * horas
        cols += [np.cos(a), np.sin(a)]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, h.to_numpy(), rcond=None)
    ajuste = pd.Series(X @ beta, index=h.index)
    pct = 1 - np.var(h.to_numpy() - ajuste.to_numpy()) / np.var(h.to_numpy())
    print(f"Ajuste armónico: explica {pct:.1%} de la variación horaria ({len(h):,} horas)")
    return ajuste


# ---------------- meteorología ----------------
def cargar_era5(era5_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(era5_dir.glob("era5_*.csv")):
        d = pd.read_csv(f, parse_dates=["time_utc"]).set_index("time_utc").sort_index()
        d = d[~d.index.duplicated(keep="last")]
        d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="1h"))
        rad = np.deg2rad(d["wind_dir_deg"])
        # convención meteorológica: dir = de dónde viene -> componentes hacia donde va
        d["u"] = -d["wind_speed_ms"] * np.sin(rad)
        d["v"] = -d["wind_speed_ms"] * np.cos(rad)
        d["p"] = d["pressure_msl_hpa"]
        out[f.stem.removeprefix("era5_")] = d[["u", "v", "p"]]
    if "norden" not in out:
        sys.exit("Falta era5_norden.csv")
    return out


def features_punto(d: pd.DataFrame, nombre: str, completo: bool) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)
    f[f"{nombre}_u"], f[f"{nombre}_v"] = d["u"], d["v"]
    for w in (6, 24, 48):
        f[f"{nombre}_u_m{w}"] = d["u"].rolling(w, min_periods=w).mean()
        f[f"{nombre}_v_m{w}"] = d["v"].rolling(w, min_periods=w).mean()
    if completo:
        f[f"{nombre}_p"] = d["p"]
        f[f"{nombre}_dp6"] = d["p"] - d["p"].shift(6)
        f[f"{nombre}_dp24"] = d["p"] - d["p"].shift(24)
    return f


def juegos_de_features(era5):
    A = features_punto(era5["norden"], "norden", completo=False)
    pn = features_punto(era5["norden"], "norden", completo=True)
    B = pn
    C = pd.concat([features_punto(d, n, completo=True) for n, d in era5.items()], axis=1)
    return {"A viento local": A, "B + presión local": B, "C + boca/plataforma": C}


# ---------------- evaluación ----------------
def metricas(y, yhat, nombre):
    e = y - yhat
    ev = y >= UMBRAL_EVENTO
    det = (yhat[ev] >= UMBRAL_ALERTA).mean() if ev.any() else np.nan
    fa = ((yhat >= UMBRAL_ALERTA) & ~ev).sum() / max((yhat >= UMBRAL_ALERTA).sum(), 1)
    return {
        "modelo": nombre, "n": len(y),
        "RMSE_cm": 100 * np.sqrt(np.mean(e ** 2)),
        "R2": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2),
        "n_eventos": int(ev.sum()),
        "RMSE_eventos_cm": 100 * np.sqrt(np.mean(e[ev] ** 2)) if ev.any() else np.nan,
        "detecta_%": 100 * det,
        "falsas_alarmas_%": 100 * fa,
    }


def loyo(X: pd.DataFrame, y: pd.Series, modelo_fn, anios_min=2000):
    """Predicción leave-one-year-out; devuelve Series alineada con y."""
    pred = pd.Series(np.nan, index=y.index)
    anio = y.index.year
    for a in sorted(set(anio)):
        te = anio == a
        if te.sum() < anios_min:
            continue
        m = modelo_fn()
        m.fit(X[~te], y[~te])
        pred[te] = m.predict(X[te])
    return pred


def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=50.0))


def hgb():
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_depth=5,
                                         min_samples_leaf=60, l2_regularization=1.0, random_state=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carp", required=True, type=Path)
    ap.add_argument("--era5", required=True, type=Path)
    ap.add_argument("--out", default="resultados_backtest.md", type=Path)
    ap.add_argument("--sin-hgb", action="store_true", help="solo ridge (más rápido)")
    a = ap.parse_args()

    marea = cargar_marea(a.carp)
    astro = ajustar_armonico(marea)
    res = (marea - astro).rename("residuo")
    era5 = cargar_era5(a.era5)
    juegos = juegos_de_features(era5)

    filas, extras = [], []
    base_idx = res.index
    for nombre, F in juegos.items():
        D = F.join(res, how="inner").dropna()
        y, X = D["residuo"], D.drop(columns="residuo")
        print(f"\n[{nombre}] {X.shape[1]} features, {len(y):,} horas, {y.index.year.min()}-{y.index.year.max()}")
        if nombre == "A viento local":
            filas.append(metricas(y.to_numpy(), np.full(len(y), y.mean()), "referencia: media (sin meteorología)") | {"juego": "—"})
        modelos = {"Ridge": ridge} | ({} if a.sin_hgb else {"HGB": hgb})
        for mn, fn in modelos.items():
            p = loyo(X, y, fn).dropna()
            m = metricas(y.loc[p.index].to_numpy(), p.to_numpy(), mn)
            m["juego"] = nombre
            filas.append(m)
            print(f"  {mn}: RMSE {m['RMSE_cm']:.1f} cm, R² {m['R2']:.3f}, eventos {m['n_eventos']} (detecta {m['detecta_%']:.0f}%)")

    # variante AR: se conoce el residuo observado h horas antes (+ juego C)
    FC = juegos["C + boca/plataforma"]
    for h in LAGS_AR:
        D = FC.join(res, how="inner").join(res.shift(h, freq="1h").rename("r_lag"), how="inner").dropna()
        y, X = D["residuo"], D.drop(columns="residuo")
        for mn, fn in ({"Ridge": ridge} | ({} if a.sin_hgb else {"HGB": hgb})).items():
            p = loyo(X, y, fn).dropna()
            m = metricas(y.loc[p.index].to_numpy(), p.to_numpy(), f"{mn} + residuo observado {h} h antes")
            m["juego"] = "C + AR"
            extras.append(m)
            # persistencia pura a ese lag, misma muestra
        pers = metricas(y.to_numpy(), D["r_lag"].to_numpy(), f"persistencia ({h} h)")
        pers["juego"] = "—"
        extras.append(pers)

    tabla = pd.DataFrame(filas + extras)[["juego", "modelo", "n", "RMSE_cm", "R2", "n_eventos",
                                           "RMSE_eventos_cm", "detecta_%", "falsas_alarmas_%"]]
    md = ["# Backtest residuo de marea — Pilote Norden", "",
          f"Eventos = residuo ≥ {UMBRAL_EVENTO} m; 'detecta' = predicción ≥ {UMBRAL_ALERTA} m cuando hubo evento.",
          "ERA5 = meteorología perfecta: es un techo, no el desempeño del pronóstico real.", "",
          tabla.round(2).to_markdown(index=False) if hasattr(tabla, "to_markdown") else tabla.round(2).to_string(index=False)]
    a.out.write_text("\n".join(md), encoding="utf-8")
    print("\n" + tabla.round(2).to_string(index=False))
    print(f"\nGuardado en {a.out}")


if __name__ == "__main__":
    main()
