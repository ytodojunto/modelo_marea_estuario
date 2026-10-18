"""Evalua los pronosticos emitidos (pronosticos_capa.jsonl) contra lo observado en San Fernando.

Cada linea emitida trae listas alineadas por h (horas desde la emision): ours (modelo propio), ina y bw
(capa combinada). Solo se comparan horas con las tres series (mismo conjunto para todos). El observado
sale del archivo del INA (san_fernando.csv) y de las capturas del SHN, interpolado a HH:00.

Uso: python3 evaluar_emitidos.py <carpeta repos> <pronosticos_capa.jsonl> [salida.json]
"""
import sys, os, json, csv, datetime as dt, statistics as st, collections

R, JL = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else None
P = lambda s: dt.datetime.fromisoformat(s.replace("Z", "")[:19])
H = dt.timedelta(hours=1)
BANDS = [(1, 6), (7, 12), (13, 24), (25, 48), (49, 66)]
METODOS = ["ours", "ina", "bw"]

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
extra = os.environ.get("SF_EXTRA_JSON")
if extra and os.path.exists(extra):
    for k, v in json.load(open(extra)).items():
        sf[P(k)] = float(v)


def obs_at(t):
    a, b = t - dt.timedelta(minutes=15), t + dt.timedelta(minutes=45)
    return 0.75 * sf[a] + 0.25 * sf[b] if a in sf and b in sf else None


recs = []  # (emision, h, obs, ours, ina, bw)
n_lineas = 0
for line in open(JL):
    if not line.strip():
        continue
    d = json.loads(line)
    n_lineas += 1
    em = P(d["emision_art"])
    for h, o, i, b in zip(d["h"], d["ours"], d["ina"], d["bw"]):
        if o is None or i is None or b is None:
            continue
        ob = obs_at(em + h * H)
        if ob is not None:
            recs.append((em, h, ob, o, i, b))
print("emisiones: %d | horas-objetivo con observado y las tres series: %d" % (n_lineas, len(recs)))
if not recs:
    print("todavia no hay observado para comparar")
    sys.exit(0)
print("emisiones desde %s hasta %s ART" % (min(r[0] for r in recs), max(r[0] for r in recs)))


def fila(rs):
    return [st.mean(abs(r[2] - r[3 + k]) for r in rs) for k in range(3)]


res = {"n_emisiones": n_lineas, "n_horas": len(recs), "por_horizonte": {}, "por_dia_de_emision": {}}
print("\n%-8s %6s %8s %8s %8s   (MAE en m)" % ("horiz", "n", *METODOS))
for a, b in BANDS:
    rs = [r for r in recs if a <= r[1] <= b]
    if rs:
        f = fila(rs)
        print("%2d-%2d h  %6d %8.3f %8.3f %8.3f" % (a, b, len(rs), *f))
        res["por_horizonte"]["%d-%d" % (a, b)] = dict(n=len(rs), **{m: round(v, 4) for m, v in zip(METODOS, f)})
f = fila(recs)
print("total    %6d %8.3f %8.3f %8.3f" % (len(recs), *f))
res["total"] = dict(n=len(recs), **{m: round(v, 4) for m, v in zip(METODOS, f)})
for k, m in enumerate(METODOS):
    er = [r[2] - r[3 + k] for r in recs]
    print("  %-5s RMSE %.3f sesgo %+.3f" % (m, st.mean(e * e for e in er) ** .5, st.mean(er)))
dia = collections.defaultdict(list)
for r in recs:
    dia[r[0].date()].append(r)
print("\npor dia de emision (MAE ours / ina / bw, n):")
for dday in sorted(dia):
    f = fila(dia[dday])
    print(dday, " ".join("%.3f" % v for v in f), "n=%d" % len(dia[dday]))
    res["por_dia_de_emision"][str(dday)] = dict(n=len(dia[dday]), **{m: round(v, 4) for m, v in zip(METODOS, f)})
mejor = sum(1 for dd in dia.values() if fila(dd)[2] < fila(dd)[1])
print("bw mejor que el INA en %d de %d dias de emision" % (mejor, len(dia)))
res["bw_mejor_que_ina_dias"] = [mejor, len(dia)]
if OUT:
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("guardado en", OUT)
