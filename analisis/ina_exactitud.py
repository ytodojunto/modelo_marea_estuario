import json, sys, glob, csv, datetime as dt, collections, statistics as st, bisect
R = sys.argv[1]
P = lambda s: dt.datetime.fromisoformat(s.replace("Z", "")[:19])

# ---- observado
def load_series(slug):
    o = {}
    for r in csv.DictReader(open(f"{R}/INA_captura_delta_2/data/series/{slug}.csv")):
        try:
            o[P(r["timestart"])] = float(r["valor"])
        except Exception:
            pass
    return o

sf = load_series("san_fernando")
# completar con SHN alturas (mas reciente)
for line in open(f"{R}/SHN_captura_pronosticos/data/historico_alturas.jsonl"):
    d = json.loads(line)
    for m in d.get("mediciones", []):
        if m["estacion"] == "San Fernando":
            try:
                sf[dt.datetime.strptime(m["fecha_hora"], "%d/%m/%Y %H:%M")] = float(m["altura_m"])
            except Exception:
                pass
last_sf = max(sf)

def sf_at(t):  # interpola observado HH:45 a HH:00
    a = t - dt.timedelta(minutes=15)
    b = t + dt.timedelta(minutes=45)
    if a in sf and b in sf:
        return 0.75 * sf[a] + 0.25 * sf[b]
    return None

# ---- corridas
corr = collections.defaultdict(dict)  # slug -> corid -> dict
for f in sorted(glob.glob(f"{R}/INA_captura_delta_2/data/historico_prono/*.json")):
    d = json.load(open(f))
    for slug, e in d["estaciones"].items():
        c = e.get("corid")
        if c is None or c in corr[slug]:
            continue
        corr[slug][c] = dict(t0=P(e["corrida_forecastdate"]), serie=[(P(x["timestart"]), x["central"]) for x in e["serie"] if x.get("central") is not None], cap=P(d.get("capturado_en") or d.get("generado_en")))

def stats(er):
    if not er:
        return "n=0"
    return "n=%d MAE %.3f RMSE %.3f sesgo %+.3f" % (len(er), st.mean(abs(x) for x in er), st.mean(x * x for x in er) ** .5, st.mean(er))

# ---- San Fernando horario
print("San Fernando: corridas unicas", len(corr["san_fernando"]), " ultimo observado", last_sf)
pairs = []  # (target, lead_h, err, t0)
for c, x in corr["san_fernando"].items():
    for t, v in x["serie"]:
        o = sf_at(t)
        if o is not None:
            pairs.append((t, (t - x["t0"]).total_seconds() / 3600, o - v, x["t0"]))
print("pares", len(pairs), "TOTAL", stats([p[2] for p in pairs]))
for a, b in [(0, 24), (24, 48), (48, 72), (72, 200)]:
    print("  lead %d-%d h" % (a, b), stats([p[2] for p in pairs if a <= p[1] < b]))
cut = dt.datetime(2026, 10, 11)
print("hasta 10/10", stats([p[2] for p in pairs if p[0] < cut]))
print("desde 11/10", stats([p[2] for p in pairs if p[0] >= cut]))
for a, b in [(0, 24), (24, 48), (48, 72), (72, 200)]:
    print("  desde 11/10 lead %d-%d h" % (a, b), stats([p[2] for p in pairs if p[0] >= cut and a <= p[1] < b]))
# persistencia
ts = sorted(sf)
def pers(t0):
    i = bisect.bisect_right(ts, t0) - 1
    return sf[ts[i]] if i >= 0 else None
pp = []
for t, lead, e, t0 in pairs:
    pv = pers(t0)
    o = sf_at(t)
    if pv is not None:
        pp.append(o - pv)
print("persistencia (ultimo observado a t0)", stats(pp))

# diario de error INA (ultimos dias)
byday = collections.defaultdict(list)
for t, lead, e, t0 in pairs:
    byday[t.date()].append(e)
print("--- por dia (todas las corridas)")
for d in sorted(byday)[-12:]:
    print(d, stats(byday[d]))

# ---- extremos San Fernando: SHN vs INA (misma muestra)
ev = json.load(open(f"{R}/../shn_eventos.json"))
snapcap = {}
for line in open(f"{R}/SHN_captura_pronosticos/data/historico.jsonl"):
    d = json.loads(line)
    snapcap[d["capturado_en"]] = dt.datetime.fromisoformat(d["capturado_en"]).replace(tzinfo=None)
runs = sorted(corr["san_fernando"].values(), key=lambda x: x["t0"])
rows = []
for r in ev:
    if r["s"] != "San Fernando":
        continue
    t = dt.datetime.fromisoformat(r["t"])
    cap_art = t - dt.timedelta(hours=r["lead"]) + dt.timedelta(hours=0)  # t es ART; cap = t - lead (ART)
    cands = [x for x in runs if x["t0"] <= cap_art]
    if not cands:
        continue
    run = cands[-1]
    w = [v for tt, v in run["serie"] if abs((tt - t).total_seconds()) <= 3 * 3600]
    if len(w) < 5:
        continue
    pv = max(w) if r["e"] == "PLEAMAR" else min(w)
    rows.append(dict(t=t, e=r["e"], obs=r["obs"], shn=r["prono"], ina=pv, lead_ina=(t - run["t0"]).total_seconds() / 3600))
print("--- extremos San Fernando n=%d (%s -> %s)" % (len(rows), rows[0]["t"], rows[-1]["t"]))
print("SHN", stats([r["obs"] - r["shn"] for r in rows]))
print("INA", stats([r["obs"] - r["ina"] for r in rows]))
print("SHN gana en", sum(abs(r["obs"] - r["shn"]) < abs(r["obs"] - r["ina"]) for r in rows), "de", len(rows), " lead INA medio %.1f h" % st.mean(r["lead_ina"] for r in rows))
rr = [r for r in rows if r["t"] >= cut]
print("desde 11/10 SHN", stats([r["obs"] - r["shn"] for r in rr]))
print("desde 11/10 INA", stats([r["obs"] - r["ina"] for r in rr]))
print("--- extremos desde 14/10 (error = obs - prono)")
for r in rows:
    if r["t"] >= dt.datetime(2026, 10, 14):
        print(r["t"], r["e"][:3], "obs %.2f  SHN %+.2f  INA %+.2f" % (r["obs"], r["obs"] - r["shn"], r["obs"] - r["ina"]))
json.dump([{**r, "t": r["t"].isoformat()} for r in rows], open(f"{R}/../sf_extremos.json", "w"))

# ---- corredor
print("--- corredor (misma hora exacta, lead desde corrida)")
for slug in ["corrientes", "barranqueras", "goya", "parana_ciudad", "santa_fe", "diamante", "san_lorenzo_san_martin", "rosario", "villa_constitucion", "san_nicolas", "ramallo", "campana"]:
    o = load_series(slug)
    prs = []
    for c, x in corr[slug].items():
        for t, v in x["serie"]:
            if t in o:
                prs.append((c, o[t] - v))
    if not prs:
        print(slug, "sin pares")
        continue
    bc = collections.defaultdict(list)
    for c, e in prs:
        bc[c].append(e)
    # sesgo leave-one-run-out
    loo = []
    for c, es in bc.items():
        others = [e for k, v in bc.items() if k != c for e in v]
        if others:
            m = st.mean(others)
            loo += [e - m for e in es]
    er = [e for _, e in prs]
    print("%-24s %s | sin sesgo %s" % (slug, stats(er), ("MAE %.3f" % st.mean(abs(x) for x in loo)) if loo else "-"))
