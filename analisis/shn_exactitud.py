import json, sys, datetime as dt, collections, statistics as st
R = sys.argv[1]
ART = dt.timedelta(hours=3)
MAP = {"PUERTO LA PLATA": "La Plata", "PUERTO DE BUENOS AIRES": "Buenos Aires",
       "SAN FERNANDO": "San Fernando", "CANAL PUNTA INDIO": "Oyarvide"}

def station(l):
    for k, v in MAP.items():
        if l.startswith(k):
            return v

# observado (hora ART naive)
obs = collections.defaultdict(dict)
for line in open(f"{R}/SHN_captura_pronosticos/data/historico_alturas.jsonl"):
    d = json.loads(line)
    for m in d.get("mediciones", []):
        try:
            t = dt.datetime.strptime(m["fecha_hora"], "%d/%m/%Y %H:%M")
            obs[m["estacion"]][t] = float(m["altura_m"])
        except Exception:
            pass
last_obs = max(max(v) for v in obs.values() if v)

# pronosticos
snaps = []
for line in open(f"{R}/SHN_captura_pronosticos/data/historico.jsonl"):
    d = json.loads(line)
    cap = dt.datetime.fromisoformat(d["capturado_en"]).replace(tzinfo=None)  # UTC
    ev = []
    for p in d.get("puertos", []):
        s = station(p["lugar"])
        if not s or "---" in (p.get("hora") or "") or not p.get("altura_m") or "-" in p["altura_m"]:
            continue
        try:
            t = dt.datetime.strptime(p["fecha"] + " " + p["hora"], "%d/%m/%Y %H:%M")  # ART
            ev.append((s, p["estado"], t, float(p["altura_m"])))
        except Exception:
            pass
    snaps.append((cap, ev))
snaps.sort(key=lambda x: x[0])

# por evento: ultimo snapshot publicado antes del evento
best = {}
for cap, ev in snaps:
    for s, e, t, h in ev:
        tu = t + ART
        if cap < tu:
            k = (s, e, t)
            if k not in best or cap > best[k][0]:
                best[k] = (cap, h)

rows = []
for (s, e, t), (cap, h) in best.items():
    o = obs.get(s, {})
    w = [(tt, v) for tt, v in o.items() if abs((tt - t).total_seconds()) <= 3 * 3600]
    if len(w) < 6 or t > last_obs - dt.timedelta(hours=3):
        continue
    if e == "PLEAMAR":
        tt, v = max(w, key=lambda x: x[1])
    else:
        tt, v = min(w, key=lambda x: x[1])
    lead = ((t + ART) - cap).total_seconds() / 3600
    rows.append(dict(s=s, e=e, t=t, prono=h, obs=v, err=v - h, dt=abs((tt - t).total_seconds()) / 3600, lead=lead))
rows.sort(key=lambda r: r["t"])

def stats(rs):
    if not rs:
        return "n=0"
    er = [r["err"] for r in rs]
    return "n=%d MAE %.3f RMSE %.3f sesgo %+.3f" % (len(rs), st.mean(abs(x) for x in er), (st.mean(x * x for x in er)) ** .5, st.mean(er))

print("ultimo observado:", last_obs, " eventos:", len(rows), rows[0]["t"], "->", rows[-1]["t"])
print("TOTAL", stats(rows), " lead medio %.1f h" % st.mean(r["lead"] for r in rows))
print("--- por estacion/evento")
for s in ["Buenos Aires", "La Plata", "Oyarvide", "San Fernando"]:
    for e in ["BAJAMAR", "PLEAMAR"]:
        print(s, e, stats([r for r in rows if r["s"] == s and r["e"] == e]))
print("--- por lead")
for a, b in [(0, 4), (4, 8), (8, 13)]:
    print("%d-%d h" % (a, b), stats([r for r in rows if a <= r["lead"] < b]))
print("--- por semana (lunes)")
wk = collections.defaultdict(list)
for r in rows:
    d0 = r["t"].date()
    wk[d0 - dt.timedelta(days=d0.weekday())].append(r)
for k in sorted(wk):
    print(k, stats(wk[k]))
cut = dt.datetime(2026, 10, 11)
print("--- desde 11/10", stats([r for r in rows if r["t"] >= cut]))
print("--- hasta 10/10 (comparable con 11/10)", stats([r for r in rows if r["t"] < cut]))
print("--- 11-18/10 por estacion")
for s in ["Buenos Aires", "La Plata", "Oyarvide", "San Fernando"]:
    print(s, stats([r for r in rows if r["s"] == s and r["t"] >= cut]))
print("--- peores 12 eventos desde 11/10")
for r in sorted([r for r in rows if r["t"] >= cut], key=lambda r: -abs(r["err"]))[:12]:
    print(r["t"], r["s"], r["e"], "prono %.2f obs %.2f err %+.2f lead %.1f" % (r["prono"], r["obs"], r["err"], r["lead"]))
print("--- mayores observados (pleamar) de toda la ventana")
for r in sorted([r for r in rows if r["e"] == "PLEAMAR"], key=lambda r: -r["obs"])[:8]:
    print(r["t"], r["s"], "prono %.2f obs %.2f err %+.2f" % (r["prono"], r["obs"], r["err"]))
json.dump([{**r, "t": r["t"].isoformat()} for r in rows], open(f"{R}/../shn_eventos.json", "w"))
