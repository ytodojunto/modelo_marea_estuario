import os, sys, runpy
R = sys.argv[1]
# datos y modelo base: ver sf_datos.py
_G = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sf_datos.py"))
globals().update({k: v for k, v in _G.items() if not k.startswith("__")})

CUT = dt.datetime(2026, 10, 11)
test = [s for s in S if s["emis"] >= CUT and s["T"] >= CUT and s["T"] <= last_obs]
print("test 11/10 -> %s: %d horas-objetivo, %d ciclos" % (last_obs, len(test), len({s["c"] for s in test})))

variants = {
    "A (entrenado hasta 7/10, sin ver el pico del 9/10)": train(dt.datetime(2026, 10, 7, 12), dt.datetime(2026, 10, 7, 12)),
    "B (reentrenado hasta 10/10, incluye el 9/10)": train(CUT, CUT),
}
for k, (m, mu, sd, n) in [(k, v[0]) for k, v in variants.items()][:0]:
    pass
for name, mod in variants.items():
    print("== modelo", name, " muestras por banda:", [mod[i][3] for i in range(5)])
    for s in test:
        s["p_" + name[0]] = predict(mod, s)

rows = []
for s in test:
    ina = ina_for(s["emis"], s["T"])
    if ina is None:
        continue
    rows.append({**s, "ina": ina})
print("pares con INA:", len(rows))

def show(label, key):
    print(label)
    for i, (a, b) in enumerate(BANDS):
        rr = [r for r in rows if a <= r["h"] <= b]
        print("   %2d-%2d h" % (a, b), stats([r["obs"] - key(r) for r in rr]))
    print("   total ", stats([r["obs"] - key(r) for r in rows]))

show("Solo astronomia", lambda r: r["ast"])
show("Astronomia + ultimo residuo", lambda r: r["ast"] + r["x"][0])
show("Nosotros A", lambda r: r["p_A"])
show("Nosotros B", lambda r: r["p_B"])
show("INA", lambda r: r["ina"])
show("Mezcla 50/50 B + INA", lambda r: 0.5 * (r["p_B"] + r["ina"]))

# por dia
print("--- por dia (MAE) nosotros B / INA / n")
byd = collections.defaultdict(list)
for r in rows:
    byd[r["T"].date()].append(r)
for d in sorted(byd):
    rr = byd[d]
    print(d, "B %.3f  INA %.3f  n=%d" % (st.mean(abs(r["obs"] - r["p_B"]) for r in rr), st.mean(abs(r["obs"] - r["ina"]) for r in rr), len(rr)))

# extremos vs SHN / INA
ext = json.load(open(f"{R}/../sf_extremos.json"))
out = []
for e in ext:
    t = dt.datetime.fromisoformat(e["t"])
    if t < CUT:
        continue
    cap_art = t - dt.timedelta(hours=0)
    # ciclo mas reciente cuya emision <= captura SHN (aprox: t - lead_shn); usamos emision <= t - 6h
    best = None
    for s in S:
        pass
    cs = sorted({(s["emis"]) for s in S if s["emis"] <= t - dt.timedelta(hours=6)})
    if not cs:
        continue
    em = cs[-1]
    cand = [s for s in S if s["emis"] == em and abs((s["T"] - t).total_seconds()) <= 3 * 3600]
    if len(cand) < 5:
        continue
    pb = [predict(variants["B (reentrenado hasta 10/10, incluye el 9/10)"], s) for s in cand]
    pa = [predict(variants["A (entrenado hasta 7/10, sin ver el pico del 9/10)"], s) for s in cand]
    f = max if e["e"] == "PLEAMAR" else min
    out.append(dict(t=t, e=e["e"], obs=e["obs"], shn=e["shn"], ina=e["ina"], A=f(pa), B=f(pb)))
print("--- extremos San Fernando desde 11/10 (n=%d)  error = obs - prono" % len(out))
for k in ["shn", "ina", "A", "B"]:
    print({"shn": "SHN", "ina": "INA", "A": "Nosotros A", "B": "Nosotros B"}[k], stats([o["obs"] - o[k] for o in out]))
for o in out:
    print(o["t"], o["e"][:3], "obs %.2f | SHN %+.2f INA %+.2f A %+.2f B %+.2f" % (o["obs"], o["obs"] - o["shn"], o["obs"] - o["ina"], o["obs"] - o["A"], o["obs"] - o["B"]))
