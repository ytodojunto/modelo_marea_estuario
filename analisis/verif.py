import sys, os, runpy, datetime as dt, statistics as st, collections
sys.argv = ["x", sys.argv[1]]
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_sf.py"))
S, train, predict, ina_for, stats, rows, last_obs = g["S"], g["train"], g["predict"], g["ina_for"], g["stats"], g["rows"], g["last_obs"]
variants = g["variants"]
A = variants["A (entrenado hasta 7/10, sin ver el pico del 9/10)"]
# 1) reproducir ventana del 11/10: ciclos emitidos desde 7/10 12h, objetivos 8/10 12h -> 10/10 21h
w0, w1 = dt.datetime(2026,10,8,12), dt.datetime(2026,10,10,21)
tt = [s for s in S if s["emis"] >= dt.datetime(2026,10,7,12) and w0 <= s["T"] <= w1]
pr = [(s["obs"] - predict(A, s)) for s in tt]
ina = [(s["obs"] - ina_for(s["emis"], s["T"])) for s in tt if ina_for(s["emis"], s["T"]) is not None]
print("REPRODUCCION 8/10-10/10: nosotros", stats(pr), "| INA", stats(ina), "(doc 11/10: nosotros MAE 0.191, INA 0.168)")
# 2) diagnostico 13-14/10
print("DIAGNOSTICO ciclos emitidos 12-14/10, objetivos 13-14/10 (residuo obs vs predicho por A)")
byc = collections.defaultdict(list)
for s in rows:
    if dt.datetime(2026,10,13) <= s["T"] < dt.datetime(2026,10,15):
        byc[s["emis"]].append(s)
for em in sorted(byc)[:8]:
    rr = byc[em]
    m, mu, sd, _ = A[0]
    pred_res = [predict(A, s) - s["ast"] for s in rr]
    print(em, "r0 %+.2f rm %+.2f | residuo obs medio %+.2f  predicho medio %+.2f | u_se medio %+.1f u_sw medio %+.1f" % (rr[0]["x"][0], rr[0]["x"][1], st.mean(s["y"] for s in rr), st.mean(pred_res), st.mean(s["x"][2] for s in rr), st.mean(s["x"][3] for s in rr)))
# 3) mezcla: sensibilidad y cuanto es cancelacion de sesgo
import numpy as np
o = np.array([r["obs"] for r in rows]); b = np.array([r["p_B"] for r in rows]); a_ = np.array([r["p_A"] for r in rows]); i = np.array([r["ina"] for r in rows])
mae = lambda x: np.mean(np.abs(o - x))
print("MEZCLA B/INA pesos (peso de INA): ", {w: round(mae(w*i+(1-w)*b),3) for w in [0,0.3,0.5,0.7,1]})
print("MEZCLA A/INA pesos (peso de INA): ", {w: round(mae(w*i+(1-w)*a_),3) for w in [0,0.3,0.5,0.7,1]})
db, di = np.mean(o-b), np.mean(o-i)
print("sesgos de la semana: B %+.3f INA %+.3f" % (db, di))
print("MAE con sesgo medio de la semana removido a cada uno (oraculo, solo diagnostico): B %.3f INA %.3f mezcla %.3f" % (mae(b+db), mae(i+di), mae(0.5*(b+db)+0.5*(i+di))))
# mezcla sin bias oraculo: error de la mezcla 50/50 vs mejor individual por dia
byd = collections.defaultdict(list)
for r in rows: byd[r["T"].date()].append(r)
print("dias donde la mezcla mejora al INA:", sum(st.mean(abs(r["obs"]-0.5*(r["p_B"]+r["ina"])) for r in v) < st.mean(abs(r["obs"]-r["ina"]) for r in v) for v in byd.values()), "de", len(byd))
