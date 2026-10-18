"""Sensibilidad a la hora de emision: el modelo se valido con emision = init + 6 h UTC.
En tiempo real se emite cada hora con el ciclo SMN mas reciente, o sea con 3-12 h desde el init.
Entrena con la convencion validada (u = 6 h), hasta el 10/10, y prueba el 11-18/10 armando las
muestras con otras horas de emision (u = horas desde el init, reloj UTC).

Uso: python3 emision_rodante.py <carpeta repos>
"""
import sys, os, datetime as dt, statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capa_lib as L

R = sys.argv[1]
CUT = dt.datetime(2026, 10, 11)
rows6 = L.enriquecer(L.cargar(R, 3))
train = [r for r in rows6 if r["emis"] < CUT and r["T"] < CUT]
W, models = L.ajustar(train)
print("pesos bw entrenados hasta el 10/10:", {"%d-%d h" % L.BANDS[i]: round(float(w), 2) for i, w in W.items()})
print("%4s %5s | %-24s | %-24s | %-24s" % ("u", "n", "total ours/ina/bw", "1-6 h ours/ina/bw", "13-24 h ours/ina/bw"))
for u in range(3, 13):
    rows = L.enriquecer(L.cargar(R, u - 3))
    test = [r for r in rows if r["has"] and r["emis"] >= CUT and r["T"] >= CUT]
    out = L.predecir(test, W, models)
    cell = lambda lo, hi: " / ".join("%.3f" % st.mean(abs(r["obs"] - o[m]) for r, o in zip(test, out) if lo <= r["h"] <= hi) for m in ["ours", "ina", "bw"])
    print("%4d %5d | %-24s | %-24s | %-24s" % (u, len(test), cell(1, 66), cell(1, 6), cell(13, 24)))
