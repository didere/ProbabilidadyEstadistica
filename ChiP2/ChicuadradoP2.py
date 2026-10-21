import math
import pandas as pd
import matplotlib.pyplot as plt

categorias = ["Excelente", "No Excelente"]
O = [10, 30]

N = sum(O)
pi = [0.30, 0.70]
E = [N * p for p in pi]

dif = [o - e for o, e in zip(O, E)]
cuad = [d ** 2 for d in dif]
contrib = [c / e for c, e in zip(cuad, E)]
chi2 = sum(contrib)
gl = len(O) - 1

critico = 3.841458
pval = math.erfc(math.sqrt(chi2 / 2))

# 5. Resultados
print(f"N total            : {N}")
print(f"Observados         : {dict(zip(categorias, O))}")
print(f"Esperados          : {[round(e, 2) for e in E]}")
print(f"Chi-cuadrado       : {chi2:.4f}")
print(f"Grados de libertad : {gl}")
print(f"Valor critico 5%   : {critico:.4f}")
print(f"Valor p            : {pval:.4f}")
print("Decision:", "RECHAZAR H0" if chi2 > critico else "NO rechazar H0")

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

x = range(len(categorias))
ancho = 0.35
ax[0].bar([i - ancho / 2 for i in x], O, ancho, label="Observado")
ax[0].bar([i + ancho / 2 for i in x], E, ancho, label="Esperado")
ax[0].set_xticks(list(x))
ax[0].set_xticklabels(categorias)
ax[0].set_ylabel("Alumnos")
ax[0].set_title("Observados vs esperados")
ax[0].legend()

ax[1].bar(categorias, contrib, color="orange")
ax[1].set_ylabel("(Oi - Ei)² / Ei")
ax[1].set_title(f"Contribución al χ² = {chi2:.4f}")

xs = [i / 100 for i in range(1, 1201)]
dens = [math.exp(-x / 2) / math.sqrt(2 * math.pi * x) for x in xs]
ax[2].plot(xs, dens, color="steelblue", label="χ² (gl = 1)")
xr = [x for x in xs if x >= critico]
ax[2].fill_between(xr, [math.exp(-x / 2) / math.sqrt(2 * math.pi * x) for x in xr],
                   alpha=0.4, color="red", label="Zona de rechazo (5%)")
ax[2].axvline(chi2, color="green", linestyle="--", label=f"χ² calculado = {chi2:.3f}")
ax[2].set_xlim(0, 8)
ax[2].set_ylim(0, 0.6)
ax[2].set_title("Distribución χ² y región crítica")
ax[2].legend()

plt.tight_layout()
plt.savefig("grafica_chi2_P2.png", dpi=150)
plt.show()