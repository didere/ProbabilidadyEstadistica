import math
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Ejercicio_1.csv")
df = df.iloc[:40]
O = df.iloc[:, 2].astype(int).tolist()
k = len(O)

N = sum(O)

p = 1 / k
E = N * p

dif = [o - E for o in O]
cuad = [d ** 2 for d in dif]
contrib = [c / E for c in cuad]
chi2 = sum(contrib)
gl = k - 1

def chi2_critico(gl, z=1.6449):
    a = 2 / (9 * gl)
    return gl * (1 - a + z * math.sqrt(a)) ** 3

def valor_p(x, gl):
    a = 2 / (9 * gl)
    z = ((x / gl) ** (1 / 3) - (1 - a)) / math.sqrt(a)
    return 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))

critico = chi2_critico(gl)
pval = valor_p(chi2, gl)

print(f"N observado        : {N}")
print(f"Frecuencia esperada: {E:.2f}")
print(f"Chi-cuadrado       : {chi2:.4f}")
print(f"Grados de libertad : {gl}")
print(f"Valor critico 5%   : {critico:.4f}")
print(f"Valor p            : {pval:.2e}")
print("Decision:", "RECHAZAR H0" if chi2 > critico else "NO rechazar H0")

ids = range(1, k + 1)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

ax[0].bar(ids, O, label="Observado")
ax[0].axhline(E, color="red", linestyle="--", label=f"Esperado = {E:.1f}")
ax[0].set_xlabel("ID de plataforma")
ax[0].set_ylabel("Accesos")
ax[0].set_title("Accesos observados vs esperados")
ax[0].legend()

ax[1].bar(ids, contrib, color="orange")
ax[1].set_xlabel("ID de plataforma")
ax[1].set_ylabel("(Oi - Ei)² / Ei")
ax[1].set_title(f"Contribución de cada plataforma al χ² = {chi2:.2f}")

plt.tight_layout()
plt.savefig("grafica_chi2.png", dpi=150)
plt.show()