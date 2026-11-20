import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ==============================================================================
# 1. GENERACIÓN DE LA BASE DE DATOS SIMULADA (N = 180)
# ==============================================================================
np.random.seed(42)  # Semilla fija para garantizar reproducibilidad total
N = 180
ALPHA = 0.05

# --- Variables Categóricas Nominales ---
facultades = np.random.choice(
    ["Ingeniería", "Derecho", "Medicina", "Comercio"],
    size=N,
    p=[0.35, 0.25, 0.20, 0.20],
)

condicion_laboral = np.random.choice(
    ["No trabaja", "Trabaja"],
    size=N,
    p=[0.60, 0.40],  # Se verifica con un assert más abajo que n >= 30 en ambos grupos
)

# La modalidad depende ligeramente de si el alumno trabaja o no
modalidad = []
for cond in condicion_laboral:
    if cond == "No trabaja":
        modalidad.append(
            np.random.choice(
                ["Presencial", "Híbrida", "Virtual"], p=[0.60, 0.25, 0.15]
            )
        )
    else:
        modalidad.append(
            np.random.choice(
                ["Presencial", "Híbrida", "Virtual"], p=[0.25, 0.45, 0.30]
            )
        )
modalidad = np.array(modalidad)

# --- Variables Cuantitativas Continuas y Discretas ---
# 1) Examen de admisión: Normal con parámetros poblacionales conocidos (mu=75, sigma=10)
examen_admision = np.clip(
    np.round(np.random.normal(loc=75.0, scale=10.0, size=N), 2), 0, 100
)

# 2) Promedio general (0-100): Normal con homocedasticidad entre grupos (sigma desconocida)
efecto_trabajo = np.where(condicion_laboral == "No trabaja", 82.5, 78.5)
efecto_modalidad = np.where(
    modalidad == "Presencial", 1.5, np.where(modalidad == "Híbrida", 0.0, -2.0)
)
promedio_general = np.clip(
    np.round(
        efecto_trabajo + efecto_modalidad + np.random.normal(0, 5.5, size=N), 2
    ),
    0,
    100,
)

# 3) Horas de estudio semanal: Distribución sesgada (Gamma) -> No normal
horas_estudio = np.where(
    condicion_laboral == "No trabaja",
    2.0 + np.random.gamma(shape=2.2, scale=4.2, size=N),
    1.0 + np.random.gamma(shape=1.8, scale=3.0, size=N),
)
horas_estudio = np.round(horas_estudio, 1)

# 4) Ausencias acumuladas: Discreta de conteo (Poisson)
ausencias = np.where(
    condicion_laboral == "No trabaja",
    np.random.poisson(lam=2.5, size=N),
    np.random.poisson(lam=4.8, size=N),
)

# 5) Notas Pre y Post Taller Intensivo (muestras dependientes / pareadas)
nota_pre_taller = np.clip(
    np.round(np.random.normal(loc=68.0, scale=7.5, size=N), 2), 0, 100
)
mejora_taller = np.random.normal(
    loc=7.2, scale=4.2, size=N
)  # Diferencias normales
nota_post_taller = np.clip(np.round(nota_pre_taller + mejora_taller, 2), 0, 100)

# --- Variables Categóricas Ordinales (Escala Likert 1 a 5) ---
# 6) Nivel de ansiedad (1 = Muy baja, 5 = Muy alta)
nivel_ansiedad = np.where(
    condicion_laboral == "No trabaja",
    np.random.choice([1, 2, 3, 4, 5], size=N, p=[0.15, 0.35, 0.30, 0.15, 0.05]),
    np.random.choice([1, 2, 3, 4, 5], size=N, p=[0.05, 0.15, 0.30, 0.32, 0.18]),
)

# 7) Autoevaluación del desempeño por parcial (3 mediciones repetidas en el mismo sujeto)
autoeval_p1 = np.random.choice(
    [1, 2, 3, 4, 5], size=N, p=[0.12, 0.33, 0.35, 0.15, 0.05]
)
autoeval_p2 = np.clip(
    autoeval_p1 + np.random.choice([-1, 0, 1], size=N, p=[0.15, 0.45, 0.40]),
    1,
    5,
)
autoeval_p3 = np.clip(
    autoeval_p2 + np.random.choice([-1, 0, 1], size=N, p=[0.10, 0.45, 0.45]),
    1,
    5,
)

# Ensamblar DataFrame limpio
df = pd.DataFrame({
    "ID_Estudiante": [f"EST-{i:03d}" for i in range(1, N + 1)],
    "Facultad": facultades,
    "Condicion_Laboral": condicion_laboral,
    "Modalidad": modalidad,
    "Examen_Admision": examen_admision,
    "Horas_Estudio_Semanal": horas_estudio,
    "Promedio_General": promedio_general,
    "Ausencias": ausencias,
    "Nota_Pre_Taller": nota_pre_taller,
    "Nota_Post_Taller": nota_post_taller,
    "Nivel_Ansiedad": nivel_ansiedad,
    "Autoeval_P1": autoeval_p1,
    "Autoeval_P2": autoeval_p2,
    "Autoeval_P3": autoeval_p3,
})

# ==============================================================================
# VALIDACIÓN Y EXPORTACIÓN DE LA BASE DE DATOS LIMPIA
# ==============================================================================
# Verificación de tamaños de grupo requeridos para las pruebas z
assert (df["Condicion_Laboral"].value_counts() >= 30).all(), "Grupo con n < 30"

# Controles de calidad: sin nulos, sin duplicados, valores dentro de rango
assert df.isna().sum().sum() == 0, "Hay valores faltantes"
assert df["ID_Estudiante"].is_unique, "IDs duplicados"
for col in ["Examen_Admision", "Promedio_General", "Nota_Pre_Taller", "Nota_Post_Taller"]:
    assert df[col].between(0, 100).all(), f"{col} fuera del rango 0-100"
for col in ["Nivel_Ansiedad", "Autoeval_P1", "Autoeval_P2", "Autoeval_P3"]:
    assert df[col].between(1, 5).all(), f"{col} fuera de la escala Likert 1-5"
assert (df["Ausencias"] >= 0).all() and (df["Horas_Estudio_Semanal"] > 0).all()

# Exportar la base de datos limpia (la misma que usan todas las pruebas de este script)
archivo_csv = "base_datos_equipo3.csv"
df.to_csv(archivo_csv, index=False, encoding="utf-8-sig")
print(
    f"Base de datos limpia '{archivo_csv}' generada y validada "
    f"({len(df)} registros, {df.shape[1]} variables, 0 nulos).\n"
)


# Función auxiliar para imprimir resultados de manera uniforme
def reportar_prueba(num, nombre, estadistico, p_valor, detalle=""):
    decision = (
        "Rechazar H0 (Significativo)"
        if p_valor < ALPHA
        else "No rechazar H0 (No significativo)"
    )
    print(f"{str(num):>3}. {nombre}")
    if detalle:
        print(f"    Contexto   : {detalle}")
    print(
        f"    Estadístico: {estadistico:.4f} | p-value: {p_valor:.4e} | Decisión (α=0.05): {decision}\n"
    )


# ==============================================================================
# BLOQUE 1: BONDAD DE AJUSTE
# ==============================================================================
print("=" * 80)
print("BLOQUE 1: PRUEBAS DE BONDAD DE AJUSTE")
print("=" * 80)

# 1. Prueba Chi-cuadrada para bondad de ajuste (Facultad vs proporciones teóricas)
orden_fac = ["Ingeniería", "Derecho", "Medicina", "Comercio"]
obs_fac = df["Facultad"].value_counts().reindex(orden_fac).values
esp_fac = np.array([0.35, 0.25, 0.20, 0.20]) * N
chi2_stat, p_chi2_ba = stats.chisquare(f_obs=obs_fac, f_exp=esp_fac)
reportar_prueba(
    1,
    "Chi-cuadrada de bondad de ajuste",
    chi2_stat,
    p_chi2_ba,
    "Distribución de Facultad vs proporciones teóricas [35%, 25%, 20%, 20%]",
)

# 2. Prueba de Kolmogorov-Smirnov (Examen_Admision vs Normal(75, 10))
ks_stat, p_ks = stats.kstest(df["Examen_Admision"], stats.norm(75.0, 10.0).cdf)
reportar_prueba(
    2,
    "Kolmogorov-Smirnov (1 muestra)",
    ks_stat,
    p_ks,
    "Ajuste de Examen_Admision a distribución teórica N(μ=75, σ=10)",
)

# 3. Prueba de Shapiro-Wilk (Submuestra n=50 de Promedio_General)
# Nota: el resultado puede estar cerca de α; la decisión depende de la submuestra elegida.
submuestra_prom = df["Promedio_General"].sample(n=50, random_state=42)
sw_stat, p_sw = stats.shapiro(submuestra_prom)
reportar_prueba(
    3,
    "Shapiro-Wilk (normalidad)",
    sw_stat,
    p_sw,
    "Normalidad en submuestra (n=50) de Promedio_General",
)

# 4. Prueba de Lilliefors (Normalidad con media y varianza desconocidas)
lillie_stat, p_lillie = lilliefors(df["Promedio_General"], dist="norm")
reportar_prueba(
    4,
    "Prueba de Lilliefors",
    lillie_stat,
    p_lillie,
    "Normalidad de Promedio_General con μ y σ desconocidas",
)


# ==============================================================================
# BLOQUE 2: PRUEBAS DE UNA MUESTRA
# ==============================================================================
print("=" * 80)
print("BLOQUE 2: PRUEBAS DE UNA MUESTRA")
print("=" * 80)

# 5. Prueba z para una sola muestra (Examen_Admision, mu_0 = 73, sigma = 10 conocida)
mu_0_z = 73.0
sigma_conocida = 10.0
media_adm = df["Examen_Admision"].mean()
z_1m = (media_adm - mu_0_z) / (sigma_conocida / np.sqrt(N))
p_z_1m = 2 * stats.norm.sf(abs(z_1m))
reportar_prueba(
    5,
    "Prueba z para una sola muestra",
    z_1m,
    p_z_1m,
    f"Examen_Admision (x̄={media_adm:.2f}) vs μ₀={mu_0_z} con σ={sigma_conocida} conocida",
)

# 6. Prueba t para una sola muestra (Promedio_General, mu_0 = 80, sigma desconocida)
mu_0_t = 80.0
t_1m, p_t_1m = stats.ttest_1samp(df["Promedio_General"], popmean=mu_0_t)
reportar_prueba(
    6,
    "Prueba t para una sola muestra",
    t_1m,
    p_t_1m,
    f"Promedio_General (x̄={df['Promedio_General'].mean():.2f}) vs μ₀={mu_0_t} (σ desconocida)",
)

# 7. Prueba de Wilcoxon Signed-Rank para 1 muestra (Mediana de Horas_Estudio vs M_0 = 8)
# Limitación a mencionar en el informe: la prueba supone simetría alrededor de la mediana
# y Horas_Estudio es sesgada (Gamma).
mediana_0 = 8.0
w_1m, p_w_1m = stats.wilcoxon(df["Horas_Estudio_Semanal"] - mediana_0)
reportar_prueba(
    7,
    "Wilcoxon Signed-Rank (1 muestra)",
    w_1m,
    p_w_1m,
    f"Mediana de Horas_Estudio_Semanal (Me={df['Horas_Estudio_Semanal'].median():.2f}) vs M₀={mediana_0}",
)


# ==============================================================================
# BLOQUE 3: PRUEBAS DE DOS MUESTRAS E INDEPENDENCIA
# ==============================================================================
print("=" * 80)
print("BLOQUE 3: PRUEBAS DE DOS MUESTRAS E INDEPENDENCIA")
print("=" * 80)

# 8. Prueba Chi-cuadrada de independencia (Condicion_Laboral vs Modalidad)
tabla_cont = pd.crosstab(df["Condicion_Laboral"], df["Modalidad"])
chi2_ind, p_chi2_ind, dof_ind, _ = stats.chi2_contingency(tabla_cont)
reportar_prueba(
    8,
    "Chi-cuadrada de independencia",
    chi2_ind,
    p_chi2_ind,
    f"Asociación entre Condicion_Laboral y Modalidad (gl={dof_ind})",
)

# Separar grupos por Condicion_Laboral
no_trabaja = df[df["Condicion_Laboral"] == "No trabaja"]
trabaja = df[df["Condicion_Laboral"] == "Trabaja"]

# 9. Prueba t para dos muestras independientes (con verificación previa de homocedasticidad)
lev_stat, p_levene = stats.levene(
    no_trabaja["Promedio_General"], trabaja["Promedio_General"]
)
varianzas_iguales = p_levene >= ALPHA
t_ind, p_t_ind = stats.ttest_ind(
    no_trabaja["Promedio_General"],
    trabaja["Promedio_General"],
    equal_var=varianzas_iguales,
)
sp = np.sqrt(((len(no_trabaja) - 1) * no_trabaja["Promedio_General"].var(ddof=1)
              + (len(trabaja) - 1) * trabaja["Promedio_General"].var(ddof=1))
             / (len(no_trabaja) + len(trabaja) - 2))
d_cohen_ind = (no_trabaja["Promedio_General"].mean() - trabaja["Promedio_General"].mean()) / sp
reportar_prueba(
    9,
    "Prueba t para dos muestras independientes",
    t_ind,
    p_t_ind,
    f"Promedio_General: No trabaja vs Trabaja (Levene p={p_levene:.4f} -> equal_var={varianzas_iguales}); d de Cohen={d_cohen_ind:.2f}",
)

# 10. Prueba z para dos muestras independientes (muestras grandes: n1, n2 >= 30)
x1, x2 = (
    no_trabaja["Promedio_General"].mean(),
    trabaja["Promedio_General"].mean(),
)
s1, s2 = (
    no_trabaja["Promedio_General"].std(ddof=1),
    trabaja["Promedio_General"].std(ddof=1),
)
n1, n2 = len(no_trabaja), len(trabaja)
z_ind = (x1 - x2) / np.sqrt((s1**2 / n1) + (s2**2 / n2))
p_z_ind = 2 * stats.norm.sf(abs(z_ind))
reportar_prueba(
    10,
    "Prueba z para dos muestras independientes",
    z_ind,
    p_z_ind,
    f"Promedio_General en muestras grandes (n₁={n1}, n₂={n2})",
)

# 11. Prueba Wilcoxon Rank-Sum / Mann-Whitney U (Horas_Estudio no normal entre grupos)
u_stat, p_mwu = stats.mannwhitneyu(
    no_trabaja["Horas_Estudio_Semanal"],
    trabaja["Horas_Estudio_Semanal"],
    alternative="two-sided",
)
reportar_prueba(
    11,
    "Wilcoxon Rank-Sum (Mann-Whitney U)",
    u_stat,
    p_mwu,
    "Horas_Estudio_Semanal (no normal): No trabaja vs Trabaja",
)

# 11b. Complemento: Mann-Whitney con variable ORDINAL (Nivel_Ansiedad, Likert 1-5)
u_ans, p_ans = stats.mannwhitneyu(
    no_trabaja["Nivel_Ansiedad"],
    trabaja["Nivel_Ansiedad"],
    alternative="two-sided",
)
reportar_prueba(
    "11b",
    "Wilcoxon Rank-Sum (Mann-Whitney U) - ordinal",
    u_ans,
    p_ans,
    "Nivel_Ansiedad (ordinal 1-5): No trabaja vs Trabaja",
)

# 12. Prueba t-Student para dos muestras dependientes / pareadas (Pre vs Post Taller)
t_par, p_t_par = stats.ttest_rel(df["Nota_Post_Taller"], df["Nota_Pre_Taller"])
reportar_prueba(
    12,
    "Prueba t-Student pareada (2 muestras dependientes)",
    t_par,
    p_t_par,
    f"Nota_Post_Taller vs Nota_Pre_Taller en el mismo estudiante; d de Cohen (pareada)="
    f"{(df['Nota_Post_Taller'] - df['Nota_Pre_Taller']).mean() / (df['Nota_Post_Taller'] - df['Nota_Pre_Taller']).std(ddof=1):.2f}",
)

# 13. Prueba z para dos muestras dependientes (muestra grande emparejada, N=180)
diferencias = df["Nota_Post_Taller"] - df["Nota_Pre_Taller"]
z_par = diferencias.mean() / (diferencias.std(ddof=1) / np.sqrt(N))
p_z_par = 2 * stats.norm.sf(abs(z_par))
reportar_prueba(
    13,
    "Prueba z pareada (2 muestras dependientes)",
    z_par,
    p_z_par,
    f"Diferencia media Post - Pre Taller (d̄={diferencias.mean():.2f}, N={N})",
)

# 14. Prueba Wilcoxon Signed-Ranks para 2 muestras dependientes (Ordinal: Autoeval P1 vs P2)
w_par, p_w_par = stats.wilcoxon(df["Autoeval_P2"], df["Autoeval_P1"])
reportar_prueba(
    14,
    "Wilcoxon Signed-Ranks (2 muestras dependientes)",
    w_par,
    p_w_par,
    "Cambio ordinal en Autoeval_P2 vs Autoeval_P1 (Escala Likert 1–5)",
)


# ==============================================================================
# BLOQUE 4: COMPARACIÓN DE MÚLTIPLES GRUPOS
# ==============================================================================
print("=" * 80)
print("BLOQUE 4: COMPARACIÓN DE MÚLTIPLES GRUPOS")
print("=" * 80)

# 15. Prueba ANOVA de un factor (Promedio_General según Modalidad) + Post-hoc Tukey
g_pres = df[df["Modalidad"] == "Presencial"]["Promedio_General"]
g_hib = df[df["Modalidad"] == "Híbrida"]["Promedio_General"]
g_virt = df[df["Modalidad"] == "Virtual"]["Promedio_General"]

# Verificación de supuestos de ANOVA: homocedasticidad (Levene) y normalidad de residuos (Shapiro)
lev_a, p_lev_a = stats.levene(g_pres, g_hib, g_virt)
residuos = pd.concat([g - g.mean() for g in (g_pres, g_hib, g_virt)])
sw_res, p_sw_res = stats.shapiro(residuos)
print(f"    Supuestos ANOVA -> Levene p={p_lev_a:.4f} | Shapiro (residuos) p={p_sw_res:.4f}\n")

f_stat, p_anova = stats.f_oneway(g_pres, g_hib, g_virt)
reportar_prueba(
    15,
    "ANOVA de un factor",
    f_stat,
    p_anova,
    "Comparación de Promedio_General entre Modalidad (Presencial, Híbrida, Virtual)",
)

if p_anova < ALPHA:
    print("    -> Post-hoc Tukey HSD (ANOVA resultó significativo):")
    tukey = pairwise_tukeyhsd(
        endog=df["Promedio_General"], groups=df["Modalidad"], alpha=ALPHA
    )
    print(tukey)
    print()

# 16. Prueba de Friedman (3 mediciones repetidas ordinales: Autoeval P1, P2 y P3)
fried_stat, p_fried = stats.friedmanchisquare(
    df["Autoeval_P1"], df["Autoeval_P2"], df["Autoeval_P3"]
)
reportar_prueba(
    16,
    "Prueba de Friedman (medidas repetidas)",
    fried_stat,
    p_fried,
    "Evolución de Autoeval_P1, Autoeval_P2 y Autoeval_P3 en el mismo sujeto",
)