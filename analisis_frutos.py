from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Datos 
ruta_csv = Path("datos_cosecha.csv")
df = pd.read_csv(ruta_csv)

# Eliminar columnas vacías y renombrar
df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
df = df.rename(columns={
    "Código Fruto": "codigo_fruto",
    "Tipo de fruto": "tipo_fruto",
    "Peso (g)": "peso",
    "Color de piel": "color",
    "Firmeza (cara B)": "firmeza_b",
    "Firmeza (cara A)": "firmeza_a",
    "Css (ºBrix)": "brix",
    "Acidez (% eq acido citrico)": "acidez",
    "RATIO BRIX/AC": "ratio_excel"
})

#Limpiar texto y extraer bloque
df["tipo_fruto"] = df["tipo_fruto"].fillna("").astype(str).str.strip().str.upper()
df["tipo_fruto"] = df["tipo_fruto"].replace({"CIRUELO": "CIRUELA", "PLUM COT": "PLUMCOT", "": np.nan})

df["color"] = df["color"].fillna("").astype(str).str.strip().str.upper().replace({"": np.nan})

def extraer_bloque(val):
    if pd.isna(val) or str(val).strip() == "":
        return "SIN_CODIGO"
    match = re.search(r"^([A-Z]\d+[A-Z]?)", str(val).strip())
    return match.group(1) if match else "OTRO"

df["bloque"] = df["codigo_fruto"].apply(extraer_bloque)

# Columnas a números y calcular variables con NumPy
df["ratio_excel"] = df["ratio_excel"].astype(str).str.replace("..", ".", regex=False)
for col in ["peso", "firmeza_b", "firmeza_a", "brix", "acidez", "ratio_excel"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["firmeza"] = np.nanmean(df[["firmeza_a", "firmeza_b"]], axis=1)
df["ratio_real"] = np.where(df["acidez"] > 0, df["brix"] / df["acidez"], np.nan)
df = df.dropna(subset=["tipo_fruto"])

# Datos limpios y tabla resumen
df.to_csv("datos_cosecha_limpios.csv", index=False)
variables = ["peso", "firmeza", "brix", "acidez", "ratio_real"]
resumen = df.groupby("tipo_fruto")[variables].agg(["count", "mean", "std", "median"]).round(2)
resumen.to_csv("resumen_estadistico.csv")
print("\n--- RESUMEN ESTADÍSTICO POR ESPECIE ---")
print(resumen)

# Gráficos
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x="tipo_fruto", y="peso", hue="tipo_fruto", ax=axes[0], palette="Set2", legend=False)
axes[0].set_title("Peso (g) por Especie")
sns.boxplot(data=df, x="tipo_fruto", y="brix", hue="tipo_fruto", ax=axes[1], palette="Set2", legend=False)
axes[1].set_title("°Brix por Especie")
plt.tight_layout()
plt.savefig("grafico_especies.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
sns.heatmap(df[variables].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlación entre Variables")
plt.tight_layout()
plt.savefig("grafico_correlacion.png", dpi=300)
plt.close()

print("\n¡Listo! Archivos y gráficos generados correctamente.")
