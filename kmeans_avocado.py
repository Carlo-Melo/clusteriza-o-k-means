# ============================================================
#  Atividade 6 - Clusterizacao com K-Means
#  Dataset: Avocado | Aluno: Carlos Eduardo
# ============================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

# ------------------------------------------------------------
# 1. CARREGAR OS DADOS
# ------------------------------------------------------------
print("=" * 55)
print("  K-MEANS - DATASET AVOCADO")
print("=" * 55)

df = pd.read_csv("Avocado.csv")

print(f"\n[1] Dataset carregado com sucesso!")
print(f"    Linhas: {df.shape[0]} | Colunas: {df.shape[1]}")
print(f"\n    Primeiras 5 linhas:")
print(df.head().to_string(index=False))

print(f"\n    Classes presentes no dataset:")
print(df["classe"].value_counts().to_string())

# ------------------------------------------------------------
# 2. PRE-PROCESSAMENTO
# ------------------------------------------------------------
features = ["peso", "diâmetro", "espessura_casca", "teor_gordura"]
X = df[features].values
y_true = df["classe"].values

# Normalizacao: deixa todos os atributos na mesma escala
# (media=0, desvio padrao=1) - importante pro K-Means funcionar bem
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n[2] Dados normalizados com StandardScaler")
print(f"    Atributos usados: {features}")

# ------------------------------------------------------------
# 3. METODO DO COTOVELO - escolher o melhor k
# ------------------------------------------------------------
print(f"\n[3] Calculando metodo do cotovelo (k=1 ate k=7)...")

inercias = []
valores_k = range(1, 8)

for k in valores_k:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    inercias.append(km.inertia_)
    print(f"    k={k} -> inercia={km.inertia_:.2f}")

# ------------------------------------------------------------
# 4. RODAR O K-MEANS COM k=3
# ------------------------------------------------------------
print(f"\n[4] Rodando K-Means com k=3...")

kmeans = KMeans(
    n_clusters=3,
    init="k-means++",   # inicializacao esperta (melhor que aleatoria)
    n_init=10,           # roda 10 vezes, pega o melhor resultado
    max_iter=300,        # maximo de iteracoes por rodada
    random_state=42      # garante resultado reproduzivel
)
kmeans.fit(X_scaled)
labels = kmeans.labels_
df["cluster"] = labels

print(f"    Convergiu em {kmeans.n_iter_} iteracoes")
print(f"    Inercia final: {kmeans.inertia_:.4f}")

# ------------------------------------------------------------
# 5. AVALIAR OS RESULTADOS
# ------------------------------------------------------------
ari = adjusted_rand_score(y_true, labels)
sil = silhouette_score(X_scaled, labels)

print(f"\n[5] Metricas de avaliacao:")
print(f"    Adjusted Rand Index (ARI): {ari:.4f}  (1.0 = perfeito)")
print(f"    Silhouette Score:          {sil:.4f}  (1.0 = clusters perfeitos)")

# Composicao de cada cluster
print(f"\n    Composicao dos clusters:")
for c in range(3):
    subset = df[df["cluster"] == c]["classe"].value_counts()
    dominant = subset.index[0]
    print(f"    Cluster {c} (~{dominant:6s}): {dict(subset)}")

# Centroides na escala original
centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
print(f"\n    Centroides finais (escala original):")
print(f"    {'Cluster':<10} {'Peso(g)':<12} {'Diametro':<12} {'Casca':<10} {'Gordura%'}")
print(f"    {'-'*55}")
for i, c in enumerate(centroids_orig):
    dominant = df[df["cluster"] == i]["classe"].value_counts().index[0]
    print(f"    C{i} ~{dominant:<7} {c[0]:<12.2f} {c[1]:<12.3f} {c[2]:<10.4f} {c[3]:.2f}%")

# ------------------------------------------------------------
# 6. GRAFICOS
# ------------------------------------------------------------
print(f"\n[6] Gerando graficos...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("K-Means - Dataset Avocado | Carlos Eduardo", fontsize=13, fontweight="bold")

cluster_colors = ["#2D2D2D", "#777777", "#BBBBBB"]
class_colors   = {"hass": "#2D2D2D", "fuerte": "#777777", "bacon": "#BBBBBB"}

# --- Grafico 1: Cotovelo ---
axes[0].plot(valores_k, inercias, "o-", color="#333333", linewidth=2, markersize=7)
axes[0].axvline(x=3, color="#888888", linestyle="--", linewidth=1.5, label="k=3")
axes[0].set_title("Metodo do Cotovelo")
axes[0].set_xlabel("Numero de clusters (k)")
axes[0].set_ylabel("Inercia")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.4)

# --- Grafico 2: Clusters encontrados ---
for c in range(3):
    mask = labels == c
    dominant = df[df["cluster"] == c]["classe"].value_counts().index[0]
    axes[1].scatter(X[mask, 0], X[mask, 3],
                    color=cluster_colors[c], s=30, alpha=0.75,
                    label=f"Cluster {c} (~{dominant})")
axes[1].set_title("Clusters K-Means")
axes[1].set_xlabel("Peso (g)")
axes[1].set_ylabel("Teor de gordura (%)")
axes[1].legend(fontsize=8)
axes[1].grid(True, linestyle="--", alpha=0.3)

# --- Grafico 3: Classes reais ---
for cls, col in class_colors.items():
    mask = y_true == cls
    axes[2].scatter(X[mask, 0], X[mask, 3],
                    color=col, s=30, alpha=0.75, label=cls.capitalize())
axes[2].set_title("Classes Reais")
axes[2].set_xlabel("Peso (g)")
axes[2].set_ylabel("Teor de gordura (%)")
axes[2].legend(fontsize=8)
axes[2].grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("resultado_kmeans.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n    Grafico salvo como 'resultado_kmeans.png'")
print("\n" + "=" * 55)
print("  CONCLUIDO!")
print("=" * 55)