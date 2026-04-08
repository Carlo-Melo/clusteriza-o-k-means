"""
Atividade 7 (comparação) — Classificação com k-NN no Dataset Avocado

Objetivo:
- Usar o MESMO dataset do K-Means (Avocado.csv) para classificar a coluna 'classe'
- Comparar o desempenho do k-NN (supervisionado) com uma "classificação derivada" do K-Means
  (mapeando cada cluster para a classe dominante por voto majoritário)
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def majority_vote_mapping(clusters: np.ndarray, y: np.ndarray) -> dict[int, str]:
    """Retorna um dicionário {cluster_id: classe_dominante}."""
    mapping: dict[int, str] = {}
    for c in np.unique(clusters):
        classes, counts = np.unique(y[clusters == c], return_counts=True)
        mapping[int(c)] = str(classes[np.argmax(counts)])
    return mapping


def main() -> None:
    print("=" * 65)
    print("  k-NN (Classificação) — Dataset Avocado | Comparação com K-Means")
    print("=" * 65)

    df = pd.read_csv("Avocado.csv")
    features = ["peso", "diâmetro", "espessura_casca", "teor_gordura"]
    X = df[features].to_numpy()
    y = df["classe"].to_numpy()

    print(f"\n[1] Dataset carregado")
    print(f"    Linhas: {df.shape[0]} | Colunas: {df.shape[1]}")
    print("\n    Distribuição de classes:")
    print(pd.Series(y).value_counts().to_string())

    # ------------------------------------------------------------
    # 2) Split supervisionado para k-NN (com estratificação)
    # ------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n[2] Split treino/teste")
    print(f"    Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

    # ------------------------------------------------------------
    # 3) Escolha do melhor k para k-NN via validação cruzada
    # ------------------------------------------------------------
    print("\n[3] Seleção de k (k-NN) com validação cruzada (StratifiedKFold, 5 folds)")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidate_ks = [1, 3, 5, 7, 9, 11, 13, 15]
    cv_scores: dict[int, float] = {}

    for k in candidate_ks:
        clf = KNeighborsClassifier(n_neighbors=k)
        fold_accs: list[float] = []
        for train_idx, val_idx in skf.split(X_train_scaled, y_train):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            fold_accs.append(accuracy_score(y_val, preds))

        cv_scores[k] = float(np.mean(fold_accs))
        print(f"    k={k:<2d} -> acc média={cv_scores[k]:.4f}")

    best_k = max(cv_scores, key=cv_scores.get)
    print(f"\n    Melhor k pela CV: k={best_k} (acc média={cv_scores[best_k]:.4f})")

    # ------------------------------------------------------------
    # 4) Treinar k-NN final e avaliar no teste
    # ------------------------------------------------------------
    print("\n[4] Treinando k-NN final e avaliando no conjunto de teste")
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    acc_knn = accuracy_score(y_test, y_pred)
    print(f"    Acurácia (k-NN): {acc_knn:.4f}")
    print("\n    Matriz de confusão (k-NN):")
    labels_sorted = sorted(np.unique(y))
    print(confusion_matrix(y_test, y_pred, labels=labels_sorted))
    print("\n    Relatório de classificação (k-NN):")
    print(classification_report(y_test, y_pred, digits=4))

    # ------------------------------------------------------------
    # 5) K-Means como "baseline" não supervisionada
    #    (a) medir ARI e silhouette (como no seu script)
    #    (b) derivar uma classificação por voto majoritário em cada cluster
    #        para comparar com a acurácia do k-NN no MESMO split de teste
    # ------------------------------------------------------------
    print("\n[5] Comparação com K-Means (não supervisionado)")
    X_all_scaled = scaler.fit_transform(X)  # scaler independente no dataset todo (avaliação do K-Means)

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    kmeans.fit(X_all_scaled)
    clusters_all = kmeans.labels_

    ari = adjusted_rand_score(y, clusters_all)
    sil = silhouette_score(X_all_scaled, clusters_all)
    print(f"    ARI (K-Means vs classe): {ari:.4f}")
    print(f"    Silhouette (K-Means):   {sil:.4f}")

    # Para "acurácia do K-Means", mapeamos cluster -> classe dominante no TREINO
    # e aplicamos no TESTE. Assim a comparação fica justa com o split supervisionado.
    scaler_km = StandardScaler()
    X_train_km = scaler_km.fit_transform(X_train)
    X_test_km = scaler_km.transform(X_test)

    kmeans_split = KMeans(n_clusters=3, n_init=10, random_state=42)
    kmeans_split.fit(X_train_km)
    train_clusters = kmeans_split.predict(X_train_km)
    test_clusters = kmeans_split.predict(X_test_km)

    mapping = majority_vote_mapping(train_clusters, y_train)
    y_pred_kmeans = np.array([mapping[int(c)] for c in test_clusters], dtype=object)
    acc_kmeans = accuracy_score(y_test, y_pred_kmeans)

    print("\n    'Classificação' derivada do K-Means (voto majoritário por cluster):")
    print(f"    Mapeamento cluster->classe: {mapping}")
    print(f"    Acurácia equivalente (K-Means): {acc_kmeans:.4f}")
    print("\n    Matriz de confusão (K-Means -> classe):")
    print(confusion_matrix(y_test, y_pred_kmeans, labels=labels_sorted))

    # ------------------------------------------------------------
    # 6) Gráficos (mesmo estilo do kmeans_avocado.py: 3 painéis, PNG + show)
    # ------------------------------------------------------------
    print("\n[6] Gerando graficos...")

    class_colors = {"hass": "#2D2D2D", "fuerte": "#777777", "bacon": "#BBBBBB"}
    ks_ordered = sorted(candidate_ks)
    accs_plot = [cv_scores[k] for k in ks_ordered]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "k-NN - Dataset Avocado | Validacao e classes (conjunto de teste)",
        fontsize=13,
        fontweight="bold",
    )

    # --- Gráfico 1: acurácia na CV vs k (análogo ao "cotovelo" do K-Means) ---
    axes[0].plot(ks_ordered, accs_plot, "o-", color="#333333", linewidth=2, markersize=7)
    axes[0].axvline(
        x=best_k,
        color="#888888",
        linestyle="--",
        linewidth=1.5,
        label=f"k={best_k}",
    )
    axes[0].set_title("Selecao de k (validacao cruzada)")
    axes[0].set_xlabel("k (vizinhos)")
    axes[0].set_ylabel("Acuracia media (CV)")
    axes[0].set_xticks(ks_ordered)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # --- Gráfico 2: predições do k-NN no teste (peso x teor de gordura) ---
    for cls in labels_sorted:
        mask = y_pred == cls
        col = class_colors[str(cls)]
        axes[1].scatter(
            X_test[mask, 0],
            X_test[mask, 3],
            color=col,
            s=30,
            alpha=0.75,
            label=str(cls).capitalize(),
        )
    axes[1].set_title(f"Predicoes k-NN (k={best_k}) — teste")
    axes[1].set_xlabel("Peso (g)")
    axes[1].set_ylabel("Teor de gordura (%)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    # --- Gráfico 3: classes reais no teste (mesmo plano do K-Means) ---
    for cls in labels_sorted:
        mask = y_test == cls
        col = class_colors[str(cls)]
        axes[2].scatter(
            X_test[mask, 0],
            X_test[mask, 3],
            color=col,
            s=30,
            alpha=0.75,
            label=str(cls).capitalize(),
        )
    axes[2].set_title("Classes reais — teste")
    axes[2].set_xlabel("Peso (g)")
    axes[2].set_ylabel("Teor de gordura (%)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("resultado_knn.png", dpi=150, bbox_inches="tight")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()

    print("\n    Grafico salvo como 'resultado_knn.png'")

    print("\n" + "=" * 65)
    print("Concluído.")
    print("=" * 65)


if __name__ == "__main__":
    main()

