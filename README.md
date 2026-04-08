# Atividade 6 — Clusterização com K-Means (Dataset Avocado)

Projeto de **Inteligência Artificial**: aplicação do algoritmo **K-Means** sobre dados de abacates (peso, diâmetro, espessura da casca e teor de gordura), com método do cotovelo, métricas de avaliação (ARI e silhouette) e gráficos comparando clusters encontrados e classes reais.

## Conteúdo do repositório

| Arquivo | Descrição |
|---------|-----------|
| `kmeans_avocado.py` | Script principal (carrega dados, normaliza, treina K-Means, avalia e gera figura) |
| `knn_avocado.py` | Script de **classificação com k-NN** e comparação com K-Means no mesmo dataset |
| `Avocado.csv` | Dataset com atributos numéricos e coluna `classe` (variedades) |
| `README.md` | Este arquivo |

Ao executar os scripts, são gerados **`resultado_kmeans.png`** (K-Means) e **`resultado_knn.png`** (k-NN). Você pode versionar só o código e o CSV, ou incluir as imagens como artefatos da execução.

## Requisitos

- **Python 3.8+** (testado com Python 3.11)
- Bibliotecas: `pandas`, `numpy`, `matplotlib`, `scikit-learn`

## Como clonar e rodar (GitHub)

### 1. Clonar o repositório

```bash
git clone <URL_DO_SEU_REPOSITORIO>.git
cd <nome-da-pasta-do-repo>/atividade6
```

Se o repositório for só a pasta `atividade6`, ajuste o `cd` para o diretório onde estão `kmeans_avocado.py` e `Avocado.csv`.

### 2. Criar ambiente virtual (recomendado)

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependências

**Windows** — se `pip` não for reconhecido, use sempre o Python como módulo:

```powershell
py -m pip install --upgrade pip
py -m pip install pandas numpy matplotlib scikit-learn
```

**Linux / macOS** (com venv ativado):

```bash
pip install --upgrade pip
pip install pandas numpy matplotlib scikit-learn
```

### 4. Executar o projeto

Na pasta onde estão `kmeans_avocado.py` e `Avocado.csv`:

**Windows:**

```powershell
py kmeans_avocado.py
```

Para rodar a **classificação com k-NN** e a comparação com K-Means:

```powershell
py knn_avocado.py
```

**Linux / macOS:**

```bash
python kmeans_avocado.py
```

Para rodar a **classificação com k-NN** e a comparação com K-Means:

```bash
python knn_avocado.py
```

O `kmeans_avocado.py` imprime estatísticas no terminal, abre a janela do gráfico (se o ambiente tiver interface gráfica) e salva **`resultado_kmeans.png`**. O `knn_avocado.py` salva **`resultado_knn.png`** (e tenta abrir a janela, exceto em backend não interativo).

### 5. Subir para o GitHub

1. Crie um repositório vazio no GitHub (sem README, se já tiver um local).
2. Na pasta do projeto:

```bash
git init
git add kmeans_avocado.py Avocado.csv README.md
git commit -m "Adiciona atividade 6 K-Means e dataset Avocado"
git branch -M main
git remote add origin https://github.com/<seu-usuario>/<seu-repo>.git
git push -u origin main
```

Se quiser incluir a imagem gerada: `git add resultado_kmeans.png` antes do `commit`.

Sugestão: adicione um `.gitignore` com `.venv/` e `__pycache__/` para não enviar ambiente virtual e cache do Python.

## Autor

Carlos Eduardo — Atividade 6 (Engenharia de Software / Inteligência Artificial).
