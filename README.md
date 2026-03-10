# MARA × H&M Fashion — Setup Guide

## 1. Dataset Download (Kaggle)

```bash
# Option A: Kaggle CLI (recommended)
pip install kaggle
kaggle competitions download -c h-and-m-personalized-fashion-recommendations
unzip h-and-m-personalized-fashion-recommendations.zip -d ./data

# Option B: HuggingFace (products only, pre-embedded)
python -c "from datasets import load_dataset; ds = load_dataset('Qdrant/hm_ecommerce_products'); ds['train'].to_csv('./data/articles_hf.csv')"
```

Files needed in `./data/`:
- `articles.csv`       — 106k products
- `customers.csv`      — 1.37M customers  
- `transactions_train.csv` — Full purchase history 2018-2020

---

## 2. Environment Setup

```bash
# Create virtual env
python -m venv mara_env
source mara_env/bin/activate  # Windows: mara_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Create `.env`:
```
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
GROQ_API_KEY=your-groq-key
```

---

## 3. Initialize & Load Data

```bash
# Load H&M products + extract demo customer memory
python hm_data_loader.py
```

---

## 4. Run 6-Month Simulation (Benchmark)

```bash
python simulate_6month.py --data_dir ./data
```

Expected output:
```
Month 1: ✅ Structural: 2.8431 vs Episodic: 1.2100  
Month 3: ✅ Structural: 2.7980 vs Episodic: 0.8900 (episodic decayed)
Month 6: ✅ Structural: 2.7650 vs Episodic: 0.3200 (fully faded)

MARA violation rate:    0/3 = 0%
Baseline RAG (sim):     ~37%
```

---

## 5. Start Backend

```bash
python main.py
# → http://localhost:8000
# → http://localhost:8000/docs (Swagger UI)
```

---

## 6. MARA Memory Architecture

```
Customer Query
      │
      ▼
┌─────────────────────────────────────┐
│         MARARetriever               │
│                                     │
│  structural_memory (λ=0.01)         │  ← Budget, gender, age — NEVER fade
│  semantic_episodic_memory (λ=0.1/0.3)│  ← Style/recent browsing — decay
│  hm_products (catalog search)       │  ← H&M article embeddings
│                                     │
│  Score = Similarity × Weight × Decay│  ← Reparameterized geometry
└─────────────────────────────────────┘
      │
      ▼
 Groq LLaMA 3.3 70B
      │
      ▼
 Fashion Recommendation
 (constraints guaranteed)
```

---

## 7. H&M → MARA Memory Mapping

| H&M Data | MARA Memory Type | λ |
|---|---|---|
| `customers.age` | Structural | 0.01 |
| `transactions.price` (p25) | Structural budget | 0.01 |
| `articles.index_name` mode | Structural gender | 0.01 |
| Repeated purchase colors | Semantic | 0.10 |
| Repeated garment types | Semantic | 0.10 |
| Last 30-day transactions | Episodic | 0.30 |
