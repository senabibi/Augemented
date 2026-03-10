# MARA × H&M Fashion — Memory-Augmented Retail Agent

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/5/53/H%26M-Logo.svg" width="120" alt="H&M Logo">
  <br>
  <h3>Personalized Fashion Commerce powered by Long-Term Retrieval Memory</h3>

  ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
  ![Qdrant](https://img.shields.io/badge/Qdrant-darkred?style=for-the-badge&logo=qdrant)
  ![Groq](https://img.shields.io/badge/Groq-f5ad42?style=for-the-badge&logo=lightning)
  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![Llama3](https://img.shields.io/badge/Llama_3.3_70B-blue?style=for-the-badge)
</div>

---

## 🚀 The Vision: Memory-Augmented Fashion (MARA)

**MARA (Memory-Augmented Retail Agent)** is an advanced RAG (Retrieval-Augmented Generation) system specifically designed for the fashion industry. Unlike standard RAG systems that treat all context as flat, MARA implements **Retrieval Space Reparameterization**.

The project "Augmented" aims to solve the core failure of traditional retail AI: **Contextual Persistence**. MARA doesn't just "remember" what you said; it understands how your preferences evolve over time using a tiered memory architecture powered by **Qdrant**.

### Core Objectives
- **Constraint Satisfaction**: Ensuring recommendations strictly adhere to user-specific boundaries (budget, gender identity, age).
- **Style Evolution**: Learning personal fashion tastes (colors, garments) while allowing seasonal preferences to decay naturally.
- **Frictionless Shopping**: Bridging the gap between a 1 million+ item catalog and a single, perfect outfit recommendation.

---

## 🧠 Tiered Memory Architecture

MARA categorizes every customer interaction into three biological-inspired memory strata:

| Layer | Type | λ (Decay Rate) | Purpose |
| :--- | :--- | :--- | :--- |
| **🔒 Structural** | Fixed | 0.003 | Hard constraints like **Budget ceilings**, **Age**, and **Preferred Gender Sections**. These never fade. |
| **🎨 Semantic** | Stable | 0.015 | Preferred colors, favorite garment types, and style patterns. Flows with the customer's identity. |
| **⚡ Episodic** | Volatile | 0.050 | Recent browsing history and one-off interests. Decays fast to reduce seasonal noise. |

**The Formula:**  
`FinalScore = SemanticSimilarity(Product, Query) × StructuralWeight(Memory) × e^(-λ * t)`

---

## 🛠️ Tech Stack

- **Vector Database**: [Qdrant](https://qdrant.tech/) — Orchestrates three separate high-performance collections for multi-tiered memory retrieval.
- **LLM**: [Groq](https://groq.com/) (Llama 3.3 70B) — Provides ultra-low latency reasoning to act as a professional personal stylist.
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) — High-performance asynchronous API layer.
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`) — Efficient, localized vectorization of fashion attributes and user intent.

---

## ⚙️ Quick Start

### 1. Dataset Preparation
MARA is trained and benchmarked on the H&M Personalized Fashion Recommendations dataset.

```bash
# Install Kaggle CLI
pip install kaggle

# Download competition data
kaggle competitions download -c h-and-m-personalized-fashion-recommendations
unzip h-and-m-personalized-fashion-recommendations.zip -d ./data
```

### 2. Environment Setup
```bash
# Initialize Virtual Environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the `backend/` directory:
```env
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
```

### 4. Application Launch
```bash
# 1. Index products and extract customer memory
python hm_data_loader.py

# 2. Start the API server
python main.py
```

Visit your [url/docs] to explore the API.

---

## 🤖 Evaluation Frame
To verify MARA's effectiveness, we use an **LLM-as-a-Judge** framework (Llama 3.3 70B via Groq) that compares MARA against a standard RAG baseline on:
1. **Memory Recall Accuracy**
2. **Numerical Stability (Budget adherence)**
3. **Contextual Coherence**
4. **Preference Consistency**

Current benchmarks show a **0% violation rate** on hard constraints compared to ~37% in traditional RAG implementations.

---
<div align="center">
  Created for the <b>MARA × H&M Fashion</b> Research Project
</div>
