"""
MARA × H&M Data Loader
======================
Loads H&M Kaggle dataset and populates Qdrant dual-collection
memory architecture with proper MARA memory stratification.

Dataset source:
  - Kaggle: kaggle competitions download -c h-and-m-personalized-fashion-recommendations
  - OR HuggingFace: datasets.load_dataset("Qdrant/hm_ecommerce_products")

Expected files in ./data/:
  - articles.csv
  - customers.csv
  - transactions_train.csv
"""

import os
import time
import math
import uuid
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

# ─── Constants ────────────────────────────────────────────────────────────────

DECAY_RATES = {
    "structural": 0.01,
    "semantic": 0.10,
    "episodic": 0.30,
}

STRATA_WEIGHTS = {
    "structural": 1.0,
    "semantic": 0.8,
    "episodic": 0.6,
}

VECTOR_SIZE = 384  # BGE-small-en-v1.5 / all-MiniLM-L6-v2

# EPISODIC: last 30 days from a reference date
# SEMANTIC: purchases repeated 2+ times across history
EPISODIC_WINDOW_DAYS = 30
SEMANTIC_MIN_PURCHASES = 2
BUDGET_PERCENTILE = 75  # p75 = realistic upper budget ceiling
HM_PRICE_SCALE = 590.0  # H&M Kaggle prices normalized; x590 = approximate real SEK


# ─── Init ─────────────────────────────────────────────────────────────────────

class HMDataLoader:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        print("✅ Connected to Qdrant Cloud")

    def _embed(self, text: str) -> list:
        return self.model.encode(str(text)).tolist()

    # ─── Collection Setup ──────────────────────────────────────────────────────

    def _safe_create_collection(self, name: str, vector_size: int):
        """Create collection if not exists, delete+recreate if exists (idempotent)."""
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )

    def init_collections(self):
        """Create MARA collections + all required payload indexes."""
        print("\n--- Initializing Qdrant Collections ---")

        self._safe_create_collection("hm_products", VECTOR_SIZE)
        print("OK Collection: hm_products")

        self._safe_create_collection("structural_memory", VECTOR_SIZE)
        print("OK Collection: structural_memory")

        self._safe_create_collection("semantic_episodic_memory", VECTOR_SIZE)
        print("OK Collection: semantic_episodic_memory")

        # IMPORTANT: Create payload indexes — required for all filter queries
        print("\n--- Creating Payload Indexes ---")
        index_config = {
            "structural_memory":        ["user_id", "constraint_type", "decay_class"],
            "semantic_episodic_memory": ["user_id", "memory_type", "decay_class"],
            "hm_products":              ["index_name", "colour_group_name", "garment_group_name"],
        }
        for collection, fields in index_config.items():
            for field in fields:
                try:
                    self.client.create_payload_index(
                        collection_name=collection,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                    print(f"  OK  {collection}.{field}")
                except Exception as e:
                    if "already exists" in str(e).lower() or "conflict" in str(e).lower():
                        print(f"  --  {collection}.{field} (already exists)")
                    else:
                        print(f"  ERR {collection}.{field}: {e}")


    def load_products(self, limit: int = 10_000):
        """
        Load H&M articles.csv → hm_products collection.
        Each product gets a rich text embedding from its description + metadata.
        """
        print(f"\n📦 Loading H&M product catalog (limit={limit})...")
        articles = pd.read_csv(f"{self.data_dir}/articles.csv", nrows=limit)

        # Fill nulls
        articles["detail_desc"] = articles["detail_desc"].fillna("")
        articles["colour_group_name"] = articles["colour_group_name"].fillna("Unknown")
        articles["garment_group_name"] = articles["garment_group_name"].fillna("Unknown")

        points = []
        for _, row in tqdm(articles.iterrows(), total=len(articles), desc="Embedding products"):
            # Rich text for semantic embedding
            text = (
                f"{row['prod_name']} — {row['product_type_name']} "
                f"in {row['colour_group_name']}, {row['garment_group_name']}. "
                f"{row['detail_desc']}"
            )
            vector = self._embed(text)

            points.append(
                models.PointStruct(
                    id=int(str(row["article_id"])[:15]),  # Trim to int-safe length
                    vector=vector,
                    payload={
                        "article_id": str(row["article_id"]),
                        "prod_name": row["prod_name"],
                        "product_type_name": row.get("product_type_name", ""),
                        "colour_group_name": row["colour_group_name"],
                        "perceived_colour_master_name": row.get("perceived_colour_master_name", ""),
                        "garment_group_name": row["garment_group_name"],
                        "index_name": row.get("index_name", ""),
                        "section_name": row.get("section_name", ""),
                        "detail_desc": row["detail_desc"],
                        "rich_text": text,
                    },
                )
            )

            # Batch upsert every 100
            if len(points) >= 100:
                self.client.upsert(collection_name="hm_products", points=points)
                points = []

        if points:
            self.client.upsert(collection_name="hm_products", points=points)

        print(f"✅ Loaded {limit} products into hm_products")

    # ─── Customer Memory Extraction ────────────────────────────────────────────

    def extract_customer_memory(self, customer_id: str, reference_date: str = None):
        """
        Given a customer_id, extract all 3 memory strata from H&M transaction history.

        reference_date: ISO string "YYYY-MM-DD" — simulates "today" for temporal demo.
                        Defaults to last transaction date for that customer.
        """
        print(f"\n🧠 Extracting memory for customer: {customer_id}")

        articles = pd.read_csv(f"{self.data_dir}/articles.csv")
        transactions = pd.read_csv(
            f"{self.data_dir}/transactions_train.csv",
            parse_dates=["t_dat"],
        )
        customers = pd.read_csv(f"{self.data_dir}/customers.csv")

        # Filter to this customer
        user_txns = transactions[transactions["customer_id"] == customer_id].copy()
        if user_txns.empty:
            print(f"⚠️ No transactions found for {customer_id}")
            return

        user_txns = user_txns.merge(articles, on="article_id", how="left")
        user_txns = user_txns.sort_values("t_dat")

        ref_date = pd.to_datetime(reference_date) if reference_date else user_txns["t_dat"].max()
        # Cap days_ago at 0 — purchases AFTER ref_date get days_ago=0 (treated as current)
        user_txns["days_ago"] = (ref_date - user_txns["t_dat"]).dt.days.clip(lower=0)

        # ── 1. STRUCTURAL MEMORY ───────────────────────────────────────────────
        # H&M Kaggle prices are normalized (~0.005–0.99); x590 = real SEK
        budget_raw = float(np.percentile(user_txns["price"], BUDGET_PERCENTILE))
        budget = round(budget_raw * HM_PRICE_SCALE, 2)
        print(f"  💰 Raw p25 price: {budget_raw:.4f} → {budget:.2f} SEK")

        gender_index = user_txns["index_name"].mode()[0] if not user_txns["index_name"].isna().all() else "Unknown"
        customer_meta = customers[customers["customer_id"] == customer_id].iloc[0] if len(customers[customers["customer_id"] == customer_id]) > 0 else None
        age = int(customer_meta["age"]) if customer_meta is not None and not pd.isna(customer_meta.get("age", float("nan"))) else None

        structural_constraints = [
            {
                "constraint_type": "budget",
                "value": budget,
                "description": f"Maximum budget: {budget:.0f} SEK (derived from purchase history p25)",
                "decay_class": "structural",
            },
            {
                "constraint_type": "gender_index",
                "value": gender_index,
                "description": f"Primary shopping section: {gender_index}",
                "decay_class": "structural",
            },
        ]
        if age:
            structural_constraints.append({
                "constraint_type": "age_group",
                "value": age,
                "description": f"Customer age: {age} — relevant for style recommendations",
                "decay_class": "structural",
            })

        self._store_structural(customer_id, structural_constraints, ref_date)

        # ── 2. SEMANTIC MEMORY ─────────────────────────────────────────────────
        # Items purchased 2+ times = stable taste signal
        article_counts = user_txns["article_id"].value_counts()
        repeated_articles = article_counts[article_counts >= SEMANTIC_MIN_PURCHASES].index.tolist()
        semantic_rows = user_txns[user_txns["article_id"].isin(repeated_articles)].drop_duplicates("article_id")

        # Top colors and garment groups from full history
        top_colors = user_txns["colour_group_name"].value_counts().head(3).index.tolist()
        top_garments = user_txns["garment_group_name"].value_counts().head(3).index.tolist()

        semantic_memories = []
        for _, row in semantic_rows.iterrows():
            text = (
                f"Repeatedly purchases: {row.get('prod_name', '')} — "
                f"{row.get('product_type_name', '')} in {row.get('colour_group_name', '')}"
            )
            semantic_memories.append({
                "text": text,
                "memory_type": "semantic",
                "timestamp": row["t_dat"].timestamp(),
                "days_ago": int(row["days_ago"]),
            })

        # Color & garment style preferences
        if top_colors:
            semantic_memories.append({
                "text": f"Preferred colors: {', '.join(top_colors)}",
                "memory_type": "semantic",
                "timestamp": ref_date.timestamp(),
                "days_ago": 0,
            })
        if top_garments:
            semantic_memories.append({
                "text": f"Preferred garment types: {', '.join(top_garments)}",
                "memory_type": "semantic",
                "timestamp": ref_date.timestamp(),
                "days_ago": 0,
            })

        self._store_semantic_episodic(customer_id, semantic_memories)

        # ── 3. EPISODIC MEMORY ─────────────────────────────────────────────────
        # Cap episodic to 15 most recent (avoid memory flood like 693 entries!)
        # Sort ascending by days_ago = freshest first
        recent_txns = (
            user_txns[user_txns["days_ago"] <= EPISODIC_WINDOW_DAYS]
            .sort_values("days_ago")
            .head(15)
        )
        episodic_memories = []
        for _, row in recent_txns.iterrows():
            price_sek = round(float(row["price"]) * HM_PRICE_SCALE, 0)
            text = (
                f"Recently purchased: {row.get('prod_name', '')} "
                f"({row.get('colour_group_name', '')}, {row.get('garment_group_name', '')}) "
                f"at {price_sek:.0f} SEK - {int(row['days_ago'])} days ago"
            )
            episodic_memories.append({
                "text": text,
                "memory_type": "episodic",
                "timestamp": row["t_dat"].timestamp(),
                "days_ago": int(row["days_ago"]),
            })

        self._store_semantic_episodic(customer_id, episodic_memories)

        print(f"✅ Memory stored: {len(structural_constraints)} structural, "
              f"{len(semantic_memories)} semantic, {len(episodic_memories)} episodic")

        return {
            "structural": structural_constraints,
            "semantic_count": len(semantic_memories),
            "episodic_count": len(episodic_memories),
            "budget": budget,
            "gender_index": gender_index,
            "top_colors": top_colors,
            "top_garments": top_garments,
        }

    # ─── Qdrant Storage Helpers ────────────────────────────────────────────────

    def _store_structural(self, user_id: str, constraints: list, ref_date):
        points = []
        for c in constraints:
            vector = self._embed(c["description"])
            points.append(
                models.PointStruct(
                    id=abs(hash(f"{user_id}_{c['constraint_type']}")) % (10**15),
                    vector=vector,
                    payload={
                        "user_id": user_id,
                        "constraint_type": c["constraint_type"],
                        "value": c["value"],
                        "description": c["description"],
                        "decay_class": "structural",
                        "timestamp": ref_date.timestamp(),
                        "days_ago_at_storage": 0,  # structural = always current
                    },
                )
            )
        if points:
            self.client.upsert(collection_name="structural_memory", points=points)

    def _store_semantic_episodic(self, user_id: str, memories: list):
        points = []
        for m in memories:
            vector = self._embed(m["text"])
            points.append(
                models.PointStruct(
                    id=abs(hash(f"{user_id}_{m['text'][:50]}_{m['timestamp']}")) % (10**15),
                    vector=vector,
                    payload={
                        "user_id": user_id,
                        "text": m["text"],
                        "memory_type": m["memory_type"],
                        "decay_class": m["memory_type"],
                        "timestamp": m["timestamp"],
                        "days_ago_at_storage": m["days_ago"],
                    },
                )
            )
        if points:
            self.client.upsert(collection_name="semantic_episodic_memory", points=points)


# ─── CLI Usage ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = HMDataLoader(data_dir="./data")

    # Step 1: Initialize Qdrant collections
    loader.init_collections()

    # Step 2: Load product catalog (start with 5k for speed)
    loader.load_products(limit=5_000)

    # Step 3: Extract memory for a demo customer
    # Find a good customer: one with long history
    transactions = pd.read_csv("./data/transactions_train.csv")
    top_customers = transactions["customer_id"].value_counts().head(10)
    demo_customer = top_customers.index[0]
    print(f"\n🎯 Demo customer: {demo_customer} ({top_customers[demo_customer]} transactions)")

    # Simulate "today" = mid-point of their history for 6-month demo
    user_txns = transactions[transactions["customer_id"] == demo_customer]
    user_txns = user_txns.copy()
    user_txns["t_dat"] = pd.to_datetime(user_txns["t_dat"])
    mid_date = user_txns["t_dat"].quantile(0.5, interpolation="nearest")
    print(f"📅 Reference date (mid-point): {mid_date.date()}")

    profile = loader.extract_customer_memory(demo_customer, reference_date=str(mid_date.date()))
    print(f"\n📊 Customer Profile Summary:")
    print(f"  Budget ceiling: {profile['budget']:.2f} SEK")
    print(f"  Primary section: {profile['gender_index']}")
    print(f"  Favorite colors: {profile['top_colors']}")
    print(f"  Favorite garments: {profile['top_garments']}")
    print(f"\n✅ MARA is ready. Run main.py to start the agent.")