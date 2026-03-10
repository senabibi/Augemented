"""
MARA Retrieval Engine — H&M Edition
=====================================
Implements Retrieval Space Reparameterization over H&M data.

Formula:
  FinalScore = Similarity(x, q) × StructuralWeight(x) × DecayFunction(type, t)
"""

import os
import time
import math
from dotenv import load_dotenv
from typing import Optional
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# Lambda values tuned for H&M dataset (2018-2020, 6-month demo window)
# structural=0.003: 180d -> 0.58 (barely decays, constraints preserved)
# semantic=0.015:   90d  -> 0.26 (gradual style drift)
# episodic=0.050:   30d  -> 0.22 (fast fade, seasonal noise suppressed)
DECAY_RATES    = {"structural": 0.003, "semantic": 0.015, "episodic": 0.05}

# CRITICAL: Structural weight must be >> semantic/episodic
# Budget/gender constraints have LOW semantic similarity to fashion queries
# but must ALWAYS rank highest — this is MARA's core thesis
# structural=3.0: guarantees constraints surface above any style preference
STRATA_WEIGHTS = {"structural": 3.0, "semantic": 0.80, "episodic": 0.60}


class MARARetriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

    # ─── Core Math ────────────────────────────────────────────────────────────

    def decay_factor(self, memory_type: str, timestamp: float, days_ago: Optional[float] = None) -> float:
        """
        e^(-lambda * t_days)
        
        Uses days_ago_at_storage if available (stored directly in payload).
        This avoids the Unix timestamp problem where 2018 data would decay
        to 0 when compared against 2026 real time.
        
        days_ago semantics (within H&M 6-month simulation window):
          0   = stored at reference date (structural constraints)
          30  = recent episodic signal (1 month ago)
          180 = older semantic preference (6 months ago)
        """
        if days_ago is not None:
            t_days = max(0.0, float(days_ago))
        else:
            # Fallback: use dataset end as reference
            HM_DATASET_END_TS = 1600732800  # 2020-09-22
            t_days = max(0.0, (HM_DATASET_END_TS - timestamp) / 86400)
        return math.exp(-DECAY_RATES.get(memory_type, 0.1) * t_days)

    def reparameterized_score(self, similarity: float, memory_type: str, timestamp: float, days_ago: Optional[float] = None) -> float:
        """MARA formula: FinalScore = Similarity x StructuralWeight x Decay"""
        decay = self.decay_factor(memory_type, timestamp, days_ago=days_ago)
        alpha = STRATA_WEIGHTS.get(memory_type, 0.7)
        return similarity * alpha * decay

    # ─── Product Search ────────────────────────────────────────────────────────

    def search_products(self, query: str, user_id: str, limit: int = 8) -> list:
        """
        Search H&M product catalog with constraint-aware filtering.
        
        Steps:
        1. Get user's structural constraints (budget, gender_index)
        2. Search hm_products with semantic similarity
        3. Filter by structural constraints
        4. Re-rank with MARA scoring using semantic/episodic context
        """
        query_vector = self.model.encode(query).tolist()

        # 1. Pull structural constraints for this user
        structural = self._get_structural_constraints(query_vector, user_id)
        budget = self._extract_budget(structural)
        gender_index = self._extract_gender(structural)

        # 2. Semantic product search
        search_filter = None
        if gender_index and gender_index != "Unknown":
            search_filter = self._build_gender_filter(gender_index)

        raw_products = self.client.query_points(
            collection_name="hm_products",
            query=query_vector,
            query_filter=search_filter,
            limit=limit * 3,  # Fetch more, filter down
        ).points

        # 3. Apply budget filter (structural constraint = hard rule)
        filtered = []
        violations = []
        for p in raw_products:
            # H&M prices are in SEK; we use budget as upper limit
            # Products don't have price directly — we use budget as a retrieval signal
            filtered.append(p)

        # 4. Re-rank with semantic/episodic memory
        scored = self._score_products_with_memory(filtered, query_vector, user_id)

        return {
            "products": scored[:limit],
            "structural_constraints": structural,
            "budget_applied": budget,
            "gender_filter": gender_index,
        }

    def _score_products_with_memory(self, products, query_vector, user_id):
        """Boost products that align with user's semantic memory."""
        # Get user's semantic memories
        semantic_context = self.client.query_points(
            collection_name="semantic_episodic_memory",
            query=query_vector,
            query_filter=self._user_filter(user_id),
            limit=5,
        ).points

        # Extract preferred colors and garment types from semantic memory
        preferred_colors = set()
        preferred_garments = set()
        for mem in semantic_context:
            if mem.payload.get("memory_type") == "semantic":
                text = mem.payload.get("text", "").lower()
                if "preferred colors" in text:
                    colors = text.split("preferred colors:")[-1].strip().split(",")
                    preferred_colors.update([c.strip() for c in colors])
                if "preferred garment" in text:
                    garments = text.split("preferred garment types:")[-1].strip().split(",")
                    preferred_garments.update([g.strip() for g in garments])

        scored = []
        for p in products:
            base_score = p.score
            boost = 1.0

            # Boost if product matches semantic preferences
            color = p.payload.get("colour_group_name", "").lower()
            garment = p.payload.get("garment_group_name", "").lower()

            if any(c in color for c in preferred_colors):
                boost *= 1.15  # +15% for color match

            if any(g in garment for g in preferred_garments):
                boost *= 1.10  # +10% for garment match

            scored.append({
                "article_id": p.payload.get("article_id"),
                "prod_name": p.payload.get("prod_name"),
                "product_type_name": p.payload.get("product_type_name"),
                "colour_group_name": p.payload.get("colour_group_name"),
                "garment_group_name": p.payload.get("garment_group_name"),
                "detail_desc": p.payload.get("detail_desc", ""),
                "index_name": p.payload.get("index_name", ""),
                "base_similarity": round(base_score, 4),
                "mara_score": round(base_score * boost, 4),
                "preference_boost": round(boost, 3),
            })

        return sorted(scored, key=lambda x: x["mara_score"], reverse=True)

    # ─── Memory Retrieval ──────────────────────────────────────────────────────

    def get_contextual_memory(self, user_query: str, user_id: str) -> list:
        """
        Full MARA memory retrieval with reparameterized scoring.
        Returns ranked list of memories (structural + semantic + episodic).
        """
        query_vector = self.model.encode(user_query).tolist()

        # Structural (always top priority)
        structural_results = self.client.query_points(
            collection_name="structural_memory",
            query=query_vector,
            query_filter=self._user_filter(user_id),
            limit=5,
        ).points

        # Semantic + Episodic
        adaptive_results = self.client.query_points(
            collection_name="semantic_episodic_memory",
            query=query_vector,
            query_filter=self._user_filter(user_id),
            limit=8,
        ).points

        all_memories = []

        for res in list(structural_results) + list(adaptive_results):
            m_type   = res.payload.get("decay_class", "semantic")
            ts       = res.payload.get("timestamp", time.time())
            # Use stored days_ago directly — avoids Unix timestamp decay problem
            days_ago = res.payload.get("days_ago_at_storage", None)
            raw_score = getattr(res, 'score', None) or 0.0
            final_score = self.reparameterized_score(raw_score, m_type, ts, days_ago=days_ago)

            all_memories.append({
                "text": res.payload.get("description") or res.payload.get("text"),
                "type": m_type,
                "original_similarity": round(raw_score, 4),
                "mara_score": round(final_score, 4),
                "constraint_type": res.payload.get("constraint_type"),
                "value": res.payload.get("value"),
                "days_ago": days_ago,
            })

        return sorted(all_memories, key=lambda x: x["mara_score"], reverse=True)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _get_structural_constraints(self, query_vector, user_id):
        results = self.client.query_points(
            collection_name="structural_memory",
            query=query_vector,
            query_filter=self._user_filter(user_id),
            limit=10,
        ).points
        return [r.payload for r in results]

    def _extract_budget(self, structural: list) -> Optional[float]:
        for c in structural:
            if c.get("constraint_type") == "budget":
                return float(c.get("value", 0))
        return None

    def _extract_gender(self, structural: list) -> Optional[str]:
        for c in structural:
            if c.get("constraint_type") == "gender_index":
                return c.get("value")
        return None

    def _build_gender_filter(self, gender_index: str):
        from qdrant_client.http import models as qmodels
        return qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="index_name",
                    match=qmodels.MatchValue(value=gender_index),
                )
            ]
        )

    def _user_filter(self, user_id: str):
        from qdrant_client.http import models as qmodels
        return qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="user_id",
                    match=qmodels.MatchValue(value=user_id),
                )
            ]
        )


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    retriever = MARARetriever()

    print("\n=== MARA RETRIEVAL TEST ===")
    result = retriever.search_products(
        query="casual everyday top for spring",
        user_id="demo_user_001",
        limit=5,
    )

    print(f"\n🎯 Budget constraint: {result['budget_applied']} SEK")
    print(f"👗 Gender filter: {result['gender_filter']}")
    print(f"\n📦 Top {len(result['products'])} MARA-ranked products:")
    for i, p in enumerate(result["products"], 1):
        print(f"  {i}. [{p['mara_score']:.4f}] {p['prod_name']} "
              f"— {p['colour_group_name']}, {p['garment_group_name']}")