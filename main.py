"""
MARA Backend — H&M Fashion Edition
====================================
FastAPI server with H&M-aware fashion agent.
Endpoints:
  POST /chat          — Main agent chat
  POST /onboard       — Extract & store customer memory from H&M data
  GET  /profile/{id}  — Get customer constraint profile
  POST /evaluate      — LLM-as-a-Judge scoring
"""

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path

from retrieval_engine import MARARetriever

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="MARA — Memory-Augmented Retail Agent (H&M Fashion)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = MARARetriever()


# ─── Request / Response Models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    message: str
    simulation_month: Optional[int] = None  # 1, 3, or 6 — for demo mode


class EvaluateRequest(BaseModel):
    user_id: str
    query: str
    mara_response: str
    baseline_response: str


# ─── LLM Helpers ──────────────────────────────────────────────────────────────

def call_groq(system_prompt: str, user_message: str, temperature: float = 0.6) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY missing."

    client = Groq(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Groq API Error: {e}"


HM_PRICE_SCALE = 590.0  # Same scale factor as hm_data_loader.py

def build_constraint_summary(structural: list) -> dict:
    """Extract key constraints from structural memory for violation checking."""
    summary = {"budget": None, "gender_index": None, "age": None}
    for c in structural:
        ct = c.get("constraint_type")
        if ct == "budget":
            raw_val = float(c.get("value", 0))
            # If value looks like a normalized price (< 5.0), rescale to SEK
            # This handles data loaded before the HM_PRICE_SCALE fix was applied
            if raw_val < 5.0:
                raw_val = round(raw_val * HM_PRICE_SCALE, 2)
            summary["budget"] = raw_val
        elif ct == "gender_index":
            summary["gender_index"] = c.get("value")
        elif ct == "age_group":
            summary["age"] = int(c.get("value", 0))
    return summary


def detect_violations(message: str, constraints: dict) -> list:
    """
    Heuristic constraint violation detection.
    Returns list of violated constraint names.
    """
    violations = []
    message_lower = message.lower()

    # Budget check: keywords implying high spend
    high_spend_words = ["expensive", "luxury", "premium", "high-end", "designer", "splurge"]
    if constraints.get("budget") and any(w in message_lower for w in high_spend_words):
        violations.append(f"budget ({constraints['budget']:.0f} SEK)")

    # Gender index mismatch signals
    if constraints.get("gender_index") == "Ladieswear":
        male_words = ["menswear", "men's suit", "tie", "blazer for men"]
        if any(w in message_lower for w in male_words):
            violations.append("gender_index (Ladieswear)")

    return violations


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 1. Retrieve all memories + product search
        memories = retriever.get_contextual_memory(request.message, request.user_id)
        product_search = retriever.search_products(request.message, request.user_id, limit=5)

        # 2. Build constraint summary
        structural_payloads = [
            m for m in memories if m["type"] == "structural"
        ]
        constraints = build_constraint_summary(
            [{"constraint_type": m.get("constraint_type"), "value": m.get("value")} 
             for m in structural_payloads]
        )
        violations = detect_violations(request.message, constraints)

        # 3. Format context for LLM
        constraint_lines = "\n".join([
            f"  🔒 {m['text']} [MARA score: {m['mara_score']:.4f}]"
            for m in memories if m["type"] == "structural"
        ])

        preference_lines = "\n".join([
            f"  🎨 {m['text']} [score: {m['mara_score']:.4f}]"
            for m in memories if m["type"] == "semantic"
        ][:4])

        episodic_lines = "\n".join([
            f"  ⚡ {m['text']} [score: {m['mara_score']:.4f}]"
            for m in memories if m["type"] == "episodic"
        ][:3])

        product_lines = "\n".join([
            f"  • {p['prod_name']} — {p['colour_group_name']}, "
            f"{p['garment_group_name']} [MARA: {p['mara_score']:.4f}]"
            for p in product_search["products"][:5]
        ])

        budget_note = (
            f"Budget ceiling: {constraints['budget']:.0f} SEK" 
            if constraints["budget"] else "Budget: not established"
        )

        # 4. Build system prompt
        system_prompt = f"""You are MARA, a warm and knowledgeable personal fashion stylist for H&M.
You know this customer well — their budget, preferred style, and purchase history.

CUSTOMER BUDGET & CONSTRAINTS:
{constraint_lines or "  No constraints stored yet."}
  {budget_note}

CUSTOMER STYLE PREFERENCES:
{preference_lines or "  No style preferences stored yet."}

RECENT PURCHASES (for context only):
{episodic_lines or "  No recent activity."}

BEST MATCHING PRODUCTS FROM H&M CATALOG:
{product_lines or "  No matching products found."}

YOUR RULES — follow strictly:
1. NEVER recommend anything above the budget ceiling stated above
2. NEVER use technical words like "MARA score", "decay", "lambda", "structural", "episodic", "embedding", or any numbers from scores
3. If the budget stored seems very low (under 20 SEK), refer to it as "your usual price range" instead
4. Recommend products naturally by name and color, as a human stylist would speak
5. If user asks for something outside budget, gently decline and offer a similar alternative that fits
6. Keep your reply to 2-4 sentences — warm, direct, and helpful
{f"IMPORTANT: This request conflicts with {', '.join(violations)} — redirect politely to an alternative." if violations else ""}
"""

        # 5. Get LLM response
        reply = call_groq(system_prompt, request.message)

        print(f"[{request.user_id}] Q: {request.message}")
        print(f"[{request.user_id}] A: {reply}\n")

        # Average MARA score per stratum — shows decay effect clearly
        def avg_score(mtype):
            scores = [m["mara_score"] for m in memories if m["type"] == mtype]
            return round(sum(scores) / len(scores), 4) if scores else 0.0

        return {
            "reply": reply,
            "meta": {
                "user_id": request.user_id,
                "constraint_violation": len(violations) > 0,
                "violations": violations,
                "budget_applied": constraints.get("budget"),
                "gender_index": constraints.get("gender_index"),
                "memories_used": {
                    "structural": len([m for m in memories if m["type"] == "structural"]),
                    "semantic":   len([m for m in memories if m["type"] == "semantic"]),
                    "episodic":   len([m for m in memories if m["type"] == "episodic"]),
                },
                "top_products": product_search["products"][:3],
                "memory_scores": {
                    "structural_avg": avg_score("structural"),
                    "semantic_avg":   avg_score("semantic"),
                    "episodic_avg":   avg_score("episodic"),
                },
                "top_memories": [
                    {"type": m["type"], "text": m["text"][:80], "score": m["mara_score"]}
                    for m in memories[:6]
                ],
            },
        }

    except Exception as e:
        print(f"Error in /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
async def evaluate(request: EvaluateRequest):
    """
    LLM-as-a-Judge: scores MARA vs Baseline on 4 dimensions.
    Returns scores 1-10 for each metric.
    """
    memories = retriever.get_contextual_memory(request.query, request.user_id)
    structural = [m for m in memories if m["type"] == "structural"]
    constraint_text = "\n".join([f"- {m['text']}" for m in structural])

    judge_prompt = f"""You are an objective evaluator for AI shopping assistants. 
Score each response 1-10 on four dimensions. Respond ONLY with valid JSON.

CUSTOMER CONSTRAINTS (ground truth):
{constraint_text or "No constraints stored"}

USER QUERY: {request.query}

SYSTEM A (MARA — Memory-Augmented):
{request.mara_response}

SYSTEM B (Baseline RAG):
{request.baseline_response}

Return exactly this JSON structure:
{{
  "mara": {{
    "memory_recall_accuracy": <1-10>,
    "numerical_stability_retention": <1-10>,
    "contextual_reasoning_coherence": <1-10>,
    "preference_consistency": <1-10>,
    "overall": <average>
  }},
  "baseline": {{
    "memory_recall_accuracy": <1-10>,
    "numerical_stability_retention": <1-10>,
    "contextual_reasoning_coherence": <1-10>,
    "preference_consistency": <1-10>,
    "overall": <average>
  }},
  "winner": "mara" | "baseline",
  "reasoning": "<one sentence>"
}}"""

    result_text = call_groq(judge_prompt, "Evaluate now.", temperature=0.1)

    import json
    try:
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        scores = json.loads(result_text)
    except:
        scores = {"raw": result_text, "parse_error": True}

    return {"evaluation": scores}


@app.get("/profile/{user_id}")
async def get_profile(user_id: str):
    """Returns the stored constraint profile for a user."""
    try:
        from qdrant_client.http import models as qmodels
        results = retriever.client.query_points(
            collection_name="structural_memory",
            query=retriever.model.encode("budget size preferences").tolist(),
            query_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(
                    key="user_id",
                    match=qmodels.MatchValue(value=user_id)
                )]
            ),
            limit=20,
        ).points

        constraints = [r.payload for r in results]
        return {
            "user_id": user_id,
            "structural_constraints": constraints,
            "total": len(constraints),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "MARA is running", "collections": ["hm_products", "structural_memory", "semantic_episodic_memory"]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)