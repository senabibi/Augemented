"""
fix_indexes.py
==============
Adds ALL required payload indexes to Qdrant collections.
Safe to run multiple times — skips already-existing indexes.

Usage:
    python3 fix_indexes.py
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60,
)

# Every field used in a Qdrant filter() call needs a payload index
INDEXES = {
    "structural_memory": [
        ("user_id",           qmodels.PayloadSchemaType.KEYWORD),
        ("constraint_type",   qmodels.PayloadSchemaType.KEYWORD),
        ("decay_class",       qmodels.PayloadSchemaType.KEYWORD),
    ],
    "semantic_episodic_memory": [
        ("user_id",           qmodels.PayloadSchemaType.KEYWORD),
        ("memory_type",       qmodels.PayloadSchemaType.KEYWORD),
        ("decay_class",       qmodels.PayloadSchemaType.KEYWORD),
    ],
    "hm_products": [
        ("index_name",        qmodels.PayloadSchemaType.KEYWORD),
        ("colour_group_name", qmodels.PayloadSchemaType.KEYWORD),
        ("garment_group_name",qmodels.PayloadSchemaType.KEYWORD),
    ],
}

print("=" * 50)
print("  MARA - Adding payload indexes")
print("=" * 50)

for collection, fields in INDEXES.items():
    print(f"\n[{collection}]")
    for field_name, field_type in fields:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=field_type,
            )
            print(f"  OK  {field_name}")
        except Exception as e:
            msg = str(e).lower()
            if "already exists" in msg or "conflict" in msg or "400" in msg:
                print(f"  --  {field_name} (already exists, skipped)")
            else:
                print(f"  ERR {field_name}: {e}")

print("\nAll indexes applied. Retry your curl request.")