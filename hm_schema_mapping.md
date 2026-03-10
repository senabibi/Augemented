# MARA × H&M Dataset Schema Mapping

## H&M Dataset Files → MARA Memory Architecture

### articles.csv → Product Catalog (Qdrant Collection: hm_products)
| H&M Field | MARA Role | Memory Type |
|---|---|---|
| article_id | unique product key | — |
| prod_name | product name | — |
| product_type_name | garment category | 🎨 Semantic |
| colour_group_name | color constraint | 🏗 Structural (if hard preference) |
| perceived_colour_master_name | color family | 🎨 Semantic |
| index_name | gender/section | 🏗 Structural (Ladieswear/Menswear) |
| garment_group_name | garment type | 🎨 Semantic |
| detail_desc | rich text for embedding | BGE embedding input |
| price (from transactions) | budget constraint | 🏗 Structural |

### customers.csv → Customer Constraints (Qdrant: structural_memory)
| H&M Field | MARA Role | Memory Type |
|---|---|---|
| customer_id | user_id | — |
| age | age group filter | 🏗 Structural (λ≈0.01) |
| club_member_status | loyalty tier | 🎨 Semantic |
| fashion_news_frequency | trend-interest | ⚡ Episodic |

### transactions_train.csv → Interaction History (Qdrant: semantic_episodic_memory)
| H&M Field | MARA Role | Memory Type |
|---|---|---|
| t_dat (date) | timestamp for decay | decay_factor(t) |
| price | avg spend → budget model | 🏗 Structural |
| sales_channel_id | channel pref | 🎨 Semantic |
| article_id (joined) | browsed item | ⚡ Episodic (recent) / 🎨 Semantic (repeated) |

---

## MARA Memory Extraction Logic from H&M Transactions

```
For each customer:
  1. STRUCTURAL constraints:
     - budget = percentile_25(transaction prices) → "never exceeds X"
     - gender_index = mode(index_name) → Ladieswear / Menswear / etc.
     - age_group = customers.age bucketed

  2. SEMANTIC preferences (repeated behavior = stable taste):
     - top 3 colour_group_name by frequency
     - top 3 garment_group_name by frequency  
     - top product_type_name (>= 3 purchases)

  3. EPISODIC signals (last 30 days = volatile):
     - last 5 purchased article descriptions
     - recent price range (last 10 transactions)
     - recent colour / style deviations from semantic baseline
```

---

## 6-Month MARA Simulation with H&M Data

We pick a real customer from transactions who:
- Has 2+ years of history (2018-2020)
- Shows seasonal variation (ideal for constraint drift demo)
- Has a clear price ceiling pattern

### Demo Customer Journey:
```
Month 1 (Sep 2018): Establishes baseline
  → Structural: budget≈179 SEK, Ladieswear, size implicit
  → Semantic: Jersey Basic, Black/Dark, everyday basics

Month 3 (Dec 2018 - Holiday season):
  → Episodic spike: Evening wear, high price (holiday shopping)
  → Structural PRESERVED: budget ceiling intact despite holiday browsing

Month 6 (Mar 2019 - Spring):
  → Query: "Suggest something for spring"
  → Baseline RAG: recommends 350 SEK party dress (episodic noise wins)
  → MARA: recommends 160 SEK cotton blouse, light colors (structural budget + semantic basics preserved)
```
