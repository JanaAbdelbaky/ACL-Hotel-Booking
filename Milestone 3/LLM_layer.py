"""
component3_llm_api.py
Milestone 3 – Component 3 (LLM Layer using ONLY HuggingFace Inference API)

This file contains:
    ✓ load_hf_api_client(model_key)
    ✓ validate_models()
    ✓ retry wrapper for API stability
    ✓ merge_retrievals()
    ✓ build_prompt()
    ✓ answer_question()
    ✓ evaluate_models()
    ✓ Ready for Milestone 3 integration

Requirements:
    pip install huggingface_hub python-dotenv

Config files required:
    1) config_llm.txt (contains free HF model IDs)
    2) .env (contains HF_TOKEN)

Example config_llm.txt:
    MODEL_1 = google/gemma-2-2b-it
    MODEL_2 = Qwen/Qwen2.5-1.5B-Instruct
    MODEL_3 = mistralai/Mistral-7B-Instruct-v0.3
"""

import os
import time
import json
import random
from typing import Dict, Any, List, Optional
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# ================================
# 1. Load HF TOKEN
# ================================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "❌ Missing HF_TOKEN.\nCreate a .env file with:\nHF_TOKEN=your_token_here\n"
    )

# ================================
# 2. Read config_llm.txt
# ================================
def read_config_llm(path: str = "config_llm.txt") -> Dict[str, str]:
    config_entries = {}
    if not os.path.exists(path):
        return config_entries

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().lower().replace("model_", "")
            config_entries[k] = v.strip()

    return config_entries


FREE_MODELS = read_config_llm("config_llm.txt")
if not FREE_MODELS:
    raise RuntimeError("❌ config_llm.txt is empty — add MODEL_1, MODEL_2, ...")


# ================================
# 3. Retry wrapper for API calls
# ================================
def retry_api_call(func, max_retries=3, delay=1.0):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay + random.random())


# ================================
# 4. Validate model availability
# ================================
def validate_models():
    """
    Ensure each model in FREE_MODELS responds to a simple ping request.
    """
    print("\n🔍 Validating HuggingFace models...")
    results = {}

    for key, model_id in FREE_MODELS.items():
        try:
            client = InferenceClient(model=model_id, token=HF_TOKEN)
            _ = retry_api_call(
                lambda: client.chat_completion(
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5,
                ),
                max_retries=2,
            )
            results[key] = True
            print(f"  ✅ {key}: {model_id} is available")
        except Exception as e:
            results[key] = False
            print(f"  ❌ {key}: {model_id} FAILED — {e}")

    return results


# ================================
# 5. Load an inference client
# ================================
def load_hf_api_client(model_key: str):
    if model_key not in FREE_MODELS:
        raise ValueError(
            f"Invalid model key '{model_key}'. Available keys: {list(FREE_MODELS.keys())}"
        )
    return InferenceClient(
        model=FREE_MODELS[model_key],
        token=HF_TOKEN
    )


# ================================
# 6. merge_retrievals()
# ================================
def merge_retrievals(baseline, embedding, top_k=10):
    seen = {}

    def canonical_key(d):
        for k in ("hotel_id", "hotel_name", "review_id", "from_country"):
            if k in d:
                return str(d[k]).lower()
        return json.dumps(d, sort_keys=True)

    for r in baseline:
        seen[canonical_key(r)] = {"item": r, "sources": {"baseline"}}

    for r in embedding:
        key = canonical_key(r)
        if key in seen:
            seen[key]["sources"].add("embedding")
        else:
            seen[key] = {"item": r, "sources": {"embedding"}}

    merged = []
    for v in seen.values():
        item = v["item"]
        item["_sources"] = list(v["sources"])
        merged.append(item)

    return {"merged_list": merged[:top_k], "counts": {"baseline": len(baseline), "embedding": len(embedding)}}


# ================================
# 7. build_prompt()
# ================================
DEFAULT_PERSONA = "You are a helpful hotel, booking, and visa assistant."
DEFAULT_TASK = "Answer using ONLY the context. If missing, say it is missing."

# def build_prompt(question, merged, persona=DEFAULT_PERSONA, task=DEFAULT_TASK):
#     lines = []
#     for i, it in enumerate(merged["merged_list"], 1):
#         parts = []
#         for k, v in it.items():
#             if not k.startswith("_"):
#                 parts.append(f"{k}: {v}")
#         lines.append(f"{i}. " + " | ".join(parts))

#     return f"""
# PERSONA:
# {persona}

# TASK:
# {task}

# CONTEXT:
# {chr(10).join(lines)}

# QUESTION:
# {question}

# ANSWER:
# """

def build_prompt(
    user_question: str,
    merged_context: Dict[str, Any],
    max_items: int = 8,
):
    """
    Milestone‑3 structured prompt:
        1) CONTEXT – KG facts (merged Cypher + embeddings)
        2) PERSONA – single role for hotel + visa
        3) TASK    – strict grounding rule
    """

    # PERSONA/ROLE defination
    persona = (
        "You are a helpful travel assistant specializing in hotels and visa information. "
        "You ONLY answer based on verified facts from the knowledge graph. "
        "You never invent information."
        "You never search the internet or Google for information or recommendations."

    )

    # TASK instructions 
    task = (
        "Answer the user's question strictly using ONLY the provided context facts. "
        "If the context does not contain the answer, say: "
        "'The required information is not in the knowledge graph.'"
    )

    # CONTEXT of baseline + embeddings (from merge_retreival)
    items = merged_context.get("merged_list", [])[:max_items]
    lines = []

    if not items:
        lines.append("No knowledge graph facts were found.")
    else:
        for i, obj in enumerate(items, 1):
            cleaned = []
            for key, value in obj.items():
                if not key.startswith("_"):
                    cleaned.append(f"{key}: {value}")
            lines.append(f"{i}. " + " | ".join(cleaned))

    context_text = "\n".join(lines)

    prompt = f"""
CONTEXT:
{context_text}

PERSONA:
{persona}

TASK:
{task}

QUESTION:
{user_question}

ANSWER:
"""

    return prompt.strip()


# ================================
# 8. generate API answer
# ================================
def chat_api(client, prompt, max_new_tokens=256):
    response = retry_api_call(
        lambda: client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.0
        )
    )
    return response.choices[0].message["content"]


# ================================
# 9. answer_question()
# ================================
def answer_question(
    question: str,
    baseline: List[Dict],
    embedding: List[Dict],
    model_key: str,
    max_new_tokens=256
):
    merged = merge_retrievals(baseline, embedding)
    prompt = build_prompt(question, merged)
    client = load_hf_api_client(model_key)

    t0 = time.time()
    answer = chat_api(client, prompt, max_new_tokens)
    elapsed = time.time() - t0

    return {
        "model_key": model_key,
        "prompt": prompt,
        "answer": answer,
        "time_s": elapsed,
        "counts": merged["counts"],
    }


# ================================
# 10. evaluate_models()
# ================================
def evaluate_models(models: List[str], testcases: List[Dict], save_to="llm_eval.json"):
    results = {}

    for m in models:
        results[m] = {"tests": [], "summary": {}}
        total_t = 0
        correct = 0

        for tc in testcases:
            out = answer_question(
                question=tc["question"],
                baseline=tc["mock_baseline"],
                embedding=tc["mock_embedding"],
                model_key=m,
            )
            total_t += out["time_s"]

            passed = tc["gold"].lower() in out["answer"].lower()
            if passed:
                correct += 1

            out["pass"] = passed
            results[m]["tests"].append(out)

        results[m]["summary"] = {
            "accuracy_pct": 100 * correct / len(testcases),
            "total_time_s": total_t,
        }

    with open(save_to, "w") as f:
        json.dump(results, f, indent=2)

    return results


# ================================
# 11. Demo block
# ================================
if __name__ == "__main__":
    print("📌 Validating models...")
    validate_models()

    # test_cases = [
    #     {
    #         "id": "Q1",
    #         "question": "Do Egyptian citizens need a visa to France?",
    #         "gold": "visa",
    #         "mock_baseline": [{"from_country": "Egypt", "to_country": "France", "visa_type": "Schengen"}],
    #         "mock_embedding": [{"similarity": 0.92, "visa_type": "Schengen"}],
    #     },
    #     {
    #         "id": "Q2",
    #         "question": "Recommend a family-friendly hotel in Paris.",
    #         "gold": "family",
    #         "mock_baseline": [{"hotel_name": "Novotel Paris", "star_rating": 4.2, "avg_review": 4.6}],
    #         "mock_embedding": [{"hotel_name": "Novotel Paris", "similarity": 0.91, "avg_review": 4.6}],
    #     },
    # ]


    hotel_test_cases = [
    {
        "id": "H1",
        "question": "Recommend a luxury 5-star hotel in Paris.",
        "gold": "l'étoile palace",
        "mock_baseline": [{
            "hotel_name": "L'Étoile Palace",
            "city": "Paris",
            "star_rating": 5,
            "luxury": "yes"
        }],
        "mock_embedding": [{
            "hotel_name": "L'Étoile Palace",
            "similarity": 0.93
        }]
    },
    {
        "id": "H2",
        "question": "Which hotel in Tokyo has excellent cleanliness?",
        "gold": "kyo-to grand",
        "mock_baseline": [{
            "hotel_name": "Kyo-to Grand",
            "city": "Tokyo",
            "cleanliness": "excellent"
        }],
        "mock_embedding": [{
            "hotel_name": "Kyo-to Grand",
            "similarity": 0.91
        }]
    },
    {
        "id": "H3",
        "question": "Suggest a hotel with the best location in Rome.",
        "gold": "colosseum gardens",
        "mock_baseline": [{
            "hotel_name": "Colosseum Gardens",
            "city": "Rome",
            "location": "excellent"
        }],
        "mock_embedding": [{
            "hotel_name": "Colosseum Gardens",
            "similarity": 0.94
        }]
    },
    {
        "id": "H4",
        "question": "Find a top-rated hotel in Cairo.",
        "gold": "nile grandeur",
        "mock_baseline": [{
            "hotel_name": "Nile Grandeur",
            "city": "Cairo",
            "star_rating": 5
        }],
        "mock_embedding": [{
            "hotel_name": "Nile Grandeur",
            "similarity": 0.89
        }]
    },
    {
        "id": "H5",
        "question": "Which hotel in Amsterdam offers high comfort?",
        "gold": "canal house grand",
        "mock_baseline": [{
            "hotel_name": "Canal House Grand",
            "city": "Amsterdam",
            "comfort": "high"
        }],
        "mock_embedding": [{
            "hotel_name": "Canal House Grand",
            "similarity": 0.92
        }]
    },
]
#     visa_test_cases = [
#     {
#         "id": "V1",
#         "question": "Do United States citizens need a visa to travel to France?",
#         "gold": "no",
#         "mock_baseline": [{
#             "from_country": "United States",
#             "to_country": "France",
#             "requires_visa": "No"
#         }],
#         "mock_embedding": [{
#             "similarity": 0.95
#         }]
#     },
#     {
#         "id": "V2",
#         "question": "Is a visa required for UK citizens visiting Japan?",
#         "gold": "no",
#         "mock_baseline": [{
#             "from_country": "United Kingdom",
#             "to_country": "Japan",
#             "requires_visa": "No"
#         }],
#         "mock_embedding": [{
#             "similarity": 0.93
#         }]
#     },
#     {
#         "id": "V3",
#         "question": "Do French citizens need a visa to enter Egypt?",
#         "gold": "no",
#         "mock_baseline": [{
#             "from_country": "France",
#             "to_country": "Egypt",
#             "requires_visa": "No"
#         }],
#         "mock_embedding": [{
#             "similarity": 0.90
#         }]
#     },
#     {
#         "id": "V4",
#         "question": "Can Japanese citizens travel to Germany without a visa?",
#         "gold": "no",
#         "mock_baseline": [{
#             "from_country": "Japan",
#             "to_country": "Germany",
#             "requires_visa": "No"
#         }],
#         "mock_embedding": [{
#             "similarity": 0.92
#         }]
#     },
# ]
    
#     test_cases = hotel_test_cases + visa_test_cases


   
    print("\n📌 Running evaluation...")
    res = evaluate_models(list(FREE_MODELS.keys()), hotel_test_cases)
    print(json.dumps(res, indent=2))
