# llm_engine.py
"""
Enhanced LLM wrapper for Component 3 with multi-LLM support.
- Converts Neo4j retrieval tables into human-readable context for LLM.
- Handles visa, hotel recommendations, and reviews intelligently.
- Supports CPU/GPU safely using HuggingFace Transformers.
- Allows dynamic switching between 3 free LLMs.
"""

from typing import List, Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os, textwrap
import torch

# -----------------------
# CONFIG: available free models
# -----------------------
AVAILABLE_LLMs = {
    "distilgpt2": "distilgpt2",         # small, CPU-friendly
    "GPT-Neo-1.3B": "EleutherAI/gpt-neo-1.3B",   # medium, GPU recommended
    "Mistral-7B": "mistralai/Mistral-7B-Instruct" # large, GPU required
}

# Default model from environment variable or fallback
MODEL_NAME = os.environ.get("LLM_MODEL", "distilgpt2")
_gen = None  # cached pipeline

# -----------------------
# LLM PIPELINE
# -----------------------
def _get_pipeline(model_name=None):
    """Returns a cached HuggingFace text-generation pipeline for the selected model."""
    global _gen
    model_name = model_name or MODEL_NAME

    if _gen is not None and _gen.model.name_or_path == model_name:
        return _gen

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )

        # Fix for meta tensors on large models
        if hasattr(model, "is_loaded") and not any(p.numel() > 0 for p in model.parameters()):
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": "cpu"})

        _gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        print(f"[LLM WARNING] Fallback to CPU pipeline: {e}")
        _gen = pipeline("text-generation", model_name, device=-1)

    return _gen

# -----------------------
# FORMAT CONTEXT FOR LLM
# -----------------------
def format_retrieval_results(context_items: List[Dict[str, Any]], intent: str) -> str:
    """Convert retrieval items into human-readable lines for LLM prompt."""
    lines = []
    if intent == "check_visa_requirement":
        for idx, item in enumerate(context_items):
            requirement = item.get("answer_text", item.get("requirement", "N/A"))
            lines.append(f"[SRC{idx}] Visa required from {item.get('from_country')} to {item.get('to_country')}: {requirement}")
    elif intent in ["find_hotel", "recommend_hotels_for_traveller_type", "hotels_by_value_for_money"]:
        for idx, item in enumerate(context_items[:5]):
            lines.append(f"[SRC{idx}] {item.get('hotel_name','N/A')}, {item.get('city','N/A')} — Rating: {item.get('star_rating','N/A')} stars")
    elif intent == "top_reviews_for_hotel":
        for idx, item in enumerate(context_items[:5]):
            review = item.get('review_text','').replace('\n',' ')[:200]
            lines.append(f"[SRC{idx}] Review (score={item.get('score_overall','N/A')}): {review}")
    else:
        for idx, item in enumerate(context_items):
            lines.append(f"[SRC{idx}] {str(item)}")
    return "\n".join(lines) if lines else "NO CONTEXT FOUND"

# -----------------------
# BUILD PROMPT
# -----------------------
def build_prompt(context_items: List[Dict[str, Any]], user_query: str, intent: str) -> str:
    """Creates LLM prompt with persona, context, and instructions."""
    persona = (
        "You are a helpful travel assistant. Use ONLY the facts from CONTEXT to answer. "
        "Keep the answer short and include a source tag for each fact."
    )
    context_text = format_retrieval_results(context_items, intent)

    prompt = textwrap.dedent(f"""
    {persona}

    CONTEXT:
    {context_text}

    USER QUESTION:
    {user_query}

    INSTRUCTIONS:
    1) Answer only using the CONTEXT. If the answer is not in the context say "I don't have enough information in the database to answer that."
    2) For each factual claim include its source tag in parentheses, e.g. (SRC0).
    3) Keep the answer concise, professional, and user-friendly.
    """).strip()
    return prompt

# -----------------------
# GENERATE ANSWER
# -----------------------
def generate_answer(context_items: List[Dict[str,Any]], user_query: str, intent: str, model_name:str=None, max_tokens:int=256) -> Dict[str,Any]:
    """
    Returns: { 'answer': str, 'prompt': str }
    Handles multiple LLMs, CPU/GPU safely, user-friendly answers.
    """
    prompt = build_prompt(context_items, user_query, intent)
    try:
        pipe = _get_pipeline(model_name)
        outputs = pipe(
            prompt,
            max_length=min(1024, max_tokens + len(prompt.split())),
            do_sample=True,
            top_p=0.9,
            temperature=0.6,
            num_return_sequences=1
        )
        text = outputs[0]["generated_text"]
        answer = text.split(prompt,1)[1].strip() if prompt in text else text.strip()
    except Exception as e:
        answer = f"[LLM ERROR] {e}"

    return {"answer": answer, "prompt": prompt}

######################################################
########################################################


###############################################################


# """
# component3_llm_api.py
# Milestone 3 – Component 3 (LLM Layer using ONLY HuggingFace Inference API)

# This file contains:
#     ✓ load_hf_api_client(model_key)
#     ✓ validate_models()
#     ✓ retry wrapper for API stability
#     ✓ merge_retrievals()
#     ✓ build_prompt()
#     ✓ answer_question()
#     ✓ evaluate_models()
#     ✓ Ready for Milestone 3 integration

# Requirements:
#     pip install huggingface_hub python-dotenv

# Config files required:
#     1) config_llm.txt (contains free HF model IDs)
#     2) .env (contains HF_TOKEN)

# Example config_llm.txt:
#     MODEL_1 = google/gemma-2-2b-it
#     MODEL_2 = Qwen/Qwen2.5-1.5B-Instruct
#     MODEL_3 = mistralai/Mistral-7B-Instruct-v0.3
# """

# import os
# import time
# import json
# import random
# from typing import Dict, Any, List, Optional
# from huggingface_hub import InferenceClient
# from dotenv import load_dotenv

# # ================================
# # 1. Load HF TOKEN
# # ================================
# load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")

# if not HF_TOKEN:
#     raise RuntimeError(
#         "❌ Missing HF_TOKEN.\nCreate a .env file with:\nHF_TOKEN=your_token_here\n"
#     )

# # ================================
# # 2. Read config_llm.txt
# # ================================
# def read_config_llm(path: str = "config_llm.txt") -> Dict[str, str]:
#     config_entries = {}
#     if not os.path.exists(path):
#         return config_entries

#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line or "=" not in line:
#                 continue
#             k, v = line.split("=", 1)
#             k = k.strip().lower().replace("model_", "")
#             config_entries[k] = v.strip()

#     return config_entries


# FREE_MODELS = read_config_llm("config_llm.txt")
# if not FREE_MODELS:
#     raise RuntimeError("❌ config_llm.txt is empty — add MODEL_1, MODEL_2, ...")


# # ================================
# # 3. Retry wrapper for API calls
# # ================================
# def retry_api_call(func, max_retries=3, delay=1.0):
#     for attempt in range(max_retries):
#         try:
#             return func()
#         except Exception as e:
#             if attempt == max_retries - 1:
#                 raise
#             time.sleep(delay + random.random())


# # ================================
# # 4. Validate model availability
# # ================================
# def validate_models():
#     """
#     Ensure each model in FREE_MODELS responds to a simple ping request.
#     """
#     print("\n🔍 Validating HuggingFace models...")
#     results = {}

#     for key, model_id in FREE_MODELS.items():
#         try:
#             client = InferenceClient(model=model_id, token=HF_TOKEN)
#             _ = retry_api_call(
#                 lambda: client.chat_completion(
#                     messages=[{"role": "user", "content": "ping"}],
#                     max_tokens=5,
#                 ),
#                 max_retries=2,
#             )
#             results[key] = True
#             print(f"  ✅ {key}: {model_id} is available")
#         except Exception as e:
#             results[key] = False
#             print(f"  ❌ {key}: {model_id} FAILED — {e}")

#     return results


# # ================================
# # 5. Load an inference client
# # ================================
# def load_hf_api_client(model_key: str):
#     if model_key not in FREE_MODELS:
#         raise ValueError(
#             f"Invalid model key '{model_key}'. Available keys: {list(FREE_MODELS.keys())}"
#         )
#     return InferenceClient(
#         model=FREE_MODELS[model_key],
#         token=HF_TOKEN
#     )


# # ================================
# # 6. merge_retrievals()
# # ================================
# def merge_retrievals(baseline, embedding, top_k=10):
#     seen = {}

#     def canonical_key(d):
#         for k in ("hotel_id", "hotel_name", "review_id", "from_country"):
#             if k in d:
#                 return str(d[k]).lower()
#         return json.dumps(d, sort_keys=True)

#     for r in baseline:
#         seen[canonical_key(r)] = {"item": r, "sources": {"baseline"}}

#     for r in embedding:
#         key = canonical_key(r)
#         if key in seen:
#             seen[key]["sources"].add("embedding")
#         else:
#             seen[key] = {"item": r, "sources": {"embedding"}}

#     merged = []
#     for v in seen.values():
#         item = v["item"]
#         item["_sources"] = list(v["sources"])
#         merged.append(item)

#     return {
#         "merged_list": merged[:top_k],
#         "counts": {"baseline": len(baseline), "embedding": len(embedding)}
#     }


# # ================================
# # 7. build_prompt()
# # ================================
# DEFAULT_PERSONA = "You are a helpful hotel, booking, and visa assistant."
# DEFAULT_TASK = "Answer using ONLY the context. If missing, say it is missing."

# def build_prompt(
#     user_question: str,
#     merged_context: Dict[str, Any],
#     intent: Optional[str] = None,                     # 🔧 FIX: intent-aware prompting
#     entities: Optional[Dict[str, Any]] = None,        # 🔧 FIX: entity grounding
#     requires_aggregation: bool = False,               # 🔧 FIX: aggregation handling
#     max_items: int = 8,
# ):
#     """
#     Milestone-3 structured prompt:
#         1) CONTEXT – KG facts (merged Cypher + embeddings)
#         2) PERSONA – single role for hotel + visa
#         3) TASK    – intent-aware grounding rule
#     """

#     # PERSONA
#     persona = (
#         "You are a helpful travel assistant specializing in hotels and visa information. "
#         "You ONLY answer based on verified facts from the knowledge graph. "
#         "You never invent information."
#     )

#     # 🔧 FIX: Dynamic task based on aggregation need
#     if requires_aggregation:
#         task = (
#             "You must ANALYZE and AGGREGATE the provided context facts to answer the question. "
#             "You may compute averages, counts, or grouped statistics ONLY using the context. "
#             "If required data is missing, say: "
#             "'The required information is not in the knowledge graph.'"
#         )
#     else:
#         task = (
#             "Answer the user's question strictly using ONLY the provided context facts. "
#             "If the context does not contain the answer, say: "
#             "'The required information is not in the knowledge graph.'"
#         )

#     # CONTEXT
#     items = merged_context.get("merged_list", [])[:max_items]
#     lines = []

#     if not items:
#         lines.append("No knowledge graph facts were found.")
#     else:
#         for i, obj in enumerate(items, 1):
#             cleaned = []
#             for key, value in obj.items():
#                 if not key.startswith("_"):
#                     cleaned.append(f"{key}: {value}")
#             lines.append(f"{i}. " + " | ".join(cleaned))

#     context_text = "\n".join(lines)

#     # 🔧 FIX: Explicit intent + entity hinting for small HF models
#     intent_hint = f"\nINTENT:\n{intent}\n" if intent else ""
#     entity_hint = f"\nENTITIES:\n{json.dumps(entities, indent=2)}\n" if entities else ""

#     prompt = f"""
# CONTEXT:
# {context_text}

# PERSONA:
# {persona}

# TASK:
# {task}
# {intent_hint}
# {entity_hint}
# QUESTION:
# {user_question}

# ANSWER:
# """

#     return prompt.strip()


# # ================================
# # 8. generate API answer
# # ================================
# def chat_api(client, prompt, max_new_tokens=256):
#     response = retry_api_call(
#         lambda: client.chat_completion(
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=max_new_tokens,
#             temperature=0.0
#         )
#     )
#     return response.choices[0].message["content"]


# # ================================
# # 9. answer_question()
# # ================================
# def answer_question(
#     question: str,
#     baseline: List[Dict],
#     embedding: List[Dict],
#     model_key: str,
#     intent: Optional[str] = None,                      # 🔧 FIX
#     entities: Optional[Dict[str, Any]] = None,         # 🔧 FIX
#     requires_aggregation: bool = False,                # 🔧 FIX
#     max_new_tokens=256
# ):
#     merged = merge_retrievals(baseline, embedding)

#     prompt = build_prompt(
#         user_question=question,
#         merged_context=merged,
#         intent=intent,                                 # 🔧 FIX
#         entities=entities,                             # 🔧 FIX
#         requires_aggregation=requires_aggregation      # 🔧 FIX
#     )

#     client = load_hf_api_client(model_key)

#     t0 = time.time()
#     answer = chat_api(client, prompt, max_new_tokens)
#     elapsed = time.time() - t0

#     return {
#         "model_key": model_key,
#         "prompt": prompt,
#         "answer": answer,
#         "time_s": elapsed,
#         "counts": merged["counts"],
#     }


# # ================================
# # 10. evaluate_models()
# # ================================
# def evaluate_models(models: List[str], testcases: List[Dict], save_to="llm_eval.json"):
#     results = {}

#     for m in models:
#         results[m] = {"tests": [], "summary": {}}
#         total_t = 0
#         correct = 0

#         for tc in testcases:
#             out = answer_question(
#                 question=tc["question"],
#                 baseline=tc["mock_baseline"],
#                 embedding=tc["mock_embedding"],
#                 model_key=m,
#             )
#             total_t += out["time_s"]

#             passed = tc["gold"].lower() in out["answer"].lower()
#             if passed:
#                 correct += 1

#             out["pass"] = passed
#             results[m]["tests"].append(out)

#         results[m]["summary"] = {
#             "accuracy_pct": 100 * correct / len(testcases),
#             "total_time_s": total_t,
#         }

#     with open(save_to, "w") as f:
#         json.dump(results, f, indent=2)

#     return results


# # ================================
# # 11. Demo block
# # ================================
# if __name__ == "__main__":
#     print("📌 Validating models...")
#     validate_models()

#     test_cases = [
#         {
#             "id": "Q1",
#             "question": "Do Egyptian citizens need a visa to France?",
#             "gold": "visa",
#             "mock_baseline": [{"from_country": "Egypt", "to_country": "France", "visa_type": "Schengen"}],
#             "mock_embedding": [{"similarity": 0.92, "visa_type": "Schengen"}],
#         },
#         {
#             "id": "Q2",
#             "question": "Recommend a family-friendly hotel in Paris.",
#             "gold": "family",
#             "mock_baseline": [{"hotel_name": "Novotel Paris", "star_rating": 4.2, "avg_review": 4.6}],
#             "mock_embedding": [{"hotel_name": "Novotel Paris", "similarity": 0.91, "avg_review": 4.6}],
#         },
#     ]

#     print("\n📌 Running evaluation...")
#     res = evaluate_models(list(FREE_MODELS.keys()), test_cases)
#     print(json.dumps(res, indent=2))

