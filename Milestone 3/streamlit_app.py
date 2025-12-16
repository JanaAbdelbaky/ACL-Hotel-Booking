# streamlit_app.py
"""
Polished Graph-RAG Travel Assistant UI
Milestone 3 – Component 4 (UI)

✔ Pastel professional theme (readable text)
✔ Switchable FREE HF LLMs (Gemma / Qwen / Mistral)
✔ Integrated with LLM_layer.py
✔ Supports visa, hotel recommendations, booking queries
✔ Clear pipeline: Query → KG Retrieval → LLM Answer
"""

import streamlit as st
import time
import pandas as pd
import preprocessing
import retriever
from LLM_layer import answer_question

# ======================================================
# Page Configuration
# ======================================================
st.set_page_config(
    page_title="Graph-RAG Travel Assistant",
    page_icon="✈️",
    layout="wide"
)

# ======================================================
# Custom Styling (Pastel + Navy, DARK TEXT)
# ======================================================
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: "Segoe UI", sans-serif;
        background-color: #f6f8fc;
        color: #0b1c2d;
    }

    h1, h2, h3, h4, h5, h6, label, p, span, div {
        color: #0b1c2d !important;
    }

    .block-container {
        padding-top: 2rem;
    }

    .card {
        background: #ffffff;
        padding: 1.6rem;
        border-radius: 16px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.06);
        margin-bottom: 1.4rem;
    }

    .subtle {
        color: #5f6f81 !important;
        font-size: 0.9rem;
    }

    .answer-box {
        background: linear-gradient(135deg, #e9f0ff, #ffffff);
        padding: 1.8rem;
        border-radius: 18px;
        border-left: 6px solid #1f3c88;
        font-size: 1.1rem;
        color: #0b1c2d !important;
    }

    .stButton>button {
        background-color: #1f3c88;
        color: #ffffff;
        border-radius: 12px;
        padding: 0.65rem 1.4rem;
        font-weight: 600;
        border: none;
    }

    .stButton>button:hover {
        background-color: #162a63;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================================================
# Sidebar – Controls
# ======================================================
st.sidebar.markdown("## ✈️ Graph-RAG Assistant")
st.sidebar.markdown(
    "<span class='subtle'>Hotels • Visas • Travel insights</span>",
    unsafe_allow_html=True
)

retrieval_strategy = st.sidebar.radio(
    "Retrieval Strategy",
    ["baseline", "embedding", "hybrid"],
    help="Hybrid combines Cypher + embeddings (recommended)"
)

embedding_model = st.sidebar.selectbox(
    "Embedding Model",
    ["model_A", "model_B"]
)

# 🔴 IMPORTANT FIX: UI LABEL ≠ BACKEND KEY
llm_choice = st.sidebar.selectbox(
    "LLM Model",
    options=[
        ("Gemma-2-2B (Fast)", "1"),
        ("Qwen-2.5-1.5B (Balanced)", "2"),
        ("Mistral-7B (Strong reasoning)", "3"),
    ],
    format_func=lambda x: x[0]
)
llm_model_key = llm_choice[1]  # MUST be "1", "2", or "3"

st.sidebar.info(
    "All models are **FREE** via HuggingFace Inference API.\n"
    "You can switch models live during evaluation."
)

# ======================================================
# Header
# ======================================================
st.markdown(
    """
    <div class="card">
        <h1>🌍 Graph-RAG Travel Assistant</h1>
        <p class="subtle">
            Knowledge-graph grounded hotel recommendations & visa answers
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ======================================================
# Query Input
# ======================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
query = st.text_area(
    "Ask your travel question",
    height=110,
    placeholder=(
        "e.g. Do I need a visa from Egypt to Italy?\n"
        "Recommend a family-friendly hotel in Paris"
    )
)
st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# Submit
# ======================================================
if st.button("✨ Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    start_time = time.time()

    # --------------------------------------------------
    # 1. Preprocessing
    # --------------------------------------------------
    with st.spinner("Understanding your request..."):
        processor = preprocessing.HotelInputPreprocessor(
            "hotels.csv",
            "users.csv"
        )
        parsed = processor.process(query, embedding_model=embedding_model)

    intents = parsed.get("intents", [])
    if isinstance(intents, str):
        intents = [intents]

    entities = parsed.get("entities", {})
    embedding_vec = parsed.get("embedding", None)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Query Understanding")
    st.markdown(f"**Detected intents:** {intents}")
    st.json(entities)
    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # 2. Retrieval (handle multi-intent safely)
    # --------------------------------------------------
    with st.spinner("Retrieving knowledge graph facts..."):

        retrieval_out = retriever.run_retrieval(
            intents_input=intents,
            entities=entities,
            embedding_vector=embedding_vec,
            model_key=embedding_model,
            strategy=retrieval_strategy if retrieval_strategy != "hybrid" else "baseline"
        )

        baseline_dict = retrieval_out.get("baseline", {})
        baseline = []
        for intent_results in baseline_dict.values():
            baseline.extend(intent_results)

        embedding = []
        if retrieval_strategy in ("embedding", "hybrid"):
            embedding_dict = retriever.run_retrieval(
                intents_input=intents,
                entities=entities,
                embedding_vector=embedding_vec,
                model_key=embedding_model,
                strategy="embedding"
            ).get("embedding", {})

            # Flatten embedding results into a single list
            for k, v in embedding_dict.items():
                if isinstance(v, list):
                    embedding.extend(v)


    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Retrieved Context")

    if baseline or embedding:
        df = pd.DataFrame((baseline + embedding)[:20])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No facts retrieved from the knowledge graph.")

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # 3. LLM Answer (LLM_layer.py)
    # --------------------------------------------------
    with st.spinner("Generating grounded answer..."):
        out = answer_question(
            question=query,
            baseline=baseline,
            embedding=embedding,
            model_key=llm_model_key
        )

    # --------------------------------------------------
    # 4. Final Answer
    # --------------------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### ✅ Final Answer")
    st.markdown(
        f"<div class='answer-box'>{out['answer']}</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🧠 Prompt sent to LLM (debug)"):
        st.code(out["prompt"])

    elapsed = time.time() - start_time
    st.markdown(
        f"<p class='subtle'>⏱ Completed in {elapsed:.2f}s</p>",
        unsafe_allow_html=True
    )

# ======================================================
# Sidebar – Examples & Notes
# ======================================================

# with st.sidebar:
#     st.markdown("---")
#     st.markdown("### 💡 Try asking")
#     st.markdown("- Do I need a visa from Egypt to Italy?")
#     st.markdown("- Recommend a romantic hotel in Paris")
#     st.markdown("- Best hotels for families in Rome")

#     st.markdown("---")

# ======================================================
# Sidebar – Examples, Notes & Model Evaluation
# ======================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("### 💡 Try asking")
    st.markdown("- Do I need a visa from Egypt to Italy?")
    st.markdown("- Recommend a romantic hotel in Paris")
    st.markdown("- Best hotels for families in Rome")
    st.markdown("---")

    # --------------------------------------------------
    # LLM Evaluation Section
    # --------------------------------------------------
    st.markdown("### 🧪 Evaluate LLM Models")
    if st.button("Run Model Evaluation"):
        import json
        from LLM_layer import evaluate_models, FREE_MODELS

        with st.spinner("Running evaluation on test cases..."):

            # Define small set of example test cases
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

            results = evaluate_models(list(FREE_MODELS.keys()), hotel_test_cases)

        st.success("✅ Evaluation Completed")

        # Display summary metrics
        st.markdown("#### Summary Metrics per Model")
        for model_key, res in results.items():
            summary = res.get("summary", {})
            st.markdown(f"**{model_key}**")
            st.markdown(f"- Accuracy: {summary.get('accuracy_pct', 0):.1f}%")
            st.markdown(f"- Total Time: {summary.get('total_time_s', 0):.2f}s")
            st.markdown("---")

        # Optional: Show individual test outputs
        with st.expander("Show Detailed Test Outputs"):
            for model_key, res in results.items():
                st.markdown(f"**Model {model_key} Detailed Outputs**")
                for t in res.get("tests", []):
                    st.markdown(f"- Question: {t.get('prompt', '').split('QUESTION:')[-1].strip().split('ANSWER:')[0].strip()}")
                    st.markdown(f"  - Answer: {t.get('answer', '')}")
                    st.markdown(f"  - Passed: {t.get('pass', False)}")
                    st.markdown(f"  - Time: {t.get('time_s', 0):.2f}s")
                    st.markdown("---")

