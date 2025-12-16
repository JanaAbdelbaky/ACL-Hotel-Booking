# import preprocessing
# import retriever
# import json

# def run_full_pipeline(user_query):
#     """
#     Orchestrates the flow from User Input -> Preprocessing -> Retrieval -> Result
#     """
#     print(f"\n{'='*80}")
#     print(f"USER QUERY: '{user_query}'")
#     print(f"{'='*80}")

#     # ---------------------------------------------------------
#     # STEP 1: PREPROCESSING (Component 1)
#     # ---------------------------------------------------------
#     print("\n[1] Running Preprocessing...")
    
#     # Initialize the processor (make sure CSVs are in the folder)
#     processor = preprocessing.HotelInputPreprocessor("hotels.csv", "users.csv")
    
#     # Process the text
#     pre_result = processor.process(user_query)
    
#     intent = pre_result["intent"]
#     entities = pre_result["entities"]
    
#     print(f"    -> Identified Intent: '{intent}'")
#     print(f"    -> Extracted Entities: {json.dumps(entities, default=str)}")

#     # ---------------------------------------------------------
#     # STEP 2: GRAPH RETRIEVAL (Component 2)
#     # ---------------------------------------------------------
#     print("\n[2] Running Graph Retrieval...")
    
#     # The retriever takes the intent and entities directly now
#     retrieval_result = retriever.run_retrieval(intent, entities)
    
#     # Check for errors
#     if "error" in retrieval_result:
#         print(f"    -> ERROR: {retrieval_result['error']}")
#         return

#     # ---------------------------------------------------------
#     # STEP 3: DISPLAY RESULTS
#     # ---------------------------------------------------------
#     records = retrieval_result["baseline_results"]
#     print(f"    -> Database returned {len(records)} records.")
    
#     if records:
#         print("\n    [Top 3 Results]:")
#         for i, rec in enumerate(records[:3]):
#             print(f"      {i+1}. {rec}")
#     else:
#         print("      (No matching records found in the database)")

# # ---------------------------------------------------------
# # EXECUTION LOOP
# # ---------------------------------------------------------
# if __name__ == "__main__":
#     # Ensure Neo4j is running before starting!
    
#     test_questions = [
#         "Find me a clean hotel in London.",           
#         "Do I need a visa for France?",               
#         "Show me hotels for a family trip.",          
#         "What are the top reviews for The Azure Tower?", 
#         "Find hotels in Cairo.",
#         "I need a cheap hotel in Paris."
#     ]

#     print("Starting System Test...")
    
#     for question in test_questions:
#         try:
#             run_full_pipeline(question)
#         except Exception as e:
#             print(f"CRITICAL FAILURE: {e}")
            
#     # Clean up connection
#     retriever.close()

import preprocessing
import retriever
import json
import time

def run_full_system_test():
    print(f"{'='*80}\n{'MILESTONE 3: FULL SYSTEM TEST':^80}\n{'='*80}\n")

    # 1. INITIALIZE
    print("[INIT] Initializing Preprocessor...")
    try:
        processor = preprocessing.HotelInputPreprocessor("hotels.csv", "users.csv")
        print("[INIT] Success.\n")
    except Exception as e:
        print(f"[INIT] Failed: {e}"); return

    # 2. TEST SCENARIOS
    test_scenarios = [
        {"category": "BASELINE: City Search", "query": "Find hotels in London.", "strategy": "baseline"},
        {"category": "BASELINE: Visa Check", "query": "Do I need a visa for France?", "strategy": "baseline"},
        {"category": "BASELINE: Family Recommendation", "query": "Show me hotels for a family trip.", "strategy": "baseline"},
        {"category": "EXPERIMENT B: Model A (MiniLM)", "query": "Find hotels in London.", "strategy": "embedding", "model": "model_A"},
        {"category": "EXPERIMENT B: Model B (MPNet)", "query": "Find hotels in London.", "strategy": "embedding", "model": "model_B"}
    ]

    # 3. EXECUTE
    for i, test in enumerate(test_scenarios):
        print(f"{'-'*80}\nTEST #{i+1}: {test['category']}\nQUERY:   '{test['query']}'\n{'-'*80}")
        start = time.time()
        
        # Preprocessing
        target_model = test.get("model", "model_A") 
        pre_result = processor.process(test['query'], embedding_model=target_model)
        
        print(f"\n[COMPONENT 1] Preprocessing:")
        print(f"   -> Intent: {pre_result['intent']}")
        print(f"   -> Entities: {json.dumps(pre_result['entities'], default=str)}")

        # Retrieval
        strategy = test.get("strategy", "baseline")
        print(f"\n[COMPONENT 2] Retrieval ({strategy.upper()}):")
        
        res_dict = retriever.run_retrieval(
            pre_result['intent'], pre_result['entities'], 
            embedding_vector=pre_result['embedding'], model_key=target_model, strategy=strategy
        )
        results = res_dict.get(strategy, [])

        if not results:
            print("   -> No results found.")
            if "error" in res_dict: print(f"   -> ERROR: {res_dict.get('error')}")
        else:
            if "error" in results[0]:
                print(f"   -> ERROR: {results[0]['error']}")
            else:
                print(f"   -> Found {len(results)} matches. Top 3:")
                for j, rec in enumerate(results[:3]):
                    # DYNAMIC PRINTING based on available keys
                    output_str = ""
                    
                    # Case 1: Hotel Result
                    if "hotel_name" in rec:
                        name = rec["hotel_name"]
                        if strategy == "embedding":
                            meta = f"Similarity: {rec.get('similarity_score', 0):.4f}"
                        else:
                            # Try to find a score or rating
                            meta = f"Rating: {rec.get('star_rating', 'N/A')}"
                            if "score" in rec: meta = f"Score: {rec['score']:.2f}"
                        output_str = f"{name} | {meta}"
                    
                    # Case 2: Visa Result
                    elif "requirement" in rec:
                        output_str = f"{rec.get('from_country')} -> {rec.get('to_country')}: {rec.get('requirement')} ({rec.get('visa_type')})"
                    
                    # Case 3: Review Result
                    elif "review_text" in rec:
                        output_str = f"Review (Score {rec.get('score_overall')}): {rec.get('review_text')[:50]}..."

                    else:
                        output_str = str(rec) # Fallback

                    print(f"      {j+1}. {output_str}")

        print(f"\n[PERFORMANCE] Time: {time.time() - start:.2f}s\n")

    retriever.close()

if __name__ == "__main__":
    run_full_system_test()