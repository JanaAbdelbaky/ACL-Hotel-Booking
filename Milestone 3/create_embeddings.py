import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

# ----------------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "MS2_Hotel"  # <--- VERIFY YOUR PASSWORD

# Configuration for Embedding Models
MODELS = {
    "model_A": {
        "name": "all-MiniLM-L6-v2",
        "dim": 384,
        # Neo4j Config (Hotels)
        "hotel_index": "hotel_index_minilm",
        "property": "embedding_minilm",
        # FAISS Config (Visa)
        "visa_faiss_file": "visa_minilm.index",
        "visa_meta_file": "visa_minilm.pkl"
    },
    "model_B": {
        "name": "all-mpnet-base-v2",
        "dim": 768,
        # Neo4j Config (Hotels)
        "hotel_index": "hotel_index_mpnet",
        "property": "embedding_mpnet",
        # FAISS Config (Visa)
        "visa_faiss_file": "visa_mpnet.index",
        "visa_meta_file": "visa_mpnet.pkl"
    }
}

# ----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ----------------------------------------------------------------------------

def prepare_rich_hotel_vectors(hotels_df, reviews_df, users_df):
    """
    Merges Reviews and User Types into the Hotel text to create context-aware embeddings.
    """
    print(f"   -> Preparing text for {len(hotels_df)} Hotels...")
    
    # Merge Reviews with User Types
    if 'user_id' in reviews_df.columns and 'user_id' in users_df.columns:
        rich_reviews = pd.merge(reviews_df, users_df[['user_id', 'traveller_type']], on='user_id', how='left')
    else:
        rich_reviews = reviews_df.copy()
        rich_reviews['traveller_type'] = "Guest"

    data = []
    for _, row in hotels_df.iterrows():
        h_id = row['hotel_id']
        
        # Base Hotel Metadata
        text = f"Hotel: {row.get('hotel_name', 'Unknown')}. "
        text += f"Location: {row.get('city', '')}, {row.get('country', '')}. "
        text += f"Rating: {row.get('star_rating', 'N/A')} stars. "
        
        # Add Top 5 Reviews with Traveller Type
        if 'hotel_id' in rich_reviews.columns:
            hotel_revs = rich_reviews[rich_reviews['hotel_id'] == h_id].head(5)
            
            if not hotel_revs.empty:
                review_text = " Guest Experiences: "
                for _, rev in hotel_revs.iterrows():
                    t_type = rev.get('traveller_type', "Guest")
                    r_text = str(rev.get('review_text', '')).replace("\n", " ").strip()
                    review_text += f"[{t_type} Traveler]: {r_text} "
                text += review_text
            
        data.append({"id": h_id, "text": text})
    return data


def prepare_visa_vectors(visa_df):
    """
    Prepares the text string for Visa embedding.
    """
    print(f"   -> Preparing text for {len(visa_df)} Visa Rules...")
    data = []
    for idx, row in visa_df.iterrows():
        # Text representation of the rule
        text = f"Visa rule for travel from {row['from']} to {row['to']}. "
        text += f"Status: {row['requires_visa']}. Type: {row['visa_type']}."
        
        # Store metadata for FAISS retrieval
        data.append({
            "id": idx, 
            "text": text,
            "name": f"Visa: {row['from']}->{row['to']}",
            "info": row['visa_type']
        })
    return data


def upload_vectors_neo4j(session, label, id_prop, index_name, vec_property, dim, data_list, vectors):
    """
    Uploads vectors to EXISTING Neo4j nodes.
    """
    print(f"      -> Indexing {label} nodes into '{index_name}' (Neo4j)...")
    
    # 1. Create Vector Index (Only works if nodes exist)
    try:
        session.run(f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{label}) ON (n.{vec_property})
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {dim},
                `vector.similarity_function`: 'cosine'
            }}}}
        """)
    except Exception as e:
        print(f"      (Note: Index creation message: {e})")

    # 2. Upload Vectors
    print(f"      -> Uploading {len(data_list)} vectors to property '{vec_property}'...")
    count = 0
    for i, item in enumerate(data_list):
        session.run(f"""
            MATCH (n:{label} {{{id_prop}: $id}})
            CALL db.create.setVectorProperty(n, '{vec_property}', $vec)
            YIELD node RETURN count(node)
        """, {"id": item['id'], "vec": vectors[i].tolist()})
        count += 1
    print("      -> Neo4j Upload Complete.")


def save_vectors_faiss(data_list, vectors, index_file, meta_file):
    """
    Saves vectors to a local FAISS index file.
    """
    print(f"      -> Creating FAISS index: {index_file}...")
    
    # 1. Build Index
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype('float32'))
    
    # 2. Save Index
    faiss.write_index(index, index_file)
    
    # 3. Save Metadata
    print(f"      -> Saving metadata to {meta_file}...")
    # Convert list to dict keyed by index ID (0, 1, 2...)
    meta_store = {i: data_list[i] for i in range(len(data_list))}
    with open(meta_file, "wb") as f:
        pickle.dump(meta_store, f)
        
    print("      -> FAISS Save Complete.")

# ----------------------------------------------------------------------------
# 3. MAIN EXECUTION FLOW
# ----------------------------------------------------------------------------
def main():
    print("=========================================================")
    print("   STARTING HYBRID SETUP (HOTELS=NEO4J, VISA=FAISS)")
    print("=========================================================\n")

    # 1. Connect to Neo4j
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        driver.verify_connectivity()
        print("[1/4] Connected to Neo4j.")
    except Exception as e:
        print(f"ERROR: Could not connect to Neo4j. {e}")
        return

    # 2. Load CSV Data
    print("[2/4] Loading CSV Files...")
    try:
        hotels_df = pd.read_csv("hotels.csv")
        reviews_df = pd.read_csv("reviews.csv")
        users_df = pd.read_csv("users.csv")
        visa_df = pd.read_csv("visa.csv")
        print(f"      Loaded: Hotels({len(hotels_df)}), Reviews({len(reviews_df)}), Users({len(users_df)}), Visa({len(visa_df)})")
    except Exception as e:
        print(f"ERROR: Missing CSV files. {e}")
        return

    # 3. Prepare Text Data for Embeddings
    print("\n[3/4] Preparing Text Data for Vectorization...")
    hotel_data = prepare_rich_hotel_vectors(hotels_df, reviews_df, users_df)
    visa_data = prepare_visa_vectors(visa_df)

    # 4. Generate & Upload Embeddings
    print("\n[4/4] Generating and Uploading Embeddings...")
    
    for key, config in MODELS.items():
        print(f"\n   --- Processing Model: {config['name']} ---")
        model = SentenceTransformer(config['name'])
        
        # A. Process Hotels (NEO4J)
        print(f"   [Encoding] {len(hotel_data)} Hotels...")
        hotel_vecs = model.encode([d['text'] for d in hotel_data], show_progress_bar=True)
        
        with driver.session() as session:
            upload_vectors_neo4j(session, "Hotel", "hotel_id", 
                           config['hotel_index'], config['property'], config['dim'], 
                           hotel_data, hotel_vecs)
            
        # B. Process Visa (FAISS)
        print(f"   [Encoding] {len(visa_data)} Visa Rules...")
        visa_vecs = model.encode([d['text'] for d in visa_data], show_progress_bar=True)
        
        save_vectors_faiss(visa_data, visa_vecs, 
                           config['visa_faiss_file'], 
                           config['visa_meta_file'])

    driver.close()
    print("\n=========================================================")
    print("   SUCCESS! HYBRID DATABASE SETUP COMPLETE.")
    print("=========================================================")

if __name__ == "__main__":
    main()