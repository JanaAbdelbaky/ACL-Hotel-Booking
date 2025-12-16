import faiss
import pickle
import numpy as np
from neo4j import GraphDatabase

# ----------------------------------------------------------------------------
# 1. CONFIGURATION & CONNECTION
# ----------------------------------------------------------------------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "MS2_Hotel"  # <--- VERIFY THIS

MODELS = {
    "model_A": {
        "neo4j_index": "hotel_index_minilm",
        "top_k": 5,
        "faiss_file": "visa_minilm.index",
        "meta_file": "visa_minilm.pkl"
    },
    "model_B": {
        "neo4j_index": "hotel_index_mpnet",
        "top_k": 5,
        "faiss_file": "visa_mpnet.index",
        "meta_file": "visa_mpnet.pkl"
    }
}

# ----------------------------------------------------------------------------
# 2. HYBRID RETRIEVER CLASS
# ----------------------------------------------------------------------------
class HybridRetriever:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        self.faiss_indices = {}
        self.faiss_meta = {}
        
        print("Loading FAISS indices for Visa...")
        for key, cfg in MODELS.items():
            try:
                self.faiss_indices[key] = faiss.read_index(cfg['faiss_file'])
                with open(cfg['meta_file'], "rb") as f:
                    self.faiss_meta[key] = pickle.load(f)
            except Exception: pass

    def close(self):
        self.driver.close()

    def search_neo4j_hotels(self, vector, model_key, country_filter=None):
        """Search Hotels in Neo4j with Country Filtering and return all Scores"""
        cfg = MODELS[model_key]
        
        # Build Where Clause dynamically based on filter type
        where_clause = "WHERE ($country IS NULL OR toLower(co.name) CONTAINS toLower($country))"
        params = {
            "idx": cfg['neo4j_index'], 
            "k": cfg['top_k'] * 2, 
            "vec": vector,
            "country": country_filter
        }

        # If filtering by a list (e.g., Europe -> [France, Germany...])
        if isinstance(country_filter, list):
            where_clause = "WHERE co.name IN $country_list"
            params = {
                "idx": cfg['neo4j_index'], 
                "k": cfg['top_k'] * 2, 
                "vec": vector,
                "country_list": country_filter
            }

        query = f"""
        CALL db.index.vector.queryNodes($idx, $k, $vec) 
        YIELD node, score 
        MATCH (node)-[:LOCATED_IN]->(:City)-[:LOCATED_IN]->(co:Country)
        {where_clause}
        RETURN node.hotel_name AS name, 
               node.star_rating AS info, 
               co.name as country, 
               score,
               node.cleanliness_base AS cleanliness,
               node.comfort_base AS comfort,
               node.value_for_money_base AS value,
               node.location_base AS location,
               node.staff_base AS staff
        """
        try:
            with self.driver.session() as s:
                return [dict(r) for r in s.run(query, params)]
        except Exception as e:
            return [{"error": str(e)}]

    def search_faiss_visa(self, vector, model_key):
        """Search Visa in FAISS"""
        if model_key not in self.faiss_indices: return []
        index = self.faiss_indices[model_key]
        meta = self.faiss_meta[model_key]
        
        vec_np = np.array([vector]).astype('float32')
        D, I = index.search(vec_np, MODELS[model_key]['top_k'])
        
        results = []
        for i, idx_id in enumerate(I[0]):
            if idx_id != -1:
                item = meta[idx_id]
                results.append({
                    "name": item['name'],
                    "info": item['info'],
                    "score": float(D[0][i])
                })
        return results

    def run_cypher(self, query, params):
        with self.driver.session() as s:
            return [dict(r) for r in s.run(query, params)]

retriever_instance = HybridRetriever()

# ----------------------------------------------------------------------------
# 3. CYPHER TEMPLATES (ROBUST MATCHING)
# ----------------------------------------------------------------------------
CYPHER_TEMPLATES = {
    # --- VISA ---
    "check_visa_requirement": {
        "cypher": """
        MATCH (from:Country) WHERE toLower(from.name) = toLower($from_country)
        MATCH (to:Country) WHERE toLower(to.name) = toLower($to_country)
        OPTIONAL MATCH (from)-[v:NEEDS_VISA]->(to)
        RETURN from.name AS from_country, to.name AS to_country, 
               CASE WHEN v IS NULL THEN 'No visa required' ELSE 'Visa required' END AS requirement, 
               v.visa_type AS visa_type
        """,
        "required": ["from_country", "to_country"]
    },
    "get_visa_type": {
        "cypher": """
        MATCH (from:Country) WHERE toLower(from.name) = toLower($from_country)
        MATCH (to:Country) WHERE toLower(to.name) = toLower($to_country)
        MATCH (from)-[v:NEEDS_VISA]->(to)
        RETURN v.visa_type AS visa_type
        """,
        "required": ["from_country", "to_country"]
    },

    # --- HOTEL SEARCH ---
    "search_hotels_by_city": {
        "cypher": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
        WHERE toLower(c.name) CONTAINS toLower($city)
        RETURN h.hotel_name AS hotel_name, 
               h.star_rating AS star_rating,
               h.cleanliness_base AS cleanliness, h.value_for_money_base AS value, h.location_base AS location
        ORDER BY h.star_rating DESC LIMIT 10
        """,
        "required": ["city"]
    },
    "hotels_by_country": {
        "cypher": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        WHERE toLower(co.name) CONTAINS toLower($country)
        RETURN h.hotel_name AS hotel_name, c.name AS city, 
               h.star_rating AS star_rating,
               h.cleanliness_base AS cleanliness, h.value_for_money_base AS value
        ORDER BY h.star_rating DESC LIMIT 10
        """,
        "required": ["country"], "defaults": {"limit": 10}
    },
    "hotels_by_continent": {
        "cypher": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        WHERE co.name IN $country_list
        RETURN h.hotel_name AS hotel_name, c.name AS city, co.name AS country, 
               h.star_rating AS star_rating,
               h.cleanliness_base AS cleanliness, h.value_for_money_base AS value
        ORDER BY h.star_rating DESC LIMIT 15
        """,
        "required": ["country_list"]
    },

    # --- FILTERING ---
    "filter_hotels_by_amenity": {
        "cypher": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        WHERE h.{amenity_key} >= $min_score
          AND ($country IS NULL OR toLower(co.name) CONTAINS toLower($country))
        RETURN h.hotel_name AS hotel_name, 
               h.star_rating AS star_rating, 
               h.{amenity_key} AS score, 
               c.name AS city, co.name AS country,
               h.cleanliness_base AS cleanliness, h.value_for_money_base AS value
        ORDER BY score DESC LIMIT 10
        """,
        "required": ["amenity_key", "min_score"], "defaults": {"country": None}
    },
    "hotels_by_value_for_money": {
        "cypher": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
        WHERE toLower(c.name) CONTAINS toLower($city)
        RETURN h.hotel_name AS hotel_name, 
               h.star_rating AS star_rating, 
               h.value_for_money_base AS value,
               h.cleanliness_base AS cleanliness
        ORDER BY h.value_for_money_base DESC LIMIT 10
        """,
        "required": ["city"]
    },
    "filter_hotels_by_rating": {
        "cypher": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        WHERE h.star_rating >= $min_rating
          AND ($city IS NULL OR toLower(c.name) CONTAINS toLower($city))
          AND ($country IS NULL OR toLower(co.name) CONTAINS toLower($country))
        RETURN h.hotel_name AS hotel_name, 
               h.star_rating AS star_rating, co.name AS country,
               h.cleanliness_base AS cleanliness, h.value_for_money_base AS Money, h.comfort_base AS comfort
        ORDER BY h.star_rating DESC LIMIT 10
        """,
        "required": ["min_rating"], "defaults": {"city": None, "country": None}
    },
    "find_available_hotels": {
        "cypher": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
        WHERE toLower(c.name) CONTAINS toLower($city)
          AND h.cleanliness_base >= 7.0 AND h.value_for_money_base >= 7.0
        RETURN h.hotel_name AS hotel_name, 
               h.star_rating AS star_rating,
               h.cleanliness_base AS cleanliness, 
               h.value_for_money_base AS value
        ORDER BY h.star_rating DESC LIMIT 10
        """,
        "required": ["city"], "defaults": {"min_cleanliness": 7.0, "min_value": 7.0}
    },
    
    # --- COMPARISON ---
    "compare_hotels": {
        "cypher": """
        UNWIND $hotel_list AS hn
        MATCH (h:Hotel) WHERE toLower(h.hotel_name) CONTAINS toLower(hn)
        RETURN h.hotel_name AS hotel_name, 
               h.star_rating AS star_rating, 
               h.cleanliness_base AS cleanliness, 
               h.value_for_money_base AS value,
               h.comfort_base AS comfort
        """,
        "required": ["hotel_list"]
    },

    # --- RECOMMENDATION ---
    "recommend_hotels_for_traveller_type": {
        "cypher": """
        MATCH (t:Traveller {type: $traveller_type})-[:WROTE]->(rev:Review)-[:REVIEWED]->(h:Hotel)
        MATCH (h)-[:LOCATED_IN]->(:City)-[:LOCATED_IN]->(co:Country)
        WHERE ($country IS NULL OR toLower(co.name) CONTAINS toLower($country))
        WITH h, AVG(rev.score_overall) AS score, co
        RETURN h.hotel_name AS hotel_name, 
               h.star_rating AS star_rating, 
               score, 
               co.name AS country,
               h.cleanliness_base AS cleanliness, h.value_for_money_base AS value
        ORDER BY score DESC LIMIT 10
        """,
        "required": ["traveller_type"], "defaults": {"country": None}
    },
    "hotels_for_family": {
        "cypher": """
        MATCH (t:Traveller {type: 'Family'})-[:WROTE]->(rev:Review)-[:REVIEWED]->(h:Hotel)
        MATCH (h)-[:LOCATED_IN]->(:City)-[:LOCATED_IN]->(co:Country)
        WHERE ($country IS NULL OR toLower(co.name) CONTAINS toLower($country))
        WITH h, AVG(rev.score_overall) AS score, co
        RETURN h.hotel_name AS hotel_name, 
               h.star_rating AS star_rating, 
               score, 
               co.name AS country,
               h.cleanliness_base AS cleanliness, h.value_for_money_base AS value
        ORDER BY score DESC LIMIT 10
        """,
        "required": [], "defaults": {"country": None}
    },

    # --- DETAILS / REVIEWS ---
    "hotel_details": {
        "cypher": """
        MATCH (h:Hotel) WHERE toLower(h.hotel_name) CONTAINS toLower($hotel_name)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        RETURN h.hotel_name AS hotel_name, 
               h.star_rating AS star_rating, 
               h.cleanliness_base AS cleanliness, h.value_for_money_base AS value,
               COLLECT(r.review_text)[0..3] AS sample_reviews
        """,
        "required": ["hotel_name"]
    },
    "top_reviews_for_hotel": {
        "cypher": """
        MATCH (h:Hotel)<-[:REVIEWED]-(rev:Review)
        WHERE toLower(h.hotel_name) CONTAINS toLower($hotel_name)
        RETURN h.hotel_name AS hotel_name,
               h.star_rating AS star_rating,
               rev.review_text AS review, 
               rev.score_overall AS score
        ORDER BY score DESC LIMIT 5
        """,
        "required": ["hotel_name"]
    },
    "hotel_reviews_by_age_group": {
        "cypher": """
        MATCH (t:Traveller)-[:WROTE]->(rev:Review)-[:REVIEWED]->(h:Hotel)
        WHERE toLower(h.hotel_name) CONTAINS toLower($hotel_name)
        RETURN t.age_group AS age_group, 
               AVG(rev.score_overall) AS avg_score, 
               h.star_rating AS star_rating
        ORDER BY avg_score DESC
        """,
        "required": ["hotel_name"]
    },
    "top_hotels_in_country_or_city": {
        "cypher": """
        CALL { 
          WITH $city AS city 
          MATCH (h:Hotel)-[:LOCATED_IN]->(c:City) WHERE toLower(c.name) CONTAINS toLower(city) RETURN h 
          UNION 
          WITH $country AS country 
          MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country) WHERE toLower(co.name) CONTAINS toLower(country) RETURN h 
        } 
        RETURN h.hotel_name AS hotel_name, 
               h.star_rating AS star_rating,
               h.cleanliness_base AS cleanliness, h.value_for_money_base AS value
        ORDER BY h.star_rating DESC LIMIT 10
        """,
        "required": [], "defaults": {"limit": 10, "city": "", "country": ""}
    }
}

# ----------------------------------------------------------------------------
# 4. ROUTING & MAPPING
# ----------------------------------------------------------------------------
def _route_intent(intent, entities):
    # Fix 1: If searching by city but only country provided -> switch to country
    if intent == "search_hotels_by_city" and not entities.get("target_city") and entities.get("target_country"):
        return "hotels_by_country"
        
    # Fix 2: If finding by country but "target_countries" is a LIST (e.g. Europe) -> switch to continent
    # This specifically fixes the "Europe" query returning No Records
    if entities.get("target_countries") and len(entities["target_countries"]) > 1:
        return "hotels_by_continent"

    return intent if intent in CYPHER_TEMPLATES else "search_hotels_by_city"
def _map_parameters(template_key, entities):
    params = {
        "city": None, "country": None, "limit": 10,
        "min_rating": 4, "min_cleanliness": 7.0, "min_value": 7.0,
        "min_score": 7.0,
        "from_country": "Egypt", "to_country": "Spain"
    }
    
    if entities.get("target_country"):
        params["country"] = entities["target_country"]
        params["to_country"] = entities["target_country"]
        params["country_list"] = [entities["target_country"]]
        
    if entities.get("target_countries"):
        params["country_list"] = entities["target_countries"]
    
    if entities.get("target_city"):
        params["city"] = entities["target_city"]

    # Visa: Locations order is [From, To]
    if entities.get("locations") and len(entities["locations"]) >= 2:
         params["from_country"] = entities["locations"][0]
         params["to_country"] = entities["locations"][-1]

    if entities.get("hotel_names"): 
        params["hotel_name"] = entities["hotel_names"][0]
        params["hotel_list"] = entities["hotel_names"]
    
    if entities.get("traveller_type"): 
        params["traveller_type"] = entities["traveller_type"]

    if entities.get("amenities_scores"): 
        params["amenity_key"] = entities["amenities_scores"][0].get("db_column")
        valid = ["cleanliness_base", "comfort_base", "facilities_base", "location_base", "staff_base", "value_for_money_base"]
        if params["amenity_key"] not in valid: params["amenity_key"] = None

    return params

# ----------------------------------------------------------------------------
# 5. MAIN RUN
# ----------------------------------------------------------------------------
def run_retrieval(intents_input, entities, embedding_vector=None, model_key="model_A", strategy="baseline"):
    results = {"baseline": {}, "embedding": {}}
    intents_list = [intents_input] if isinstance(intents_input, str) else intents_input

    # 1. BASELINE
    if strategy in ["baseline", "hybrid"]:
        for intent in intents_list:
            tk = _route_intent(intent, entities)
            if tk in CYPHER_TEMPLATES:
                tmpl = CYPHER_TEMPLATES[tk]
                params = _map_parameters(tk, entities)
                
                q = tmpl["cypher"]
                if "{amenity_key}" in q: 
                    if params.get("amenity_key"): q = q.format(amenity_key=params["amenity_key"])
                    else: q = None
                
                if q:
                    try:
                        results["baseline"][intent] = retriever_instance.run_cypher(q, params)
                    except Exception as e:
                        results["baseline"][intent] = [{"error": str(e)}]

    # 2. EMBEDDING
    if strategy in ["embedding", "hybrid"] and embedding_vector:
        target_country = entities.get("target_country", None)
        
        # A. Hotels (Neo4j)
        results["embedding"]["hotel"] = retriever_instance.search_neo4j_hotels(
            vector=embedding_vector, model_key=model_key, country_filter=target_country
        )
        
        # B. Visa (FAISS) - Conditional
        if any(k in str(intents_list).lower() for k in ["visa", "passport", "entry", "permit"]):
            results["embedding"]["visa"] = retriever_instance.search_faiss_visa(
                vector=embedding_vector, model_key=model_key
            )
            
    return results

def close():
    retriever_instance.close()