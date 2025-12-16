import spacy
import pandas as pd
import re
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIGURATION
# -----------------------------
EMBEDDING_MODELS = {
    "model_A": "all-MiniLM-L6-v2",    
    "model_B": "all-mpnet-base-v2"    
}

class HotelInputPreprocessor:
    def __init__(self, hotels_csv_path="hotels.csv", users_csv_path="users.csv"):
        print("--- Component 1 Initialization ---")
        
        # 1. Load Spacy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Error: Spacy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None

        # 2. Load Embedding Models
        self.embedders = {}
        try:
            for key, name in EMBEDDING_MODELS.items():
                self.embedders[key] = SentenceTransformer(name)
        except: pass

        # 3. Load Data & Mappings
        print(f"Loading known entities from {hotels_csv_path}...")
        try:
            hotels_df = pd.read_csv(hotels_csv_path)
            self.known_cities = set(hotels_df['city'].dropna().str.lower().unique())
            # Normalize DB countries
            self.known_countries = set([self.normalize_country_name(c).lower() for c in hotels_df['country'].dropna().unique()])
            self.known_hotels = set(hotels_df['hotel_name'].dropna().str.lower().unique())
            
            # Map Cities to Countries
            self.city_to_country = {}
            for _, row in hotels_df.iterrows():
                if pd.notna(row['city']) and pd.notna(row['country']):
                    norm_country = self.normalize_country_name(row['country'])
                    self.city_to_country[row['city'].lower()] = norm_country
            
            # Continent Map
            self.continent_map = {
                "europe": ["France", "United Kingdom", "Italy", "Germany", "Spain", "Netherlands", "Russia", "Turkey"],
                "africa": ["Egypt", "South Africa", "Nigeria"],
                "asia": ["Japan", "China", "India", "Thailand", "United Arab Emirates", "Singapore", "South Korea"],
                "north america": ["United States", "Canada", "Mexico"],
                "south america": ["Brazil", "Argentina"],
                "oceania": ["Australia", "New Zealand"]
            }
            
        except Exception as e:
            print(f"Warning: Could not load CSV files ({e}).")
            self.known_cities, self.known_countries, self.known_hotels = set(), set(), set()
            self.city_to_country = {}
            self.continent_map = {}

        # 4. Keyword Mappings
        self.traveller_type_map = {
            "solo": "Solo", "business": "Business", "family": "Family", "kids": "Family",
            "child": "Family", "couple": "Couple", "romantic": "Couple", "honey": "Couple"
        }
        self.score_map = {
            "clean": "cleanliness_base", "comfort": "comfort_base", "facilities": "facilities_base",
            "location": "location_base", "staff": "staff_base", "value": "value_for_money_base",
            "cheap": "value_for_money_base", "rating": "star_rating",
            "gym": "facilities_base", "spa": "facilities_base", "wifi": "facilities_base"
        }

    # --- HELPER: NORMALIZE COUNTRY NAMES ---
    def normalize_country_name(self, name: str) -> str:
        if not isinstance(name, str): return ""
        name = name.strip()
        if name.lower().startswith("the "):
            name = name[4:]
        
        abbrev_map = {
            "us": "United States", "usa": "United States", 
            "uk": "United Kingdom", "uae": "United Arab Emirates"
        }
        
        if name.lower() in abbrev_map:
            name = abbrev_map[name.lower()]
        
        return " ".join([w.capitalize() for w in name.split()])

    # ------------------------------------------------------------------------
    # 1.b ENTITY EXTRACTION (SMART PREPOSITION HANDLING)
    # ------------------------------------------------------------------------
    def extract_entities(self, text):
        text_lower = text.lower()
        entities = {
            "locations": [], "hotel_names": [], "traveller_type": None, 
            "amenities_scores": [], "target_countries": [], 
            "target_country": None, "target_city": None 
        }

        # Structure: (start_index, "Standardized Name", original_span)
        found_locs = []

        # 1. Extract Cities
        for city in self.known_cities:
            pattern = r'\b' + re.escape(city) + r'\b'
            for match in re.finditer(pattern, text_lower):
                found_locs.append((match.start(), city.title(), match.group()))
            
        # 2. Extract Countries (Full Names)
        for country_lower in self.known_countries:
            pattern = r'\b' + re.escape(country_lower) + r'\b'
            for match in re.finditer(pattern, text_lower):
                found_locs.append((match.start(), country_lower.title(), match.group()))

        # 3. Extract Abbreviations
        abbreviations = {"usa": "United States", "uk": "United Kingdom", "uae": "United Arab Emirates"}
        for abbrev, full_name in abbreviations.items():
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            for match in re.finditer(pattern, text_lower):
                found_locs.append((match.start(), full_name, match.group()))

        # 4. Sort by Position
        found_locs.sort(key=lambda x: x[0])

        # --- SMART LOGIC: DETECT FROM/TO ---
        origin = None
        destination = None
        
        for i, (start, name, original) in enumerate(found_locs):
            # Look at the 10 chars before the entity
            preceding_text = text_lower[max(0, start-10):start]
            
            if "from " in preceding_text:
                origin = name
            elif any(x in preceding_text for x in ["to ", "in ", "visit ", "trip ", "travel "]):
                destination = name
                
        # Fill entities list (If we found semantic Origin/Dest, prioritize that order)
        if origin and destination:
            entities["locations"] = [origin, destination]
            entities["target_country"] = destination
        else:
            # Fallback: Just list them in order
            entities["locations"] = [loc[1] for loc in found_locs]
            # Deduplicate preserving order
            seen = set()
            unique = []
            for l in entities["locations"]:
                if l not in seen: unique.append(l); seen.add(l)
            entities["locations"] = unique
            
            if unique:
                entities["target_country"] = unique[-1] # Default to last mentioned

        # Resolve Target City/Country Logic
        if entities["target_country"]:
            # Check if the identified target is actually a city
            if entities["target_country"].lower() in self.city_to_country:
                city = entities["target_country"]
                entities["target_city"] = city
                entities["target_country"] = self.city_to_country[city.lower()]
            else:
                # Normalize just in case
                norm = self.normalize_country_name(entities["target_country"])
                entities["target_country"] = norm

        # 5. Extract Continents
        for continent, countries in self.continent_map.items():
            if continent in text_lower:
                entities["target_countries"].extend(countries)

        # 6. Extract Hotels
        for hotel in self.known_hotels:
            if hotel in text_lower: entities["hotel_names"].append(hotel.title())

        # 7. Attributes
        for k, v in self.traveller_type_map.items():
            if k in text_lower: entities["traveller_type"] = v
        for k, v in self.score_map.items():
            if k in text_lower: entities["amenities_scores"].append({"db_column": v})

        return entities

    # ------------------------------------------------------------------------
    # 1.a INTENT CLASSIFICATION
    # ------------------------------------------------------------------------
    def classify_intent(self, text, entities):
        text_lower = text.lower()
        found_intents = []

        if "visa" in text_lower or "permit" in text_lower:
            found_intents.append("get_visa_type" if "type" in text_lower else "check_visa_requirement")

        if entities.get("hotel_names"):
            if len(entities["hotel_names"]) > 1: found_intents.append("compare_hotels")
            else: found_intents.append("hotel_details")
        
        elif entities.get("traveller_type") == "Family": 
            found_intents.append("hotels_for_family")
        elif entities.get("traveller_type"): 
            found_intents.append("recommend_hotels_for_traveller_type")
        
        elif "rating" in text_lower or "stars" in text_lower: 
            found_intents.append("filter_hotels_by_rating")
        elif entities.get("amenities_scores"):
            cols = [x["db_column"] for x in entities["amenities_scores"]]
            if "value_for_money_base" in cols and len(cols) == 1: 
                found_intents.append("hotels_by_value_for_money")
            elif len(cols) > 1 and entities.get("locations"): 
                found_intents.append("find_available_hotels")
            else:
                found_intents.append("filter_hotels_by_amenity")
        
        elif entities.get("target_countries"):
             if not any("recommend" in x or "family" in x for x in found_intents):
                found_intents.append("hotels_by_country")
        elif entities.get("locations"):
             if not any("recommend" in x or "family" in x for x in found_intents):
                found_intents.append("search_hotels_by_city")

        if not found_intents:
            found_intents.append("search_hotels_by_city")
            
        return found_intents

    # ------------------------------------------------------------------------
    # PROCESS
    # ------------------------------------------------------------------------
    def process(self, text, embedding_model="model_A"):
        entities = self.extract_entities(text)
        intents = self.classify_intent(text, entities)
        vector = []
        if embedding_model in self.embedders:
            vector = self.embedders[embedding_model].encode(text).tolist()
        
        return {
            "original_text": text,
            "intents": intents, 
            "entities": entities,
            "embedding": vector,
            "model_used": embedding_model
        }