import pandas as pd
from neo4j import GraphDatabase
import re

# ------------------------------------------------------
# Utility: Convert "18-24" style age ranges to integers
# ------------------------------------------------------
def read_config(file_path="config.txt"):
    config = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    return config

def convert_age_group(age_group_str):
    """
    Converts age group strings like '18-24', '25-34', '45+' into integer midpoints.
    """
    if pd.isna(age_group_str):
        return None

    s = age_group_str.strip()

    if "+" in s:        # Example: "45+"
        base = int(s.replace("+", ""))
        return base + 5  # approximate midpoint

    match = re.match(r"(\d+)-(\d+)", s)
    if match:
        low = int(match.group(1))
        high = int(match.group(2))
        return (low + high) // 2

    return None

# ------------------------------------------------------
# Knowledge Graph Builder Class
# ------------------------------------------------------
class KGBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def execute(self, cypher, parameters=None):
        with self.driver.session(database="neo4j") as session:
            result = session.run(cypher, parameters or {})
            return list(result)

    # ------------------ Constraints ------------------
    def create_constraints(self):
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Traveller) REQUIRE t.user_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (h:Hotel) REQUIRE h.hotel_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Review) REQUIRE r.review_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:City) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (co:Country) REQUIRE co.name IS UNIQUE"
        ]
        for c in constraints:
            print("Running constraint:", c)
            self.execute(c)

    # ------------------ Countries ------------------
    def create_countries(self, hotels_df):
        cypher = """
        UNWIND $rows AS row
        MERGE (c:Country {name: row.country})
        """
        countries = [{"country": c} for c in hotels_df["country"].unique()]
        self.execute(cypher, {"rows": countries})
        print(f"Created {len(countries)} Country nodes.")

    # ------------------ Cities ------------------
    def create_cities(self, hotels_df):
        rows = [{"city": r["city"], "country": r["country"]} for _, r in hotels_df.iterrows()]
        cypher = """
        UNWIND $rows AS row
        MERGE (c:City {name: row.city})
        MERGE (co:Country {name: row.country})
        MERGE (c)-[:LOCATED_IN]->(co)
        """
        self.execute(cypher, {"rows": rows})
        print(f"Created {len(rows)} City nodes.")

    # ------------------ Hotels ------------------
    def create_hotels(self, hotels_df):
        rows = []
        for _, r in hotels_df.iterrows():
            rows.append({
                "hotel_id": r["hotel_id"],
                "hotel_name": r["hotel_name"],
                "city": r["city"],
                "star_rating": r["star_rating"],
                "cleanliness_base": r["cleanliness_base"],
                "comfort_base": r["comfort_base"],
                "facilities_base": r["facilities_base"],
                "location_base": r["location_base"],
                "staff_base": r["staff_base"],
                "value_for_money_base": r["value_for_money_base"]
            })
        cypher = """
        UNWIND $rows AS row
        MERGE (h:Hotel {hotel_id: row.hotel_id})
        SET h.hotel_name = row.hotel_name,
            h.star_rating = row.star_rating,
            h.cleanliness_base = row.cleanliness_base,
            h.comfort_base = row.comfort_base,
            h.facilities_base = row.facilities_base,
            h.location_base = row.location_base,
            h.staff_base = row.staff_base,
            h.value_for_money_base = row.value_for_money_base
        WITH h, row
        MATCH (c:City {name: row.city})
        MERGE (h)-[:LOCATED_IN]->(c)
        """
        self.execute(cypher, {"rows": rows})
        print(f"Created {len(rows)} Hotel nodes.")

    # ------------------ Travellers ------------------
    def create_travellers(self, users_df):
        rows = []
        for _, r in users_df.iterrows():
            rows.append({
                "user_id": r["user_id"],
                "gender": r["user_gender"],
                "type": r["traveller_type"],
                "age_group": r["age_group"],  # keep as string, do not convert
                "country": r["country"]
            })
        cypher = """
        UNWIND $rows AS row
        MERGE (t:Traveller {user_id: row.user_id})
        SET t.gender = row.gender,
            t.type = row.type,
            t.age_group = row.age_group
        WITH t, row
        MERGE (co:Country {name: row.country})
        MERGE (t)-[:FROM_COUNTRY]->(co)
        """
        self.execute(cypher, {"rows": rows})
        print(f"Created {len(rows)} Traveller nodes.")

    # ------------------ Reviews ------------------
    def create_reviews(self, reviews_df):
        rows = []
        for _, r in reviews_df.iterrows():
            rows.append({
                "review_id": r["review_id"],
                "user_id": r["user_id"],
                "hotel_id": r["hotel_id"],
                "review_date": r["review_date"],
                "score_overall": r["score_overall"],
                "score_cleanliness": r["score_cleanliness"],
                "score_comfort": r["score_comfort"],
                "score_facilities": r["score_facilities"],
                "score_location": r["score_location"],
                "score_staff": r["score_staff"],
                "score_value_for_money": r["score_value_for_money"],
                "review_text": r["review_text"]
            })
        cypher = """
        UNWIND $rows AS row
        MERGE (rev:Review {review_id: row.review_id})
        SET rev.review_date = row.review_date,
            rev.score_overall = row.score_overall,
            rev.score_cleanliness = row.score_cleanliness,
            rev.score_comfort = row.score_comfort,
            rev.score_facilities = row.score_facilities,
            rev.score_location = row.score_location,
            rev.score_staff = row.score_staff,
            rev.score_value_for_money = row.score_value_for_money,
            rev.review_text = row.review_text
        WITH rev, row
        MATCH (t:Traveller {user_id: row.user_id})
        MATCH (h:Hotel {hotel_id: row.hotel_id})
        MERGE (t)-[:WROTE]->(rev)
        MERGE (rev)-[:REVIEWED]->(h)
        MERGE (t)-[:STAYED_AT]->(h)
        """
        self.execute(cypher, {"rows": rows})
        print(f"Created {len(rows)} Review nodes.")

    # ------------------ Visa ------------------
    def create_visa_rules(self, visa_df):
        rows = []
        for _, r in visa_df.iterrows():
            if str(r["requires_visa"]).strip().lower() in ("yes", "true", "1"):
                rows.append({
                    "from_country": r["from"],
                    "to_country": r["to"],
                    "visa_type": r["visa_type"]
                })
        cypher = """
        UNWIND $rows AS row
        MATCH (c1:Country {name: row.from_country})
        MATCH (c2:Country {name: row.to_country})
        MERGE (c1)-[:NEEDS_VISA {visa_type: row.visa_type}]->(c2)
        """
        self.execute(cypher, {"rows": rows})
        print(f"Created {len(rows)} NEEDS_VISA relationships.")

    def compute_exceeds_expectations(self, output_csv="exceeds_report.csv"):
        """
        Computes per age_group statistics (min, max, avg improvement) for hotels
        that exceed expectations for Solo Female travellers, keeping age_group as string.
        """
        print("Computing exceeds expectations per age group...")

        # Get all unique age_groups from Solo Female travellers
        cypher_age_groups = """
        MATCH (t:Traveller)
        WHERE t.type = 'Solo' AND t.gender = 'Female' AND t.age_group IS NOT NULL
        RETURN DISTINCT t.age_group AS age_group
        ORDER BY t.age_group
        """
        age_groups = [r["age_group"] for r in self.execute(cypher_age_groups)]

        results = []

        for age_group in age_groups:
            # Compute improvement % for hotels for this age_group
            cypher = """
            MATCH (t:Traveller)-[:WROTE]->(rev:Review)-[:REVIEWED]->(h:Hotel)
            WHERE t.type = 'Solo' AND t.gender = 'Female' AND t.age_group = $age_group
            AND h.cleanliness_base IS NOT NULL AND h.comfort_base IS NOT NULL AND h.facilities_base IS NOT NULL
            WITH h,
                AVG(rev.score_cleanliness + rev.score_comfort + rev.score_facilities) AS avg_review_sum,
                (h.cleanliness_base + h.comfort_base + h.facilities_base) AS base_sum
            WHERE base_sum >= avg_review_sum
            WITH ((base_sum - avg_review_sum)/avg_review_sum)*100 AS improvement
            RETURN MIN(improvement) AS min_improve,
                MAX(improvement) AS max_improve,
                AVG(improvement) AS avg_improve
            """
            stats = self.execute(cypher, {"age_group": age_group})

            if stats:
                r = stats[0]
                results.append({
                    "age_group": age_group,
                    "min_improve": round(r["min_improve"], 2) if r["min_improve"] is not None else None,
                    "max_improve": round(r["max_improve"], 2) if r["max_improve"] is not None else None,
                    "avg_improve": round(r["avg_improve"], 2) if r["avg_improve"] is not None else None
                })

        # Convert to DataFrame and save
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Exceeds expectations report saved to {output_csv}")
        return df


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    config = read_config("config.txt")

    builder = KGBuilder(
        uri=config["URI"],
        user=config["USERNAME"],
        password=config["PASSWORD"]
    )

    # Load CSVs
    users_df = pd.read_csv("users.csv")
    hotels_df = pd.read_csv("hotels.csv")
    reviews_df = pd.read_csv("reviews.csv")
    visa_df = pd.read_csv("visa.csv")

    # Build KG
    builder.create_constraints()
    builder.create_countries(hotels_df)
    builder.create_cities(hotels_df)
    builder.create_hotels(hotels_df)
    builder.create_travellers(users_df)
    builder.create_reviews(reviews_df)
    builder.create_visa_rules(visa_df)

    # Compute exceeds expectations
    builder.compute_exceeds_expectations()

    builder.close()

if __name__ == "__main__":
    main()
