# -------------------------
# LLM Multimodal Géospatial + RAG
# -------------------------

# 1️⃣ Imports
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
from shapely.geometry import mapping
import os
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# 2️⃣ Charger les données
# -------------------------
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# a) Tabulaire
tabular = pd.read_csv("../data/disaster_locations/disaster_location_1960-2018.csv")

# b) Geo
countries_boundaries = gpd.read_file("../data/countries_boundaries/ne_110m_admin_0_boundary_lines_land.shp")
population_zones = gpd.read_file("../data/population_shp/JRC_POPULATION_2018.shp")

# c) Raster
raster = rasterio.open("../data/population/population_AF01_2018-10-01.tif")

# -------------------------
# 3️⃣ Fusion spatiale & métriques
# -------------------------

# Jointure spatiale : population × zones inondables
merged = gpd.sjoin(population_zones, countries_boundaries, how="inner", predicate="intersects")
print(merged)
# merged["population_exposed"] = merged["population"]  # exemple simple

# Extraire valeur moyenne raster pour chaque polygone
ndvi_means = []
for geom in merged.geometry:
    out_image, out_transform = mask(raster, [mapping(geom)], crop=True)
    ndvi_means.append(out_image.mean())
merged["ndvi_mean"] = ndvi_means

# -------------------------
# 4️⃣ Résumer en JSON pour LLM
# -------------------------
# insights_json = merged[["zone_name", "population", "population_exposed", "ndvi_mean"]].to_dict(orient="records")
insights_json = merged.to_dict(orient="records")

# -------------------------
# 5️⃣ Embeddings et vector DB
# -------------------------
embedder = SentenceTransformer("../models/all-MiniLM-L6-v2")
documents = [str(record) for record in insights_json]
embeddings = embedder.encode(documents)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# -------------------------
# 6️⃣ Query utilisateur + récupération
# -------------------------
query = "Zones avec population exposée > 1000 et NDVI moyen < 0.5"
query_embedding = embedder.encode([query])
D, I = index.search(np.array(query_embedding), k=5)
results = [documents[i] for i in I[0]]

# -------------------------
# 7️⃣ LLM pour synthèse
# -------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

context = "\n".join(results)
prompt = f"""
Voici les informations sur les zones à risque :
{context}

Peux-tu générer un rapport synthétique avec recommandations pour prioriser les interventions ?
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

print("=== Rapport Synthétique ===")
print(response.choices[0].message.content)
