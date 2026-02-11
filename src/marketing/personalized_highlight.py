import json
from src.marketing.persona import build_persona
from src.marketing.highlights import generate_highlights

def load_user_favorites(profile_path="data/user_profiles.json", user_id="local"):
    try:
        with open(profile_path, "r") as f:
            profiles = json.load(f)
        return profiles.get(user_id, {}).get("favorites", [])
    except Exception:
        return []

def get_persona_and_highlights(isbn, books, user_id="local"):
    fav_isbns = load_user_favorites(user_id=user_id)
    persona = build_persona(fav_isbns, books)
    return generate_highlights(isbn, persona, books)
