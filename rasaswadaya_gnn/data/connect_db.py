import os
import sys
import psycopg2
import json
import re
from datetime import datetime
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preference_mapping import build_preference_profile, get_art_form, get_genre, normalise_city, normalise_token

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

DB_URL = os.environ.get("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL environment variable not set")

def default_if_none(val, default):
    return val if val is not None else default


PROFESSION_ART_FORM_HINTS = {
    "singer": "music",
    "vocalist": "music",
    "musician": "music",
    "composer": "music",
    "rapper": "music",
    "band": "music",
    "dancer": "dance",
    "choreographer": "dance",
    "actor": "drama",
    "actress": "drama",
    "theatre": "drama",
    "theater": "drama",
    "playwright": "drama",
    "director": "film",
    "filmmaker": "film",
    "cinema": "film",
}


def split_profile_tokens(value):
    tokens = []
    for part in re.split(r"[,/|;&]+", str(value or "")):
        token = normalise_token(part)
        if token:
            tokens.append(token)
            tokens.extend(piece for piece in token.split("_") if len(piece) > 2)
    return tokens


def append_unique(target, values):
    for value in values:
        token = normalise_token(value)
        if token and token not in target:
            target.append(token)


def infer_artist_art_forms(profession, genre):
    art_forms = []
    for token in [*split_profile_tokens(profession), *split_profile_tokens(genre)]:
        art_form = get_art_form(token) or PROFESSION_ART_FORM_HINTS.get(token)
        if art_form and art_form not in art_forms:
            art_forms.append(art_form)
    return art_forms or ["music"]


def fetch_single_user(user_id: str):
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute('''
        SELECT u.id, u."fullName", u.city, u."createdAt",
               p.categories, p.interests
        FROM "User" u
        LEFT JOIN "UserPreference" p ON u.id = p."userId"
        WHERE u.id = %s
    ''', (user_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return None

    uid, name, city, created_at, categories, interests = row
    profile = build_preference_profile(categories or ["music"], interests or ["contemporary"], city)
    return {
        "user_id": str(uid),
        "name": str(name),
        "ethnicity": "sinhala",
        "language_preferences": ["sinhala", "english"],
        "city": profile["city"],
        "art_interests": profile["art_interests"],
        "genres": profile["genres"],
        "styles": profile["styles"],
        "culture_preferences": profile["culture_preferences"],
        "mood_preferences": profile["mood_preferences"],
        "match_terms": profile["match_terms"],
        "activity_level": "medium",
        "join_date": created_at.isoformat() if created_at else datetime.now().isoformat(),
    }


def fetch_user_followed_artists(user_id: str):
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute('SELECT "artistId" FROM "Follower" WHERE "userId" = %s', (user_id,))
    artist_ids = [str(row[0]) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return artist_ids

def fetch_data():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    dataset = {
        "users": [],
        "artists": [],
        "events": [],
        "interactions": {
            "follows": [],
            "attends": []
        }
    }
    
    # 1. Fetch Users
    cur.execute('''
        SELECT u.id, u."fullName", u.city, u."createdAt",
               p.categories, p.interests
        FROM "User" u
        LEFT JOIN "UserPreference" p ON u.id = p."userId"
    ''')
    for row in cur.fetchall():
        uid, name, city, created_at, categories, interests = row
        profile = build_preference_profile(categories or ["music"], interests or ["contemporary"], city)
        dataset["users"].append({
            "user_id": str(uid),
            "name": str(name),
            "ethnicity": "sinhala", # default
            "language_preferences": ["sinhala", "english"], # default
            "city": profile["city"],
            "art_interests": profile["art_interests"],
            "genres": profile["genres"],
            "styles": profile["styles"],
            "culture_preferences": profile["culture_preferences"],
            "mood_preferences": profile["mood_preferences"],
            "match_terms": profile["match_terms"],
            "activity_level": "medium",
            "join_date": created_at.isoformat() if created_at else datetime.now().isoformat()
        })
        
    # 2. Fetch Artists
    cur.execute('''
        SELECT id, name, profession, genre, location
        FROM "Artist"
    ''')
    for row in cur.fetchall():
        aid, name, profession, genre, location = row
        art_forms = infer_artist_art_forms(profession, genre)
        genres_list = split_profile_tokens(genre) or ["contemporary"]
        artist_profile = build_preference_profile(art_forms, genres_list, location)
        dataset["artists"].append({
            "artist_id": str(aid),
            "name": str(name),
            "art_forms": art_forms,
            "genres": artist_profile["genres"] or genres_list,
            "styles": artist_profile["styles"] or genres_list[:1],
            "language": ["sinhala", "english"],
            "city": normalise_city(location),
            "style": artist_profile["culture_preferences"],
            "mood_tags": artist_profile["mood_preferences"],
            "festivals": [],
            "popularity": "established",
            "follower_count": 0,
            "verified": True
        })
        
    # 3. Fetch Events & Performances
    # First get Performances
    cur.execute('SELECT "eventId", "artistId" FROM "Performance"')
    event_artists = {}
    for eid, aid in cur.fetchall():
        if (eid not in event_artists): event_artists[eid] = []
        event_artists[eid].append(str(aid))

    cur.execute('''
        SELECT id, title, category, city, venue, "eventDate", capacity, price
        FROM "Event"
    ''')
    for row in cur.fetchall():
        eid, title, category, city, venue, eventDate, capacity, price = row
        cat_token = normalise_token(category) or "cultural"
        art_form = get_art_form(category)
        genre_token = get_genre(category)

        art_interests = [art_form] if art_form else []
        genres = []
        if genre_token:
            genres.append(genre_token)
        if art_form:
            genres.append(art_form)
        if cat_token not in ("general", "other", ""):
            append_unique(genres, [cat_token])

        name_lower = str(title or "").lower()
        moods = []
        if any(word in name_lower for word in ["traditional", "heritage", "folk", "classical", "ancient"]):
            moods.extend(["traditional", "cultural_pride"])
        if any(word in name_lower for word in ["festival", "celebration", "perahera"]):
            moods.extend(["festive", "energetic"])
        if any(word in name_lower for word in ["contemporary", "modern", "fusion"]):
            moods.append("contemporary")
        if any(word in name_lower for word in ["vesak", "poson", "pooja", "religious", "temple"]):
            moods.extend(["devotional", "spiritual"])
        if any(word in name_lower for word in ["film", "cinema", "screening"]):
            moods.append("romantic_longing" if "romantic" in name_lower else "dramatic")

        event_profile = {
            "art_interests": art_interests,
            "genres": genres,
            "culture_preferences": [cat_token] if cat_token not in ("general", "other", "") else [],
            "mood_preferences": moods,
        }
        dataset["events"].append({
            "event_id": str(eid),
            "name": str(title),
            "artist_ids": event_artists.get(eid, []),
            "art_forms": event_profile["art_interests"],
            "art_interests": event_profile["art_interests"],
            "genres": event_profile["genres"] or [normalise_token(category) or "contemporary"],
            "language": ["sinhala"],
            "city": normalise_city(city),
            "venue": default_if_none(venue, "Main Hall"),
            "style": event_profile["culture_preferences"],
            "mood_tags": event_profile["mood_preferences"],
            "mood_preferences": event_profile["mood_preferences"],
            "festival": None,
            "festivals": [],
            "event_type": cat_token,
            "date": eventDate.isoformat() if eventDate else datetime.now().isoformat(),
            "capacity": default_if_none(capacity, 100),
            "ticket_price": float(default_if_none(price, 0))
        })
        
    # 4. Fetch Follows
    cur.execute('SELECT "userId", "artistId", "createdAt" FROM "Follower"')
    for uid, aid, cat in cur.fetchall():
        dataset["interactions"]["follows"].append({
            "user_id": str(uid),
            "artist_id": str(aid),
            "timestamp": cat.isoformat() if cat else datetime.now().isoformat(),
            "compatibility_score": 5
        })
        
    # 5. Fetch Attends (Interest + Tickets)
    cur.execute('SELECT "userId", "eventId", "createdAt" FROM "Interest"')
    for uid, eid, cat in cur.fetchall():
        dataset["interactions"]["attends"].append({
            "user_id": str(uid),
            "event_id": str(eid),
            "timestamp": cat.isoformat() if cat else datetime.now().isoformat(),
            "rating": 5
        })

    conn.close()
    
    # Save the file
    out_path = os.path.join(os.path.dirname(__file__), 'sample_dataset', 'rasaswadaya_db_snapshot.json')
    with open(out_path, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    # Also save as pkl for compatibility
    import pickle
    pkl_path = out_path.replace('.json', '.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(dataset, f)
        
    print(f"✅ Successfully exported DB to {out_path} and {pkl_path}")
    print(f"Users: {len(dataset['users'])}, Artists: {len(dataset['artists'])}, Events: {len(dataset['events'])}")
    print(f"Follows: {len(dataset['interactions']['follows'])}, Attends: {len(dataset['interactions']['attends'])}")
    return dataset

if __name__ == "__main__":
    fetch_data()
