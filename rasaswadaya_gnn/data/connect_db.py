import os
import sys
import psycopg2
import json
from datetime import datetime

DB_URL = "postgresql://postgres.gmyqbkladleqthcdpeja:rasaya1234rasaya@aws-1-ap-northeast-2.pooler.supabase.com:6543/postgres?sslmode=require"

def default_if_none(val, default):
    return val if val is not None else default

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
        dataset["users"].append({
            "user_id": str(uid),
            "name": str(name),
            "ethnicity": "sinhala", # default
            "language_preferences": ["sinhala", "english"], # default
            "city": default_if_none(city, "colombo").lower(),
            "art_interests": categories if categories else ["music"],
            "culture_preferences": interests if interests else ["contemporary"],
            "mood_preferences": ["energetic", "relaxed"],# default
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
        genres_list = [g.strip().lower() for g in genre.split(',')] if genre else ["contemporary"]
        dataset["artists"].append({
            "artist_id": str(aid),
            "name": str(name),
            "art_forms": [profession.lower() if profession else "music"],
            "genres": genres_list,
            "styles": ["contemporary"],
            "language": ["sinhala", "english"],
            "city": default_if_none(location, "colombo").lower(),
            "style": ["contemporary"],
            "mood_tags": ["energetic"],
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
        dataset["events"].append({
            "event_id": str(eid),
            "name": str(title),
            "artist_ids": event_artists.get(eid, []),
            "art_forms": [category.lower() if category else "music"],
            "genres": ["contemporary"],
            "language": ["sinhala"],
            "city": default_if_none(city, "colombo").lower(),
            "venue": default_if_none(venue, "Main Hall"),
            "style": ["contemporary"],
            "mood_tags": ["energetic"],
            "festival": None,
            "festivals": [],
            "event_type": "concert",
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
    out_path = os.path.join(os.path.dirname(__file__), 'sample_dataset', 'rasaswadaya_dataset.json')
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

if __name__ == "__main__":
    fetch_data()
