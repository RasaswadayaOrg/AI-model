import psycopg2
from typing import Dict, Any
from datetime import datetime

DB_URL = "postgresql://postgres.gmyqbkladleqthcdpeja:rasaya1234rasaya@aws-1-ap-northeast-2.pooler.supabase.com:6543/postgres?sslmode=require"

def get_live_dataset() -> Dict[str, Any]:
    print("📥 Fetching live data from Supabase...")
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # 1. Fetch Users
    cur.execute('SELECT "id", "city" FROM "User"')
    users_raw = cur.fetchall()
    users = [{"user_id": str(row[0]), "location": row[1] or "Colombo"} for row in users_raw]
    
    # 2. Fetch Artists
    cur.execute('SELECT "id", "genre" FROM "Artist"')
    artists_raw = cur.fetchall()
    artists = []
    for row in artists_raw:
        genres = [g.strip() for g in row[1].split(",")] if row[1] else ["General"]
        artists.append({
            "artist_id": str(row[0]),
            "genres": genres
        })
        
    # 3. Fetch Events & Performances
    cur.execute('SELECT "id", "city", "category" FROM "Event"')
    events_raw = cur.fetchall()
    
    cur.execute('SELECT "eventId", "artistId" FROM "Performance"')
    performances = cur.fetchall()
    
    event_artists = {}
    for ev_id, art_id in performances:
        ev_str = str(ev_id)
        if ev_str not in event_artists:
            event_artists[ev_str] = []
        event_artists[ev_str].append(str(art_id))
        
    events = []
    for row in events_raw:
        ev_id = str(row[0])
        region = row[1] or "Colombo"
        category = str(row[2]) if row[2] else "General"
        
        events.append({
            "event_id": ev_id,
            "region": region,
            "genres": [category],
            "artist_ids": event_artists.get(ev_id, [])
        })
        
    # 4. Fetch Follows
    cur.execute('SELECT "userId", "artistId", "createdAt" FROM "Follower"')
    follows = []
    for row in cur.fetchall():
        follows.append({
            "user_id": str(row[0]),
            "artist_id": str(row[1]),
            "timestamp": str(row[2]) if row[2] else str(datetime.now())
        })
        
    conn.close()
    
    dataset = {
        "users": users,
        "artists": artists,
        "events": events,
        "interactions": {
            "follows": follows
        }
    }
    
    print(f"✅ Fetched {len(users)} Users, {len(artists)} Artists, {len(events)} Events, {len(follows)} Follows")
    return dataset

if __name__ == "__main__":
    d = get_live_dataset()
    print(d["users"][:2] if len(d["users"])>0 else "No users")
