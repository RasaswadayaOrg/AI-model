import json
import pandas as pd
from pathlib import Path

# Load the rich dataset
with open('rasaswadaya_gnn/data/sample_dataset/rasaswadaya_dataset_updated.json') as f:
    data = json.load(f)

out_dir = Path('rasaswadaya_gnn/data/sample_dataset/csv_export_updated_real')

# USERS
users_rows = []
for u in data['users']:
    users_rows.append({
        'user_id':       u['user_id'],
        'name':          u['name'],
        'city':          u['city'],
        'art_interests': str(u.get('art_interests', [])),
        'genres':        str(u.get('culture_preferences', [])),
        'interests':     str(u.get('art_interests', [])),
        'moods':         str(u.get('mood_preferences', [])),
        'language':      u.get('language_preferences', ['sinhala'])[0] if u.get('language_preferences') else 'sinhala',
    })
users_df = pd.DataFrame(users_rows)
users_df.to_csv(out_dir / 'users.csv', index=False)
print(f"users.csv: {len(users_df)} rows")

# ARTISTS - rebuild from JSON (100 artists with proper fields)
artists_rows = []
for a in data['artists']:
    art_forms = a.get('art_forms', [])
    art_form = art_forms[0] if art_forms else 'music'
    artists_rows.append({
        'artist_id':     a['artist_id'],
        'name':          a['name'],
        'art_form':      art_form,
        'art_forms':     str(art_forms),
        'genres':        str(a.get('genres', [])),
        'styles':        str(a.get('styles', a.get('style', []))),
        'language':      str(a.get('language', ['sinhala'])[0] if a.get('language') else 'sinhala'),
        'languages':     str(a.get('language', [])),
        'city':          a.get('city', ''),
        'mood_tags':     str(a.get('mood_tags', [])),
        'festivals':     str(a.get('festivals', [])),
        'popularity':    a.get('popularity', 'mid_tier'),
        'follower_count': a.get('follower_count', 0),
        'verified':      a.get('verified', False),
        'era':           a.get('era', 'contemporary'),
        'notable_works': str(a.get('notable_works', [])),
        'awards':        str(a.get('awards', [])),
        'bio':           a.get('bio', ''),
    })
artists_df = pd.DataFrame(artists_rows)
artists_df.to_csv(out_dir / 'artists.csv', index=False)
print(f"artists.csv: {len(artists_df)} rows")

# EVENTS - keep existing (already has real data)
events_df = pd.read_csv(out_dir / 'events.csv')
print(f"events.csv: {len(events_df)} rows (kept)")

# FOLLOWS
interactions = data.get('interactions', {})
follows = interactions.get('follows', [])
follows_rows = [{'user_id': f['user_id'], 'artist_id': f['artist_id'], 'timestamp': f.get('timestamp', '')} for f in follows]
follows_df = pd.DataFrame(follows_rows)
follows_df.to_csv(out_dir / 'follows.csv', index=False)
print(f"follows.csv: {len(follows_df)} rows")

# ATTENDS
attends = interactions.get('attends', [])
attends_rows = [{'user_id': a['user_id'], 'event_id': a['event_id'], 'timestamp': a.get('timestamp', ''), 'rsvp_status': a.get('rsvp_status', 'interested')} for a in attends]
attends_df = pd.DataFrame(attends_rows)
attends_df.to_csv(out_dir / 'attends.csv', index=False)
print(f"attends.csv: {len(attends_df)} rows")

print("\nSample users:")
print(users_df[['user_id','name','city']].head(8).to_string())
print(f"\nFollow sample:")
print(follows_df.head(5).to_string())
