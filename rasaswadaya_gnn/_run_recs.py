#!/usr/bin/env python3
"""Run recommendation tests and collect output."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'streamlit_app')

from streamlit_app.utils.data_loader import load_users_list
from streamlit_app.utils.recommender import (
    get_recommendations,
    find_similar_users,
    calculate_user_similarity,
    load_model_and_data,
)

print("=" * 60)
print("RECOMMENDATION SYSTEM TEST")
print("=" * 60)

# Load data
_, dataset, _ = load_model_and_data()
users = load_users_list()
print(f"Users loaded: {len(users)}")
print(f"Artists in dataset: {len(dataset['artists'])}")
print(f"Events in dataset: {len(dataset['events'])}")

# --- Test 1: Specific user ---
print()
print("--- Test 1: Amila Wickremasinghe ---")
amila = next((u for u in users if 'Amila' in u.get('name', '')), None)
if amila:
    print(f"User: {amila['name']}")
    print(f"City: {amila.get('city', 'N/A')}")
    print(f"Interests: {amila.get('interests', amila.get('art_interests', 'N/A'))}")
    result = get_recommendations(amila)
    print(f"Artists recommended: {len(result.get('artists', []))}")
    print(f"Events recommended:  {len(result.get('events', []))}")
    print(f"Similar users found: {len(result.get('similar_users', []))}")
    print("\nTop 5 artist recommendations:")
    for i, a in enumerate(result.get('artists', [])[:5], 1):
        print(f"  {i}. {a['name']:<35} score={a['score']:.4f}  reason={a.get('reason','')}")
    print("\nTop 3 event recommendations:")
    for i, e in enumerate(result.get('events', [])[:3], 1):
        print(f"  {i}. {e.get('name','?'):<35} city={e.get('city','')}")
    print("\nTop 3 similar users:")
    for u in result.get('similar_users', [])[:3]:
        print(f"  - {u['name']:<30} similarity={u['similarity']:.4f}")
else:
    print("Amila not found, using first user")
    amila = users[0]
    result = get_recommendations(amila)
    print(f"User: {amila['name']}")
    print(f"Artists: {len(result.get('artists', []))}, Events: {len(result.get('events', []))}")
    for a in result.get('artists', [])[:5]:
        print(f"  {a['name']:<35} score={a['score']:.4f}")

# --- Test 2: Three more users ---
print()
print("--- Test 2: Multi-user sample ---")
for u in users[1:4]:
    result = get_recommendations(u)
    top = result.get('artists', [])[:3]
    print(f"\nUser: {u['name']} ({u.get('city','?')})")
    for a in top:
        print(f"  {a['name']:<35} score={a['score']:.4f}")

# --- Test 3: Collaborative filtering similarity ---
print()
print("=" * 60)
print("COLLABORATIVE FILTERING - SIMILARITY SCORES")
print("=" * 60)

# Enrich users with follows
from collections import defaultdict
import ast

def parse_list(val):
    if isinstance(val, list): return val
    if isinstance(val, str):
        try: return ast.literal_eval(val)
        except: return [val]
    return []

follows_map = defaultdict(list)
for row in dataset.get('follows', []):
    uid = row.get('user_id') or row.get('userId')
    aid = row.get('artist_id') or row.get('artistId')
    if uid and aid:
        follows_map[str(uid)].append(str(aid))

enriched = []
for u in dataset['users'][:10]:
    uid = str(u.get('user_id', u.get('userId', '')))
    u_copy = dict(u)
    u_copy['follows'] = follows_map.get(uid, [])
    enriched.append(u_copy)

print("\nPairwise user similarity (first 5 users):")
for i in range(5):
    for j in range(i+1, 5):
        sim = calculate_user_similarity(enriched[i], enriched[j])
        n1 = enriched[i].get('name', f"user{i}")[:20]
        n2 = enriched[j].get('name', f"user{j}")[:20]
        print(f"  {n1:<20} <-> {n2:<20} = {sim:.4f}")

# Find similar users for first user
print(f"\nTop 5 similar users to '{enriched[0].get('name','user0')}':")
similar = find_similar_users(enriched[0], {'users': enriched, 'artists': dataset['artists'], 'events': dataset['events'], 'follows': dataset.get('follows', [])}, top_k=5)
for u_data, score in similar[:5]:
    print(f"  {u_data.get('name','?'):<30} similarity={score:.4f}")

print()
print("=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
