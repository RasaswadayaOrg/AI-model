#!/usr/bin/env python3
"""Collect all pipeline stats without requiring PyTorch."""
import sys, json, types
sys.path.insert(0, '.')

# --- DATASET STATISTICS ---
print("=" * 60)
print("STEP 1: DATASET STATISTICS")
print("=" * 60)

with open('data/sample_dataset/rasaswadaya_dataset_with_real_artists.json') as f:
    dataset = json.load(f)

users   = dataset['users']
artists = dataset['artists']
events  = dataset['events']
follows = dataset['interactions']['follows']
attends = dataset['interactions']['attends']

real_artists  = [a for a in artists if a.get('is_real_artist', False)]
gen_artists   = [a for a in artists if not a.get('is_real_artist', False)]

print(f"Users:             {len(users)}")
print(f"Artists (total):   {len(artists)}")
print(f"  - Real artists:  {len(real_artists)}")
print(f"  - Synthetic:     {len(gen_artists)}")
print(f"Events:            {len(events)}")
print(f"Follows edges:     {len(follows)}")
print(f"Attends edges:     {len(attends)}")
print(f"Total interactions:{len(follows) + len(attends)}")
avg_follows = len(follows) / len(users)
avg_attends = len(attends) / len(users)
print(f"Avg follows/user:  {avg_follows:.2f}")
print(f"Avg attends/user:  {avg_attends:.2f}")

# Art form distribution
from collections import Counter
art_forms = Counter()
for a in artists:
    for af in (a.get('art_forms') or [a.get('art_form', 'unknown')]):
        art_forms[af] += 1
print("\nArtist art-form distribution:")
for k, v in art_forms.most_common():
    print(f"  {k}: {v}")

# --- GRAPH STATISTICS ---
print()
print("=" * 60)
print("STEP 2: GRAPH STATISTICS (NetworkX)")
print("=" * 60)

import networkx as nx
from collections import defaultdict

G = nx.Graph()

# Nodes
for u in users:   G.add_node(u['user_id'], node_type='user')
for a in artists: G.add_node(a['artist_id'], node_type='artist')
for e in events:  G.add_node(e['event_id'], node_type='event')

genres = set()
for a in artists: genres.update(a.get('genres', []))
for e in events:  genres.update(e.get('genres', []))
for g in genres:  G.add_node(f"G_{g}", node_type='genre')

regions = set()
for a in artists:
    if a.get('region'): regions.add(a['region'])
for e in events:
    if e.get('region'): regions.add(e['region'])
for r in regions: G.add_node(f"L_{r}", node_type='location')

# Edges
for f in follows:
    G.add_edge(f['user_id'], f['artist_id'], edge_type='follows')

for e in events:
    for aid in e.get('artist_ids', []):
        G.add_edge(aid, e['event_id'], edge_type='performs_at')

for a in artists:
    for g in a.get('genres', []):
        gid = f"G_{g}"
        if gid in G: G.add_edge(a['artist_id'], gid, edge_type='belongs_to')

for e in events:
    for g in e.get('genres', []):
        gid = f"G_{g}"
        if gid in G: G.add_edge(e['event_id'], gid, edge_type='features')

    if e.get('region'):
        rid = f"L_{e['region']}"
        if rid in G: G.add_edge(e['event_id'], rid, edge_type='held_at')

print(f"Total nodes:       {G.number_of_nodes()}")
node_type_counts = Counter(d['node_type'] for _, d in G.nodes(data=True))
for nt, cnt in sorted(node_type_counts.items()):
    print(f"  {nt}: {cnt}")
print(f"Total edges:       {G.number_of_edges()}")
edge_type_counts = Counter(d.get('edge_type','?') for _, _, d in G.edges(data=True))
for et, cnt in sorted(edge_type_counts.items()):
    print(f"  {et}: {cnt}")

# Degree stats
degrees = [d for _, d in G.degree()]
import statistics
print(f"\nDegree stats:")
print(f"  Mean degree:   {statistics.mean(degrees):.2f}")
print(f"  Median degree: {statistics.median(degrees):.2f}")
print(f"  Max degree:    {max(degrees)}")
print(f"  Min degree:    {min(degrees)}")

# Connected components
comps = list(nx.connected_components(G))
print(f"\nConnected components: {len(comps)}")
print(f"  Largest component:  {max(len(c) for c in comps)} nodes")

# --- CULTURAL DNA ---
print()
print("=" * 60)
print("STEP 3: CULTURAL DNA ENCODER")
print("=" * 60)

try:
    from models.cultural_dna import CulturalDNAEncoder
    encoder = CulturalDNAEncoder()
    print(f"DNA vector dimensions: {encoder.total_dims}")

    # Encode a sample of artists
    sample_scores = []
    print("\nSample encodings:")
    for a in artists[:5]:
        dna = encoder.encode_artist(a)
        print(f"  {a['name'][:40]:<40} shape={dna.vector.shape} norm={float((dna.vector**2).sum()**0.5):.4f}")

    # Compute artist-artist cosine similarities
    import numpy as np
    print("\nArtist cultural similarity (cosine, sample 5×5):")
    vecs = []
    names = []
    for a in artists[:5]:
        dna = encoder.encode_artist(a)
        v = dna.vector.astype(float)
        norm = (v**2).sum()**0.5
        vecs.append(v / (norm + 1e-8))
        names.append(a['name'][:20])
    
    for i, n in enumerate(names):
        row = [f"{float(vecs[i] @ vecs[j]):.3f}" for j in range(len(names))]
        print(f"  {n:<20} | {' '.join(row)}")
except Exception as ex:
    print(f"Cultural DNA error: {ex}")

print()
print("=" * 60)
print("DONE")
print("=" * 60)
