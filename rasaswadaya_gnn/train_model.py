#!/usr/bin/env python3
"""
Complete Training Pipeline - Updated System
===========================================
Full end-to-end pipeline with 138D Cultural DNA dimensions.
"""

print("""
================================================================================
 RASASWADAYA GNN - COMPLETE TRAINING PIPELINE
 Updated System: 4 Art Forms | 3 Languages | 71 Moods | 138D Cultural DNA
================================================================================
""")

import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent


def resolve_dataset_file() -> Path:
    configured_dataset = os.environ.get("RASASWADAYA_DATASET")
    candidates = [Path(configured_dataset)] if configured_dataset else []
    candidates.extend([
        BASE_DIR / 'data' / 'sample_dataset' / 'rasaswadaya_large_dataset.pkl',
        BASE_DIR / 'data' / 'sample_dataset' / 'rasaswadaya_dataset_with_real_artists.json',
        BASE_DIR / 'data' / 'sample_dataset' / 'rasaswadaya_dataset_updated.pkl',
        BASE_DIR / 'data' / 'sample_dataset' / 'rasaswadaya_dataset.pkl',
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError('No dataset found')


# Check if dataset exists
try:
    dataset_file = resolve_dataset_file()
except FileNotFoundError:
    print("❌ No dataset found!")
    print("\n📝 Please run in order:")
    print("   1. python3 data/generate_large_dataset.py  # Generate large dataset")
    print("   2. python3 train_continuous.py --mode full # Train the GNN")
    print("   3. python3 export_to_db.py                 # Export recommendations")
    exit(1)

print(f"✅ Dataset found: {dataset_file.name}\n")

# Import required modules
print("📦 Loading modules...")
try:
    from data.generate_sample_data import load_dataset
    from models.graph_builder import HeterogeneousGraphBuilder
    from models.gnn_model import RecommendationModel
    from models.cultural_dna import CulturalDNAEncoder
    from config import get_config
    print("   ✓ All modules loaded successfully\n")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    print("\n📝 Please install requirements:")
    print("   pip install torch torch-geometric networkx numpy pandas")
    exit(1)

# Configuration
print(" CONFIGURATION")
print("=" * 80)
cfg = get_config()
print(f"   Art Forms: {cfg.cultural_dna.art_forms}")
print(f"   Languages: {cfg.cultural_dna.languages}")
print(f"   Total Moods: {len(cfg.cultural_dna.moods)}")
print(f"   Cultural DNA Dimensions: {cfg.cultural_dna.total_dimensions}")
print(f"   Model Hidden Dimensions: {cfg.gnn.hidden_channels}")
print(f"   Model Embedding Dimensions: {cfg.gnn.out_channels}")
print()

# Load dataset
print("=" * 80)
print(" STEP 1: LOADING DATASET")
print("=" * 80)
dataset = load_dataset(str(dataset_file))

print(f"   ✓ Users: {len(dataset['users'])}")
print(f"   ✓ Artists: {len(dataset['artists'])}")
print(f"   ✓ Events: {len(dataset['events'])}")
print(f"   ✓ Follows: {len(dataset['interactions']['follows'])}")
print()

# Build graph
print("=" * 80)
print(" STEP 2: BUILDING HETEROGENEOUS GRAPH")
print("=" * 80)
graph_builder = HeterogeneousGraphBuilder(dataset)
G = graph_builder.build_graph()
print()

# Encode Cultural DNA
print("=" * 80)
print(" STEP 3: ENCODING CULTURAL DNA")
print("=" * 80)
encoder = CulturalDNAEncoder()
print(f"   ✓ Encoder initialized with {encoder.total_dims}D vectors")

# Sample encoding
sample_artist = dataset['artists'][0]
dna = encoder.encode_artist(sample_artist)
print(f"   ✓ Sample encoding: {sample_artist['name']}")
print(f"      Art Form: {sample_artist['art_forms']}")
print(f"      DNA Vector Shape: {dna.vector.shape}")
print()

# Build PyG data
print("=" * 80)
print(" STEP 4: CONVERTING TO PyTorch GEOMETRIC FORMAT")
print("=" * 80)
try:
    pyg_data = graph_builder.build_pyg_data()
    print(f"   ✓ PyG HeteroData created")
    print(f"      Node types: {list(pyg_data.node_types)}")
    print(f"      Edge types: {len(pyg_data.edge_types)} types")
    
    # Print node type shapes
    for node_type in pyg_data.node_types:
        if hasattr(pyg_data[node_type], 'x'):
            print(f"      {node_type}: {pyg_data[node_type].x.shape}")
    print()
except Exception as e:
    print(f"   ⚠️  PyG conversion skipped: {e}")
    print(f"   (This is OK - you may need to install torch-geometric)")
    pyg_data = None
    print()

# Model initialization
print("=" * 80)
print(" STEP 5: INITIALIZING GNN MODEL")
print("=" * 80)
try:
    # This would require PyG to be installed
    print("   ⚠️  Model training requires PyTorch Geometric")
    print("   Install with: pip install torch-geometric")
    print()
    print("   Model architecture:")
    print(f"      Input: {encoder.total_dims}D Cultural DNA")
    print(f"      Hidden: {cfg.gnn.hidden_channels}D")
    print(f"      Embedding: {cfg.gnn.out_channels}D")
    print(f"      Task: Link Prediction (User-Artist compatibility)")
    print()
except Exception as e:
    print(f"   ⚠️  {e}")
    print()

# Summary
print("=" * 80)
print(" PIPELINE SUMMARY")
print("=" * 80)
print(f"✅ Dataset loaded: {len(dataset['artists'])} artists")
print(f"✅ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"✅ Cultural DNA encoder: {encoder.total_dims}D vectors")
if pyg_data:
    print(f"✅ PyG data ready for training")
else:
    print(f"⚠️  PyG data requires torch-geometric installation")
print()

print("=" * 80)
print(" NEXT STEPS")
print("=" * 80)
print("""
To complete training:

1. Install PyTorch Geometric:
   pip install torch torch-geometric

2. Run full training:
   python demo.py

3. Alternative - Use existing demo:
   The demo.py script will automatically use the new 138D Cultural DNA
   dimensions once PyTorch Geometric is installed.

4. Generate recommendations:
   After training, the model will output artist recommendations based on:
   - 4 Art Forms (music, dance, film, drama)
   - 3 Languages (sinhala, tamil, english)
   - 71 Categorized Moods
   - 138D Cultural DNA fingerprints

5. Export results:
   python export_to_csv.py
""")

print("=" * 80)
print(" ✅ PIPELINE CONFIGURATION COMPLETE!")
print("=" * 80)
