"""
Heterogeneous Graph Construction Module
=======================================
Builds a multi-relational graph from Rasaswadaya.lk platform data.
"""

import networkx as nx
import numpy as np
import torch
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from models.cultural_dna import CulturalDNAEncoder


class HeterogeneousGraphBuilder:
    """
    Constructs a heterogeneous graph with multiple node and edge types.
    
    Node types: User, Artist, Event, Genre, Location
    Edge types: follows, performs_at, belongs_to, features, held_at
    (Event recommendations via 2-hop path: User->Artist->Event, not direct edges)
    """
    
    def __init__(self, dataset: Dict[str, Any]):
        """
        Args:
            dataset: Dictionary with 'users', 'artists', 'events', 'interactions'
        """
        self.dataset = dataset
        self.users = {u['user_id']: u for u in dataset['users']}
        self.artists = {a['artist_id']: a for a in dataset['artists']}
        self.events = {e['event_id']: e for e in dataset['events']}
        
        # Initialize Cultural DNA encoder
        self.dna_encoder = CulturalDNAEncoder()
        
        # Node ID mappings
        self.node_mappings = {}
        self.reverse_mappings = {}
        
        # Graph structures
        self.graph = None
        self.pyg_data = None

    def build_user_features(self, users: List[Dict[str, Any]], follows_map: Dict[str, List[str]]):
        artist_vectors = {}
        for artist_id, artist in self.artists.items():
            artist_vector = self.dna_encoder.encode_artist(artist).vector.astype(np.float32)
            artist_norm = np.linalg.norm(artist_vector)
            artist_vectors[artist_id] = artist_vector / (artist_norm + 1e-8)

        features = []
        for user in users:
            user_id = user.get('user_id', '')

            preference_profile = {
                'art_forms': user.get('art_interests') or user.get('art_forms') or [],
                'genres': user.get('genres') or [],
                'language': user.get('language_preferences') or user.get('language') or ['sinhala'],
                'style': user.get('culture_preferences') or user.get('styles') or user.get('style') or [],
                'mood_tags': user.get('mood_preferences') or user.get('mood_tags') or [],
                'festivals': user.get('festivals') or [],
            }
            preference_vector = self.dna_encoder.encode_artist(preference_profile).vector.astype(np.float32)
            preference_norm = np.linalg.norm(preference_vector)
            has_preferences = preference_norm > 1e-6
            if has_preferences:
                preference_vector = preference_vector / preference_norm
            else:
                preference_vector = np.zeros_like(preference_vector)

            followed_artist_ids = follows_map.get(user_id, [])
            followed_vectors = [artist_vectors[artist_id] for artist_id in followed_artist_ids if artist_id in artist_vectors]
            if followed_vectors:
                interaction_vector = np.mean(followed_vectors, axis=0)
                interaction_norm = np.linalg.norm(interaction_vector)
                if interaction_norm > 1e-6:
                    interaction_vector = interaction_vector / interaction_norm
                else:
                    interaction_vector = np.zeros_like(preference_vector)
                has_interactions = True
            else:
                interaction_vector = np.zeros_like(preference_vector)
                has_interactions = False

            if has_preferences and has_interactions:
                blended_vector = 0.5 * preference_vector + 0.5 * interaction_vector
            elif has_preferences:
                blended_vector = preference_vector
            elif has_interactions:
                blended_vector = interaction_vector
            else:
                blended_vector = np.zeros(self.dna_encoder.total_dims, dtype=np.float32)

            blended_norm = np.linalg.norm(blended_vector)
            if blended_norm > 1e-6:
                blended_vector = blended_vector / blended_norm
            features.append(blended_vector.astype(np.float32))

        if not features:
            return torch.empty((0, self.dna_encoder.total_dims), dtype=torch.float32)
        return torch.tensor(np.array(features), dtype=torch.float32)
        
    def build_graph(self) -> nx.Graph:
        """Build NetworkX graph for analysis and community detection."""
        print("\n🔨 Building Heterogeneous Graph...")
        print("=" * 60)
        
        G = nx.Graph()
        
        # Add nodes with attributes
        print("📌 Adding nodes...")
        
        # Users
        for user_id, user in self.users.items():
            G.add_node(user_id, 
                      node_type='user',
                      **user)
        print(f"  ✓ Added {len(self.users)} user nodes")
        
        # Artists
        for artist_id, artist in self.artists.items():
            G.add_node(artist_id,
                      node_type='artist',
                      **artist)
        print(f"  ✓ Added {len(self.artists)} artist nodes")
        
        # Events
        for event_id, event in self.events.items():
            G.add_node(event_id,
                      node_type='event',
                      **event)
        print(f"  ✓ Added {len(self.events)} event nodes")
        
        # Genre nodes (extracted from artists/events)
        genres = set()
        for artist in self.artists.values():
            genres.update(artist.get('genres', []))
        for event in self.events.values():
            genres.update(event.get('genres', []))
        
        for genre in genres:
            genre_id = f"G_{genre}"
            G.add_node(genre_id,
                      node_type='genre',
                      name=genre)
        print(f"  ✓ Added {len(genres)} genre nodes")
        
        # Location nodes (regions)
        regions = set()
        for artist in self.artists.values():
            if artist.get('region'):
                regions.add(artist['region'])
        for event in self.events.values():
            if event.get('region'):
                regions.add(event['region'])
        
        for region in regions:
            region_id = f"L_{region}"
            G.add_node(region_id,
                      node_type='location',
                      name=region)
        print(f"  ✓ Added {len(regions)} location nodes")
        
        # Add edges
        print("\n🔗 Adding edges...")
        
        # User -> Artist (follows)
        follows = self.dataset['interactions']['follows']
        for follow in follows:
            G.add_edge(follow['user_id'], follow['artist_id'],
                      edge_type='follows',
                      timestamp=follow.get('timestamp'))
        print(f"  ✓ Added {len(follows)} follow edges")
        
        # NOTE: User -> Event (attends) edges are intentionally removed.
        # Events are recommended via: User -> Artist -> Event (path-based)
        # This prevents data leakage and simulates real scenario where attendance
        # is not known in advance. Attendance data is kept in dataset for evaluation.
        
        # Artist -> Event (performs_at)
        for event_id, event in self.events.items():
            for artist_id in event.get('artist_ids', []):
                G.add_edge(artist_id, event_id,
                          edge_type='performs_at')
        
        # Artist -> Genre (belongs_to)
        for artist_id, artist in self.artists.items():
            for genre in artist.get('genres', []):
                genre_id = f"G_{genre}"
                if genre_id in G:
                    G.add_edge(artist_id, genre_id,
                              edge_type='belongs_to')
        
        # Event -> Genre (features)
        # NOTE: Event genres are the UNION of all performing artists' genres
        # This is computed during data generation, ensuring semantic consistency
        for event_id, event in self.events.items():
            for genre in event.get('genres', []):
                genre_id = f"G_{genre}"
                if genre_id in G:
                    G.add_edge(event_id, genre_id,
                              edge_type='features')
        
        # Event -> Location (held_at)
        for event_id, event in self.events.items():
            if event.get('region'):
                region_id = f"L_{event['region']}"
                if region_id in G:
                    G.add_edge(event_id, region_id,
                              edge_type='held_at')
        
        print(f"\n✅ Graph built successfully!")
        print(f"   Total nodes: {G.number_of_nodes()}")
        print(f"   Total edges: {G.number_of_edges()}")
        
        self.graph = G
        return G
    
    def detect_communities(self, algorithm: str = 'louvain') -> Dict[str, int]:
        """
        Detect communities/clusters in the graph.
        
        Args:
            algorithm: 'louvain' or 'label_propagation'
        
        Returns:
            Dictionary mapping node_id -> community_id
        """
        if self.graph is None:
            self.build_graph()
        
        print(f"\n🔍 Detecting communities using {algorithm}...")
        
        if algorithm == 'louvain':
            try:
                import community as community_louvain
                communities = community_louvain.best_partition(self.graph)
            except ImportError:
                print("⚠️  python-louvain not installed, using label propagation instead")
                algorithm = 'label_propagation'
        
        if algorithm == 'label_propagation':
            from networkx.algorithms import community
            communities_gen = community.label_propagation_communities(self.graph)
            communities = {}
            for comm_id, comm_nodes in enumerate(communities_gen):
                for node in comm_nodes:
                    communities[node] = comm_id
        
        num_communities = len(set(communities.values()))
        print(f"✓ Found {num_communities} communities")
        
        # Analyze community composition
        community_types = defaultdict(lambda: defaultdict(int))
        for node_id, comm_id in communities.items():
            if node_id in self.graph.nodes:
                node_type = self.graph.nodes[node_id].get('node_type', 'unknown')
                community_types[comm_id][node_type] += 1
        
        print(f"\n📊 Community composition (top 5):")
        for comm_id in sorted(community_types.keys())[:5]:
            types = community_types[comm_id]
            print(f"  Community {comm_id}: {dict(types)}")
        
        return communities
    
    def build_pyg_data(self) -> Any:
        """
        Convert to PyTorch Geometric HeteroData format for GNN training.
        
        Returns:
            torch_geometric.data.HeteroData object
        """
        try:
            from torch_geometric.data import HeteroData
        except ImportError:
            print("❌ PyTorch Geometric not installed!")
            return None
        
        if self.graph is None:
            self.build_graph()
        
        print("\n🔄 Converting to PyTorch Geometric format...")
        
        data = HeteroData()
        
        # Create node ID mappings for each node type
        node_types = ['user', 'artist', 'event', 'genre', 'location']
        
        for node_type in node_types:
            nodes = [n for n, attr in self.graph.nodes(data=True) 
                    if attr.get('node_type') == node_type]
            self.node_mappings[node_type] = {node_id: idx for idx, node_id in enumerate(nodes)}
            self.reverse_mappings[node_type] = {idx: node_id for node_id, idx in self.node_mappings[node_type].items()}
        
        # Create node features using Cultural DNA encoding
        print("🧬 Encoding Cultural DNA features...")
        
        users_in_node_order = [self.users[user_id] for user_id in self.reverse_mappings['user'].values()]
        follows_map = defaultdict(list)
        for follow in self.dataset['interactions']['follows']:
            follows_map[follow['user_id']].append(follow['artist_id'])

        data['user'].x = self.build_user_features(users_in_node_order, follows_map)
        print(f"  ✓ User features: {data['user'].x.shape}")
        
        # Artists: encode cultural metadata
        artist_features = []
        for artist_id in self.reverse_mappings['artist'].values():
            artist = self.artists[artist_id]
            dna = self.dna_encoder.encode_artist(artist)
            artist_features.append(dna.vector)
        
        data['artist'].x = torch.tensor(np.array(artist_features), dtype=torch.float)
        print(f"  ✓ Artist features: {data['artist'].x.shape}")
        
        # Events: encode cultural metadata
        event_features = []
        for event_id in self.reverse_mappings['event'].values():
            event = self.events[event_id]
            dna = self.dna_encoder.encode_event(event)
            event_features.append(dna.vector)
        
        data['event'].x = torch.tensor(np.array(event_features), dtype=torch.float)
        print(f"  ✓ Event features: {data['event'].x.shape}")
        
        # Genres: simple one-hot encoding
        genre_features = torch.eye(len(self.node_mappings['genre']))
        data['genre'].x = genre_features
        print(f"  ✓ Genre features: {data['genre'].x.shape}")
        
        # Locations: simple one-hot encoding
        location_features = torch.eye(len(self.node_mappings['location']))
        data['location'].x = location_features
        print(f"  ✓ Location features: {data['location'].x.shape}")
        
        # Create edge indices for each edge type
        print("\n🔗 Creating edge indices...")
        
        edge_types_to_add = [
            ('user', 'follows', 'artist'),
            ('artist', 'performs_at', 'event'),
            ('artist', 'belongs_to', 'genre'),

            ('event', 'features', 'genre'),
            ('event', 'held_at', 'location'),
            # Reverse edges for message passing
            ('artist', 'followed_by', 'user'),
            ('event', 'attended_by', 'user'),
            ('event', 'hosts', 'artist'),
            ('genre', 'includes', 'artist'),
            ('location', 'home_of', 'artist'),
            ('genre', 'featured_in', 'event'),
            ('location', 'venue_for', 'event'),
        ]
        
        for src_type, edge_type, dst_type in edge_types_to_add:
            edge_list = []
            
            for u, v, attr in self.graph.edges(data=True):
                u_type = self.graph.nodes[u].get('node_type')
                v_type = self.graph.nodes[v].get('node_type')
                e_type = attr.get('edge_type')
                
                # Match forward edge
                if u_type == src_type and v_type == dst_type and e_type == edge_type:
                    src_idx = self.node_mappings[src_type][u]
                    dst_idx = self.node_mappings[dst_type][v]
                    edge_list.append([src_idx, dst_idx])
                
                # Match reverse edge
                reverse_edge_type = edge_type.replace('followed_by', 'follows').replace('hosts', 'performs_at').replace('includes', 'belongs_to').replace('featured_in', 'features').replace('venue_for', 'held_at')
                
                if v_type == src_type and u_type == dst_type and e_type == reverse_edge_type:
                    # Add reverse edge
                    src_idx = self.node_mappings[src_type][v]
                    dst_idx = self.node_mappings[dst_type][u]
                    edge_list.append([src_idx, dst_idx])
            
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                data[src_type, edge_type, dst_type].edge_index = edge_index
                print(f"  ✓ {src_type} -> {edge_type} -> {dst_type}: {edge_index.shape[1]} edges")
        
        self.pyg_data = data
        
        print("\n✅ PyTorch Geometric data created successfully!")
        return data
    
    def get_node_id(self, node_type: str, internal_idx: int) -> str:
        """Convert internal PyG index back to original node ID."""
        return self.reverse_mappings[node_type][internal_idx]
    
    def get_node_idx(self, node_type: str, node_id: str) -> int:
        """Convert original node ID to internal PyG index."""
        return self.node_mappings[node_type][node_id]


if __name__ == "__main__":
    # Demo
    from data.generate_sample_data import load_dataset, generate_sample_dataset
    import os
    
    # Generate or load dataset
    dataset_path = "data/sample_dataset/rasaswadaya_dataset.pkl"
    if not os.path.exists(dataset_path):
        print("Dataset not found, generating...")
        dataset = generate_sample_dataset()
    else:
        print("Loading existing dataset...")
        dataset = load_dataset(dataset_path)
    
    # Build graph
    builder = HeterogeneousGraphBuilder(dataset)
    G = builder.build_graph()
    
    # Detect communities
    communities = builder.detect_communities('louvain')
    
    # Convert to PyG
    pyg_data = builder.build_pyg_data()
    
    print("\n" + "=" * 60)
    print("✅ Graph construction demo complete!")
    print("=" * 60)
