import os
import sys
import torch
import psycopg2
import uuid
from datetime import datetime
import string, random
from dotenv import load_dotenv

from config import get_config
from data.generate_sample_data import load_dataset
from models.graph_builder import HeterogeneousGraphBuilder
from models.gnn_model import RecommendationModel

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

DB_URL = os.environ.get("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL environment variable not set")

BASE_DIR = os.path.dirname(__file__)
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model.pt")
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "data", "sample_dataset", "rasaswadaya_large_dataset.pkl")
PILOT_DATASET_PATH = os.path.join(BASE_DIR, "data", "sample_dataset", "rasaswadaya_dataset.pkl")


def resolve_dataset_path() -> str:
    dataset_path = os.environ.get("RASASWADAYA_DATASET", DEFAULT_DATASET_PATH)
    if not os.path.exists(dataset_path):
        dataset_path = PILOT_DATASET_PATH
        print(f"[WARNING] Large dataset not found, using pilot dataset: {dataset_path}")
    print(f"[EXPORT] Loading dataset from: {dataset_path}")
    return dataset_path


def load_checkpoint_state():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Trained checkpoint not found at {CHECKPOINT_PATH}. "
            "Run train_continuous.py before exporting recommendations."
        )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    return checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint


def align_features_to_checkpoint(data, state_dict):
    for node_type in data.node_types:
        key = f"gnn.input_proj.{node_type}.weight"
        if key not in state_dict:
            continue
        expected_dim = state_dict[key].shape[1]
        current = data[node_type].x
        current_dim = current.size(1)
        if current_dim == expected_dim:
            continue
        if current_dim < expected_dim:
            padding = torch.zeros((current.size(0), expected_dim - current_dim), dtype=current.dtype)
            data[node_type].x = torch.cat([current, padding], dim=1)
        else:
            data[node_type].x = current[:, :expected_dim]
    return data


def get_existing_ids(cur, table_name):
    cur.execute(f'SELECT id FROM "{table_name}"')
    return {row[0] for row in cur.fetchall()}


def top_recommendations(user_emb, embeddings, valid_indices, top_k):
    if not valid_indices:
        return []
    index_tensor = torch.tensor(valid_indices, dtype=torch.long, device=embeddings.device)
    candidate_embeddings = embeddings.index_select(0, index_tensor)
    scores = torch.nn.functional.cosine_similarity(user_emb, candidate_embeddings)
    limit = min(top_k, len(valid_indices))
    top_scores, top_positions = torch.topk(scores, k=limit)
    return [(top_scores[i], valid_indices[top_positions[i].item()]) for i in range(limit)]

def export_recommendations_to_db():
    print("🚀 Starting Recommendation Export Process...")
    config = get_config()
    device = config.gnn.device
    
    dataset_path = resolve_dataset_path()
    dataset = load_dataset(dataset_path)
    
    graph_builder = HeterogeneousGraphBuilder(dataset)
    graph_builder.build_graph()
    state_dict = load_checkpoint_state()
    data = graph_builder.build_pyg_data()
    if data is None:
        print("❌ Error building graph.")
        return
    data = align_features_to_checkpoint(data, state_dict)
        
    data = data.to(device)
    
    model = RecommendationModel(
        hidden_channels=config.gnn.hidden_channels,
        out_channels=config.gnn.out_channels,
        metadata=data.metadata(),
        num_layers=config.gnn.num_layers
    ).to(device)
    
    with torch.no_grad():
        model(data.x_dict, data.edge_index_dict)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[EXPORT] Loaded trained checkpoint from {CHECKPOINT_PATH}")
    
    with torch.no_grad():
        z_dict = model(data.x_dict, data.edge_index_dict)
    
    user_embeddings = z_dict['user']
    artist_embeddings = z_dict['artist']
    event_embeddings = z_dict.get('event', None)
    
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    cur.execute("DELETE FROM \"Recommendation\"")
    print("🧹 Cleared old recommendations.")
    valid_user_ids = get_existing_ids(cur, "User")
    valid_artist_ids = get_existing_ids(cur, "Artist")
    valid_event_ids = get_existing_ids(cur, "Event")
    
    user_nodes = graph_builder.node_mappings['user']
    artist_nodes = graph_builder.node_mappings['artist']
    idx_to_artist = {v: k for k, v in artist_nodes.items()}
    valid_artist_indices = [idx for idx, artist_id in idx_to_artist.items() if artist_id in valid_artist_ids]
    
    event_nodes = graph_builder.node_mappings.get('event', {})
    idx_to_event = {v: k for k, v in event_nodes.items()}
    valid_event_indices = [idx for idx, event_id in idx_to_event.items() if event_id in valid_event_ids]
    valid_user_nodes = [(user_id, user_idx) for user_id, user_idx in user_nodes.items() if user_id in valid_user_ids]
    print(
        f"🔎 Valid DB IDs in graph: users={len(valid_user_nodes)}, "
        f"artists={len(valid_artist_indices)}, events={len(valid_event_indices)}"
    )
    
    tot_recs = 0
    for user_id, user_idx in valid_user_nodes:
        user_emb = user_embeddings[user_idx].unsqueeze(0)
        
        # 1. Artist Recommendations
        top_artist_recs = top_recommendations(user_emb, artist_embeddings, valid_artist_indices, 3)
        for score, artist_idx in top_artist_recs:
            artist_id = idx_to_artist[artist_idx]
            rec_id = "rec_" + "".join(random.choices(string.ascii_letters + string.digits, k=20))
            score_val = max(0.0, min(1.0, (float(score.item()) + 1) / 2))
            reason = "Based on your recent activity, we think you'll love this artist!"
            
            cur.execute("""
                INSERT INTO "Recommendation" (id, "userId", "recommendedType", "recommendedId", score, reason, "createdAt", "updatedAt")
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (rec_id, user_id, 'ARTIST', artist_id, score_val, reason, datetime.now(), datetime.now()))
            tot_recs += 1

        # 2. Event Recommendations
        if event_embeddings is not None and len(event_embeddings) > 0:
            top_event_recs = top_recommendations(user_emb, event_embeddings, valid_event_indices, 3)
            for score, event_idx in top_event_recs:
                event_id = idx_to_event[event_idx]
                rec_id = "rec_" + "".join(random.choices(string.ascii_letters + string.digits, k=20))
                score_val = max(0.0, min(1.0, (float(score.item()) + 1) / 2))
                reason = f"Matches your interest with {int(score_val*100)}% accuracy."
                
                cur.execute("""
                    INSERT INTO "Recommendation" (id, "userId", "recommendedType", "recommendedId", score, reason, "createdAt", "updatedAt")
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (rec_id, user_id, 'EVENT', event_id, score_val, reason, datetime.now(), datetime.now()))
                tot_recs += 1

    conn.commit()
    conn.close()
    
    print(f"✅ Successfully inserted {tot_recs} recommendations into database!")

if __name__ == "__main__":
    export_recommendations_to_db()
