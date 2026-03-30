import os
import sys
import torch
import psycopg2
import uuid
from datetime import datetime
import string, random

from config import get_config
from data.generate_sample_data import load_dataset
from models.graph_builder import HeterogeneousGraphBuilder
from models.gnn_model import RecommendationModel, evaluate

DB_URL = "postgresql://postgres.gmyqbkladleqthcdpeja:rasaya1234rasaya@aws-1-ap-northeast-2.pooler.supabase.com:6543/postgres?sslmode=require"

def export_recommendations_to_db():
    print("🚀 Starting Recommendation Export Process...")
    config = get_config()
    device = config.gnn.device
    
    dataset_path = "data/sample_dataset/rasaswadaya_dataset.pkl"
    dataset = load_dataset(dataset_path)
    
    graph_builder = HeterogeneousGraphBuilder(dataset)
    graph_builder.build_graph()
    data = graph_builder.build_pyg_data()
    if data is None:
        print("❌ Error building graph.")
        return
        
    data = data.to(device)
    
    model = RecommendationModel(
        hidden_channels=config.gnn.hidden_channels,
        out_channels=config.gnn.out_channels,
        metadata=data.metadata(),
        num_layers=config.gnn.num_layers
    ).to(device)
    
    model.eval()
    
    with torch.no_grad():
        z_dict = model(data.x_dict, data.edge_index_dict)
    
    user_embeddings = z_dict['user']
    artist_embeddings = z_dict['artist']
    event_embeddings = z_dict.get('event', None)
    
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    cur.execute("DELETE FROM \"Recommendation\"")
    print("🧹 Cleared old recommendations.")
    
    user_nodes = graph_builder.node_mappings['user']
    artist_nodes = graph_builder.node_mappings['artist']
    idx_to_artist = {v: k for k, v in artist_nodes.items()}
    
    event_nodes = graph_builder.node_mappings.get('event', {})
    idx_to_event = {v: k for k, v in event_nodes.items()}
    
    tot_recs = 0
    for user_id, user_idx in user_nodes.items():
        user_emb = user_embeddings[user_idx].unsqueeze(0)
        
        # 1. Artist Recommendations
        scores = torch.nn.functional.cosine_similarity(user_emb, artist_embeddings)
        top_k = min(3, len(artist_embeddings))
        if top_k > 0:
            top_scores, top_indices = torch.topk(scores, k=top_k)
            for score, artist_idx in zip(top_scores, top_indices):
                artist_id = idx_to_artist[artist_idx.item()]
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
            event_scores = torch.nn.functional.cosine_similarity(user_emb, event_embeddings)
            top_k_e = min(3, len(event_embeddings))
            if top_k_e > 0:
                top_e_scores, top_e_indices = torch.topk(event_scores, k=top_k_e)
                for score, event_idx in zip(top_e_scores, top_e_indices):
                    event_id = idx_to_event[event_idx.item()]
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
