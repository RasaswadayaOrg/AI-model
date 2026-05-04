from fastapi import FastAPI, HTTPException
import torch
import uvicorn
from typing import Dict, Any

from config import get_config
from data.generate_sample_data import load_dataset
from models.graph_builder import HeterogeneousGraphBuilder
from models.gnn_model import RecommendationModel

app = FastAPI(title="Rasaswadaya AI Recommendation API")

# Global variables to store our loaded model and data
model = None
graph_builder = None
data = None
device = None
z_dict = None

@app.on_event("startup")
async def startup_event():
    global model, graph_builder, data, device, z_dict
    print("🚀 Loading AI Model and Graph Data into Memory...")
    
    config = get_config()
    device = config.gnn.device
    
    # Load your dataset
    dataset_path = "data/sample_dataset/rasaswadaya_dataset.pkl"
    try:
        dataset = load_dataset(dataset_path)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Build Graph
    graph_builder = HeterogeneousGraphBuilder(dataset)
    graph_builder.build_graph()
    data = graph_builder.build_pyg_data()
    if data is None:
        print("❌ Error building PyG data.")
        return
        
    data = data.to(device)
    
    # Initialize and load the model
    model = RecommendationModel(
        hidden_channels=config.gnn.hidden_channels,
        out_channels=config.gnn.out_channels,
        metadata=data.metadata(),
        num_layers=config.gnn.num_layers
    ).to(device)
    
    model.eval()
    
    # Pre-compute embeddings on startup to make requests lightning fast
    with torch.no_grad():
        z_dict = model(data.x_dict, data.edge_index_dict)
        
    print("✅ AI Model successfully loaded and ready for real-time requests!")

@app.get("/recommend/{user_id}")
async def get_recommendations(user_id: str):
    if model is None or z_dict is None:
        raise HTTPException(status_code=503, detail="Model is still loading or failed to load")
        
    user_nodes = graph_builder.node_mappings.get('user', {})
    
    if user_id not in user_nodes:
        raise HTTPException(status_code=404, detail="User not found in the graph data")
        
    user_idx = user_nodes[user_id]
    user_emb = z_dict['user'][user_idx].unsqueeze(0)
    
    # 1. Get Artist Recommendations
    artist_nodes = graph_builder.node_mappings.get('artist', {})
    idx_to_artist = {v: k for k, v in artist_nodes.items()}
    artist_embeddings = z_dict['artist']
    
    artist_scores = torch.nn.functional.cosine_similarity(user_emb, artist_embeddings)
    top_k_artists = min(5, len(artist_embeddings)) # top 5 artists
    top_a_scores, top_a_indices = torch.topk(artist_scores, k=top_k_artists)
    
    recommended_artists = []
    for score, idx in zip(top_a_scores, top_a_indices):
        score_val = max(0.0, min(1.0, (float(score.item()) + 1) / 2))
        recommended_artists.append({
            "artist_id": idx_to_artist[idx.item()],
            "score": round(score_val, 4),
            "match_percentage": int(score_val * 100)
        })
        
    # 2. Get Event Recommendations
    recommended_events = []
    event_nodes = graph_builder.node_mappings.get('event', {})
    event_embeddings = z_dict.get('event', None)
    
    if event_embeddings is not None and len(event_embeddings) > 0:
        idx_to_event = {v: k for k, v in event_nodes.items()}
        event_scores = torch.nn.functional.cosine_similarity(user_emb, event_embeddings)
        top_k_events = min(5, len(event_embeddings)) # top 5 events
        top_e_scores, top_e_indices = torch.topk(event_scores, k=top_k_events)
        
        for score, idx in zip(top_e_scores, top_e_indices):
            score_val = max(0.0, min(1.0, (float(score.item()) + 1) / 2))
            recommended_events.append({
                "event_id": idx_to_event[idx.item()],
                "score": round(score_val, 4),
                "match_percentage": int(score_val * 100)
            })

    return {
        "user_id": user_id,
        "recommendations": {
            "artists": recommended_artists,
            "events": recommended_events
        }
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
