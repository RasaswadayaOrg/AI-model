from fastapi import FastAPI, HTTPException
import os
import torch
import uvicorn
from typing import Dict, Any, Optional

from config import get_config
from data.connect_db import fetch_data
from data.generate_sample_data import load_dataset
from models.cultural_dna import CulturalDNAEncoder
from models.graph_builder import HeterogeneousGraphBuilder
from models.gnn_model import RecommendationModel
from new_user_pipeline import NewUserPipeline

app = FastAPI(title="Rasaswadaya AI Recommendation API")

BASE_DIR = os.path.dirname(__file__)
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model.pt")
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "data", "sample_dataset", "rasaswadaya_large_dataset.pkl")
PILOT_DATASET_PATH = os.path.join(BASE_DIR, "data", "sample_dataset", "rasaswadaya_dataset.pkl")

# Global variables to store our loaded model and data
model = None
graph_builder = None
data = None
device = None
z_dict = None
dataset = None
new_user_pipeline = None

ART_FORMS = {"music", "dance", "film", "drama"}


def normalise_cosine_score(score: float) -> float:
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def preference_weighted_score(user_profile: dict, item: dict, graph_score: float, city_weight: float = 0.08):
    metadata_score = 0.0
    if new_user_pipeline is not None:
        metadata_score = new_user_pipeline._metadata_match_score(user_profile, item, city_weight=city_weight)

    preferred_forms = set(user_profile.get("art_interests") or user_profile.get("art_forms") or []) & ART_FORMS
    item_forms = set(item.get("art_forms") or item.get("art_interests") or []) & ART_FORMS

    if preferred_forms:
        if item_forms & preferred_forms:
            metadata_score = min(1.0, metadata_score + 0.45)
        elif item_forms:
            metadata_score *= 0.25
        else:
            metadata_score *= 0.2

    combined_score = (0.35 * graph_score) + (0.65 * metadata_score)
    if preferred_forms and item_forms and not (item_forms & preferred_forms):
        combined_score *= 0.4
    return max(0.0, min(1.0, combined_score)), metadata_score


def resolve_dataset_path() -> str:
    dataset_path = os.environ.get("RASASWADAYA_DATASET", DEFAULT_DATASET_PATH)
    if not os.path.exists(dataset_path):
        dataset_path = PILOT_DATASET_PATH
        print(f"[WARNING] Large dataset not found, using pilot dataset: {dataset_path}")
    print(f"[STARTUP] Loading dataset from: {dataset_path}")
    return dataset_path


def load_checkpoint_state() -> Dict[str, torch.Tensor]:
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Trained checkpoint not found at {CHECKPOINT_PATH}. "
            "Run train_continuous.py before starting the API."
        )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    return checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint


def align_features_to_checkpoint(pyg_data, state_dict: Dict[str, torch.Tensor]):
    for node_type in pyg_data.node_types:
        key = f"gnn.input_proj.{node_type}.weight"
        if key not in state_dict:
            continue
        expected_dim = state_dict[key].shape[1]
        current = pyg_data[node_type].x
        current_dim = current.size(1)
        if current_dim == expected_dim:
            continue
        if current_dim < expected_dim:
            padding = torch.zeros((current.size(0), expected_dim - current_dim), dtype=current.dtype, device=current.device)
            pyg_data[node_type].x = torch.cat([current, padding], dim=1)
        else:
            pyg_data[node_type].x = current[:, :expected_dim]
    return pyg_data


def build_pyg_graph(source_dataset: Dict[str, Any], state_dict: Optional[Dict[str, torch.Tensor]] = None):
    builder = HeterogeneousGraphBuilder(source_dataset)
    builder.build_graph()
    pyg_data = builder.build_pyg_data()
    if pyg_data is None:
        raise RuntimeError("Error building PyG data")
    if state_dict is not None:
        pyg_data = align_features_to_checkpoint(pyg_data, state_dict)
    return builder, pyg_data


def load_trained_weights(recommendation_model: RecommendationModel, pyg_data, runtime_device: str):
    state_dict = load_checkpoint_state()
    align_features_to_checkpoint(pyg_data, state_dict)
    with torch.no_grad():
        recommendation_model(pyg_data.x_dict, pyg_data.edge_index_dict)
    recommendation_model.load_state_dict(state_dict)
    recommendation_model.eval()
    print(f"[STARTUP] Loaded trained checkpoint from {CHECKPOINT_PATH}")
    return state_dict

@app.on_event("startup")
async def startup_event():
    global model, graph_builder, data, device, z_dict, dataset, new_user_pipeline
    print("🚀 Loading AI Model and Graph Data into Memory...")
    
    config = get_config()
    device = config.gnn.device
    
    # Load your dataset
    dataset_path = resolve_dataset_path()
    try:
        dataset = load_dataset(dataset_path)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Build Graph
    state_dict = load_checkpoint_state()
    try:
        graph_builder, data = build_pyg_graph(dataset, state_dict)
    except Exception as e:
        print(f"❌ {e}")
        return
        
    data = data.to(device)
    
    # Initialize and load the model
    model = RecommendationModel(
        hidden_channels=config.gnn.hidden_channels,
        out_channels=config.gnn.out_channels,
        metadata=data.metadata(),
        num_layers=config.gnn.num_layers
    ).to(device)
    load_trained_weights(model, data, device)
    
    # Pre-compute embeddings on startup to make requests lightning fast
    with torch.no_grad():
        z_dict = model(data.x_dict, data.edge_index_dict)

    new_user_pipeline = NewUserPipeline(
        dataset=dataset,
        encoder=CulturalDNAEncoder(),
        graph_builder=graph_builder,
        model=model,
    )
    print("[STARTUP] GNN embeddings pre-computed for graph injection")
        
    print("✅ AI Model successfully loaded and ready for real-time requests!")


@app.post("/refresh")
async def refresh_graph():
    global dataset, graph_builder, data, z_dict, new_user_pipeline
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading or failed to load")
    try:
        dataset = fetch_data()
        state_dict = model.state_dict()
        graph_builder, data = build_pyg_graph(dataset, state_dict)
        data = data.to(device)
        model.eval()
        with torch.no_grad():
            z_dict = model(data.x_dict, data.edge_index_dict)
        new_user_pipeline = NewUserPipeline(
            dataset=dataset,
            encoder=CulturalDNAEncoder(),
            graph_builder=graph_builder,
            model=model,
        )
        return {
            "status": "refreshed",
            "users": len(dataset.get("users", [])),
            "artists": len(dataset.get("artists", [])),
            "events": len(dataset.get("events", [])),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/{user_id}")
async def get_recommendations(user_id: str):
    if model is None or z_dict is None or new_user_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is still loading or failed to load")
        
    user_nodes = graph_builder.node_mappings.get('user', {})
    
    if user_id not in user_nodes:
        print(f"[RECOMMEND] User {user_id} not in graph - activating new user pipeline")
        user_profile = next(
            (user for user in dataset.get("users", []) if user.get("user_id") == user_id),
            None,
        )
        if user_profile is None:
            try:
                from data.connect_db import fetch_single_user
                user_profile = fetch_single_user(user_id)
            except Exception:
                user_profile = None

        if user_profile is None:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found in dataset or database")

        followed_artist_ids = [
            follow.get("artist_id", "")
            for follow in dataset.get("interactions", {}).get("follows", [])
            if follow.get("user_id") == user_id and follow.get("artist_id")
        ]
        if not followed_artist_ids:
            try:
                from data.connect_db import fetch_user_followed_artists
                followed_artist_ids = fetch_user_followed_artists(user_id)
            except Exception as exc:
                print(f"[RECOMMEND] Could not load live follows for {user_id}: {exc}")
        result = new_user_pipeline.get_recommendations_for_new_user(
            new_user_profile=user_profile,
            followed_artist_ids=followed_artist_ids,
            z_dict=z_dict,
        )
        result["user_id"] = user_id
        result["recommendations"] = result["artists"]
        result["grouped"] = {"artists": result["artists"], "events": result["events"]}
        return result
        
    user_idx = user_nodes[user_id]
    user_profile = next(
        (user for user in dataset.get("users", []) if user.get("user_id") == user_id),
        {},
    )
    user_emb = z_dict['user'][user_idx].unsqueeze(0)
    
    # 1. Get Artist Recommendations
    artist_nodes = graph_builder.node_mappings.get('artist', {})
    idx_to_artist = {v: k for k, v in artist_nodes.items()}
    artist_embeddings = z_dict['artist']
    
    artist_scores = torch.nn.functional.cosine_similarity(user_emb, artist_embeddings)
    user_follows = {
        follow.get("artist_id")
        for follow in dataset.get("interactions", {}).get("follows", [])
        if follow.get("user_id") == user_id
    }
    artist_lookup = {artist.get("artist_id"): artist for artist in dataset.get("artists", [])}
    artist_candidates = []
    for idx in range(len(artist_embeddings)):
        artist_id = idx_to_artist[idx]
        if artist_id in user_follows:
            continue
        graph_score = normalise_cosine_score(float(artist_scores[idx].item()))
        artist = artist_lookup.get(artist_id, {})
        score_val, metadata_score = preference_weighted_score(user_profile, artist, graph_score, city_weight=0.05)
        artist_candidates.append((score_val, metadata_score, artist_id))

    artist_candidates.sort(key=lambda item: item[0], reverse=True)
    recommended_artists = []
    for score_val, metadata_score, artist_id in artist_candidates[:10]:
        match_percentage = int(score_val * 100)
        recommended_artists.append({
            "id": artist_id,
            "type": "ARTIST",
            "recommendedId": artist_id,
            "artist_id": artist_id,
            "score": round(score_val, 4),
            "semantic_score": round(metadata_score, 4),
            "match_percentage": match_percentage,
            "reason": f"{match_percentage}% preference-weighted cultural graph artist match"
        })
        
    # 2. Get Event Recommendations
    recommended_events = []
    event_nodes = graph_builder.node_mappings.get('event', {})
    event_embeddings = z_dict.get('event', None)
    
    if event_embeddings is not None and len(event_embeddings) > 0:
        idx_to_event = {v: k for k, v in event_nodes.items()}
        event_lookup = {event.get("event_id"): event for event in dataset.get("events", [])}
        event_scores = torch.nn.functional.cosine_similarity(user_emb, event_embeddings)

        # Hard category filter: never recommend off-art-form events when the user
        # has explicit preferences. We split candidates into in-form / off-form
        # and only fall back to off-form if there aren't enough in-form matches.
        preferred_forms = set(user_profile.get("art_interests") or user_profile.get("art_forms") or []) & ART_FORMS
        in_form_candidates = []
        off_form_candidates = []
        for idx in range(len(event_embeddings)):
            event_id = idx_to_event[idx]
            graph_score = normalise_cosine_score(float(event_scores[idx].item()))
            event = event_lookup.get(event_id, {})
            score_val, metadata_score = preference_weighted_score(user_profile, event, graph_score, city_weight=0.18)
            event_forms = set(event.get("art_forms") or event.get("art_interests") or []) & ART_FORMS
            if preferred_forms and event_forms and not (event_forms & preferred_forms):
                off_form_candidates.append((score_val, metadata_score, event_id))
            else:
                in_form_candidates.append((score_val, metadata_score, event_id))

        in_form_candidates.sort(key=lambda item: item[0], reverse=True)
        off_form_candidates.sort(key=lambda item: item[0], reverse=True)
        # Top 5 from in-form; only top up from off-form if the user would
        # otherwise see fewer than 2 results.
        event_candidates = in_form_candidates[:5]
        if len(event_candidates) < 2:
            event_candidates += off_form_candidates[: 2 - len(event_candidates)]

        for score_val, metadata_score, event_id in event_candidates[:5]:
            match_percentage = int(score_val * 100)
            recommended_events.append({
                "id": event_id,
                "type": "EVENT",
                "recommendedId": event_id,
                "event_id": event_id,
                "score": round(score_val, 4),
                "semantic_score": round(metadata_score, 4),
                "match_percentage": match_percentage,
                "reason": f"{match_percentage}% preference-weighted cultural graph event match"
            })

    flat_recommendations = sorted(
        [*recommended_artists, *recommended_events],
        key=lambda rec: rec["score"],
        reverse=True
    )

    return {
        "user_id": user_id,
        "artists": recommended_artists,
        "events": recommended_events,
        "new_user": False,
        "recommendations": flat_recommendations,
        "grouped": {
            "artists": recommended_artists,
            "events": recommended_events
        }
    }


@app.post("/recommend/new_user")
async def recommend_new_user(payload: Dict[str, Any]):
    if new_user_pipeline is None:
        raise HTTPException(status_code=503, detail="New user pipeline is still loading")

    profile = payload.get("profile") or {}
    followed_artist_ids = payload.get("followed_artist_ids") or []
    result = new_user_pipeline.get_recommendations_for_new_user(
        new_user_profile=profile,
        followed_artist_ids=followed_artist_ids,
        z_dict=z_dict,
    )
    result["user_id"] = profile.get("user_id")
    result["recommendations"] = result["artists"]
    result["grouped"] = {"artists": result["artists"], "events": result["events"]}
    return result

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
