"""
GNN Core Model - GraphSAGE Implementation
=========================================
Graph Neural Network for learning node embeddings on cultural platform data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

try:
    from torch_geometric.nn import SAGEConv, GATConv, HeteroConv, Linear
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("⚠️  PyTorch Geometric not available. Please install: pip install torch-geometric")

from config import get_config


class RasaswadayaGNN(nn.Module):
    """
    Graph Neural Network for Rasaswadaya cultural recommendation.
    
    Architecture:
    - Input: Cultural DNA feature vectors (varies by node type)
    - 2-3 layers of GraphSAGE or GAT message passing
    - Output: Rich node embeddings for downstream tasks
    
    Supports heterogeneous graphs with multiple node and edge types.
    """
    
    def __init__(self, metadata: Tuple, hidden_channels: int = 64, out_channels: int = 32, num_layers: int = 2, model_type: str = "graphsage"):
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric is required. Install with: pip install torch-geometric")
        
        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.model_type = model_type
        
        # Input projections for different node types
        # Project all node types to same hidden dimension
        self.input_proj = nn.ModuleDict()
        for node_type in metadata[0]:
            # Will be set dynamically based on input features
            self.input_proj[node_type] = None
        
        # Message passing layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_ch = hidden_channels
            out_ch = hidden_channels if i < num_layers - 1 else out_channels
            
            if model_type == "graphsage":
                conv = HeteroConv({
                    edge_type: SAGEConv(in_ch, out_ch, aggr='mean')
                    for edge_type in metadata[1]
                }, aggr='sum')
            elif model_type == "gat":
                conv = HeteroConv({
                    edge_type: GATConv(in_ch, out_ch // 4, heads=4, concat=True, dropout=0.1)
                    for edge_type in metadata[1]
                }, aggr='sum')
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            self.convs.append(conv)
            
            # Layer normalization for each node type
            norm_dict = nn.ModuleDict()
            for node_type in metadata[0]:
                norm_dict[node_type] = nn.LayerNorm(out_ch)
            self.norms.append(norm_dict)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the GNN.
        
        Args:
            x_dict: Dictionary mapping node_type -> feature tensor
            edge_index_dict: Dictionary mapping edge_type -> edge_index tensor
        
        Returns:
            Dictionary mapping node_type -> embedding tensor
        """
        # Initialize input projections if not done
        for node_type, x in x_dict.items():
            if self.input_proj[node_type] is None:
                input_dim = x.size(1)
                self.input_proj[node_type] = Linear(input_dim, self.hidden_channels).to(x.device)
        
        # Project all inputs to hidden dimension
        h_dict = {
            node_type: F.relu(self.input_proj[node_type](x))
            for node_type, x in x_dict.items()
        }
        
        # Message passing layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_dict = conv(h_dict, edge_index_dict)
            
            # Apply normalization and activation
            h_dict = {
                node_type: norm[node_type](h)
                for node_type, h in h_dict.items()
            }
            
            if i < self.num_layers - 1:
                h_dict = {
                    node_type: F.relu(h)
                    for node_type, h in h_dict.items()
                }
                h_dict = {
                    node_type: self.dropout(h)
                    for node_type, h in h_dict.items()
                }
        
        return h_dict


class LinkPredictionHead(nn.Module):
    """
    Prediction head for link prediction tasks.
    Scores potential edges between nodes.
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, src_embedding: torch.Tensor, dst_embedding: torch.Tensor) -> torch.Tensor:
        """
        Predict link probability between source and destination nodes.
        
        Args:
            src_embedding: Source node embeddings [batch_size, embedding_dim]
            dst_embedding: Destination node embeddings [batch_size, embedding_dim]
        
        Returns:
            Link probabilities [batch_size]
        """
        combined = torch.cat([src_embedding, dst_embedding], dim=1)
        return self.predictor(combined).squeeze()


class RecommendationModel(nn.Module):
    """
    Complete recommendation model combining GNN + prediction heads.
    """
    
    def __init__(self, metadata: Tuple, hidden_channels: int = 64, out_channels: int = 32, num_layers: int = 2, model_type: str = "graphsage"):
        super().__init__()
        
        self.gnn = RasaswadayaGNN(
            metadata=metadata,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            model_type=model_type
        )
        
        # Prediction heads for different tasks
        self.link_predictor = LinkPredictionHead(out_channels)
    
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get node embeddings."""
        return self.gnn(x_dict, edge_index_dict)
    
    def predict_link(self, embeddings: Dict[str, torch.Tensor], src_type: str, src_idx: torch.Tensor, dst_type: str, dst_idx: torch.Tensor) -> torch.Tensor:
        """
        Predict link probability between nodes.
        
        Args:
            embeddings: Node embeddings from GNN
            src_type: Source node type
            src_idx: Source node indices
            dst_type: Destination node type
            dst_idx: Destination node indices
        
        Returns:
            Link probabilities
        """
        src_emb = embeddings[src_type][src_idx]
        dst_emb = embeddings[dst_type][dst_idx]
        return self.link_predictor(src_emb, dst_emb)
    
    def recommend(self, embeddings: Dict[str, torch.Tensor], user_idx: torch.Tensor, candidate_type: str, candidate_indices: torch.Tensor, top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate top-K recommendations for a user.
        
        Args:
            embeddings: Node embeddings from GNN
            user_idx: User node index
            candidate_type: Type of candidates ('artist' or 'event')
            candidate_indices: Indices of candidate nodes
            top_k: Number of recommendations
        
        Returns:
            (top_indices, top_scores) - Indices and scores of top-K candidates
        """
        user_emb = embeddings['user'][user_idx].squeeze(0).unsqueeze(0).expand(len(candidate_indices), -1)
        candidate_emb = embeddings[candidate_type][candidate_indices]
        
        scores = self.link_predictor(user_emb, candidate_emb)
        top_scores, top_idx = torch.topk(scores, min(top_k, len(scores)))
        top_indices = candidate_indices[top_idx]
        
        return top_indices, top_scores
    
    def recommend_events(self, embeddings: Dict[str, torch.Tensor], user_idx: int, graph_builder, candidate_event_indices: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recommend events using combined scoring (Option D).
        
        Combines:
        1. Path-based: User -> Artist -> Event (2-hop traversal)
        2. Genre-based: User -> Genre -> Event (shared genres)
        3. Embedding-based: cosine_similarity(User_emb, Event_emb)
        4. Location-based: proximity to user's preferred region
        
        Args:
            embeddings: Node embeddings from GNN
            user_idx: User node index
            graph_builder: HeterogeneousGraphBuilder instance
            candidate_event_indices: Indices of candidate events
            top_k: Number of recommendations
        
        Returns:
            (top_indices, combined_scores)
        """
        num_candidates = len(candidate_event_indices)
        combined_scores = torch.zeros(num_candidates, device=embeddings['user'].device)
        
        # Score 1: Embedding-based similarity
        user_emb = embeddings['user'][user_idx].unsqueeze(0)
        event_embs = embeddings['event'][candidate_event_indices]
        
        # Cosine similarity
        user_norm = torch.norm(user_emb, dim=1, keepdim=True)
        event_norm = torch.norm(event_embs, dim=1, keepdim=True)
        embedding_scores = torch.mm(user_emb, event_embs.t()).squeeze(0) / (user_norm.squeeze(0) + 1e-8) / (event_norm.squeeze(1) + 1e-8)
        embedding_scores = torch.clamp(embedding_scores, min=0, max=1)
        
        combined_scores += embedding_scores * 0.4  # Weight: 40%
        
        # Note: Path-based (2-hop) and genre-based scores would require
        # access to graph structure. These can be computed if needed.
        # For now, we use embedding similarity as primary signal.
        
        top_scores, top_idx = torch.topk(combined_scores, min(top_k, len(combined_scores)))
        top_indices = candidate_event_indices[top_idx]
        
        return top_indices, top_scores


def train_step(model: RecommendationModel, data, optimizer: torch.optim.Optimizer, pos_edges: Dict, neg_edges: Dict, device: str = 'cpu') -> float:
    """
    Single training step for link prediction.
    
    Args:
        model: Recommendation model
        data: PyG HeteroData
        optimizer: PyTorch optimizer
        pos_edges: Dictionary of positive edges for each edge type
        neg_edges: Dictionary of negative edges for each edge type
        device: Device to train on
    
    Returns:
        Training loss
    """
    model.train()
    optimizer.zero_grad()
    
    # Move data to device
    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    
    # Forward pass
    embeddings = model(x_dict, edge_index_dict)
    
    # Compute loss for each edge type
    total_loss = 0.0
    num_edge_types = 0
    
    for edge_type in pos_edges.keys():
        src_type, rel, dst_type = edge_type
        
        pos_src, pos_dst = pos_edges[edge_type]
        neg_src, neg_dst = neg_edges[edge_type]
        
        if len(pos_src) == 0:
            continue
        
        pos_src = pos_src.to(device)
        pos_dst = pos_dst.to(device)
        neg_src = neg_src.to(device)
        neg_dst = neg_dst.to(device)
        
        # Positive scores
        pos_scores = model.predict_link(embeddings, src_type, pos_src, dst_type, pos_dst)
        
        # Negative scores (corrupted-tail: same users, different artists)
        neg_scores = model.predict_link(embeddings, src_type, neg_src, dst_type, neg_dst)
        
        # BPR loss: maximise log sigmoid(score_pos - score_neg)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        total_loss += loss
        num_edge_types += 1
    
    if num_edge_types > 0:
        total_loss = total_loss / num_edge_types
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return total_loss.item()
    
    return 0.0


@torch.no_grad()
def evaluate(model: RecommendationModel, data, pos_edges: Dict, neg_edges: Dict, device: str = 'cpu') -> Tuple[float, float]:
    """
    Evaluate model on validation/test set.
    
    Returns:
        (loss, accuracy)
    """
    model.eval()
    
    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    
    embeddings = model(x_dict, edge_index_dict)
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_edge_types = 0
    
    for edge_type in pos_edges.keys():
        src_type, rel, dst_type = edge_type
        
        pos_src, pos_dst = pos_edges[edge_type]
        neg_src, neg_dst = neg_edges[edge_type]
        
        if len(pos_src) == 0:
            continue
        
        pos_src = pos_src.to(device)
        pos_dst = pos_dst.to(device)
        neg_src = neg_src.to(device)
        neg_dst = neg_dst.to(device)
        
        pos_scores = model.predict_link(embeddings, src_type, pos_src, dst_type, pos_dst)
        neg_scores = model.predict_link(embeddings, src_type, neg_src, dst_type, neg_dst)
        
        # BPR loss (consistent with training)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        total_loss += loss.item()
        num_edge_types += 1
        
        # Accuracy
        pos_pred = (pos_scores > 0.5).long()
        neg_pred = (neg_scores > 0.5).long()
        
        total_correct += pos_pred.sum().item() + (1 - neg_pred).sum().item()
        total_samples += pos_pred.numel() + neg_pred.numel()
    
    avg_loss = total_loss / num_edge_types if num_edge_types > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, accuracy


if __name__ == "__main__":
    print("=" * 60)
    print(" GNN MODEL MODULE")
    print("=" * 60)
    
    if TORCH_GEOMETRIC_AVAILABLE:
        print("\n✅ PyTorch Geometric is available")
        print("✅ Model classes defined successfully")
        print("\nAvailable classes:")
        print("  - RasaswadayaGNN: Core GNN model")
        print("  - LinkPredictionHead: Link prediction head")
        print("  - RecommendationModel: Complete recommendation system")
        print("  - train_step(): Training function")
        print("  - evaluate(): Evaluation function")
    else:
        print("\n❌ PyTorch Geometric not installed")
        print("Install with: pip install torch-geometric")
