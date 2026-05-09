"""
Real-time recommendation pipeline for users not yet present in the trained GNN graph.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np

from preference_mapping import build_preference_profile, normalise_token


@dataclass
class NewUserRecommendation:
    artist_id: str
    score: float
    reason: str
    layer: str


class NewUserPipeline:
    """Progressive cold-start recommender for preference-only and first-follow users."""

    def __init__(self, dataset: dict, encoder, graph_builder=None, model=None):
        self.dataset = dataset
        self.encoder = encoder
        self.graph_builder = graph_builder
        self.model = model
        self._artist_vecs: Dict[str, np.ndarray] = {}
        self._artist_data: Dict[str, dict] = {}
        self._event_data: Dict[str, dict] = {}
        self._user_vecs: Dict[str, tuple] = {}
        self._follows_map: Dict[str, Set[str]] = {}
        self._artist_followers: Dict[str, Set[str]] = {}
        self._build_cache()

    def _profile_to_metadata(self, profile: dict) -> dict:
        categories = []
        for key in ("categories", "art_interests", "art_forms"):
            value = profile.get(key) or []
            categories.extend(value if isinstance(value, list) else [value])

        interests = []
        for key in ("interests", "genres", "styles", "culture_preferences"):
            value = profile.get(key) or []
            interests.extend(value if isinstance(value, list) else [value])

        mapped = build_preference_profile(categories or ["music"], interests, profile.get("city", "colombo"))

        def unique_tokens(*groups):
            tokens = []
            for group in groups:
                for raw_value in group or []:
                    value = normalise_token(raw_value)
                    if value and value not in tokens:
                        tokens.append(value)
            return tokens

        return {
            "art_forms": unique_tokens(mapped.get("art_interests"), profile.get("art_interests"), profile.get("art_forms")) or ["music"],
            "genres": unique_tokens(mapped.get("genres"), profile.get("genres"), profile.get("styles")),
            "language": profile.get("language_preferences") or profile.get("language") or ["sinhala"],
            "style": unique_tokens(mapped.get("culture_preferences"), profile.get("culture_preferences"), profile.get("style")) or ["contemporary"],
            "mood_tags": unique_tokens(mapped.get("mood_preferences"), profile.get("mood_preferences"), profile.get("mood_tags")),
            "festivals": profile.get("festivals") or [],
            "city": mapped.get("city") or normalise_token(profile.get("city")) or "colombo",
            "match_terms": unique_tokens(mapped.get("match_terms"), profile.get("match_terms")),
        }

    def _encode_profile(self, profile: dict) -> np.ndarray:
        vector = self.encoder.encode_artist(self._profile_to_metadata(profile)).vector.astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / (norm + 1e-8) if norm > 1e-8 else vector

    def _metadata_values(self, item: dict) -> List[str]:
        values = []
        for key in (
            "art_forms",
            "genres",
            "styles",
            "style",
            "mood_tags",
            "moods",
            "festivals",
            "event_type",
            "city",
            "name",
            "venue",
        ):
            value = item.get(key)
            if isinstance(value, list):
                values.extend(value)
            elif value:
                values.append(value)
        return [normalise_token(value) for value in values if normalise_token(value)]

    def _metadata_match_score(self, profile: dict, item: dict, city_weight: float = 0.08) -> float:
        metadata = self._profile_to_metadata(profile)
        terms = set(self._metadata_values(metadata))
        terms.update(metadata.get("match_terms", []))
        terms = {normalise_token(term) for term in terms if normalise_token(term)}
        if not terms:
            return 0.0

        item_values = set(self._metadata_values(item))
        score = 0.0
        matched = set()
        for term in terms:
            for value in item_values:
                if term in matched:
                    continue
                if value == term or value in term or term in value:
                    score += 0.14 if value == term else 0.08
                    matched.add(term)

        profile_city = normalise_token(metadata.get("city"))
        item_city = normalise_token(item.get("city"))
        if profile_city and item_city and (profile_city == item_city or profile_city in item_city or item_city in profile_city):
            score += city_weight

        return min(1.0, score)

    def _build_cache(self):
        print("[NewUserPipeline] Building Cultural DNA vector cache...")
        start = time.time()

        self._artist_vecs = {}
        self._artist_data = {}
        for artist in self.dataset.get("artists", []):
            artist_id = artist.get("artist_id")
            if artist_id:
                self._artist_data[artist_id] = artist
                vector = self.encoder.encode_artist(artist).vector.astype(np.float32)
                norm = np.linalg.norm(vector)
                self._artist_vecs[artist_id] = vector / (norm + 1e-8) if norm > 1e-8 else vector

        self._event_data = {
            event.get("event_id"): event
            for event in self.dataset.get("events", [])
            if event.get("event_id")
        }

        self._user_vecs = {}
        for user in self.dataset.get("users", []):
            user_id = user.get("user_id")
            if user_id:
                self._user_vecs[user_id] = (self._encode_profile(user), user)

        self._follows_map = {}
        for follow in self.dataset.get("interactions", {}).get("follows", []):
            user_id = follow.get("user_id")
            artist_id = follow.get("artist_id")
            if user_id and artist_id:
                self._follows_map.setdefault(user_id, set()).add(artist_id)

        self._artist_followers = {}
        for user_id, artist_ids in self._follows_map.items():
            for artist_id in artist_ids:
                self._artist_followers.setdefault(artist_id, set()).add(user_id)

        elapsed = time.time() - start
        print(
            f"[NewUserPipeline] Cache built: {len(self._artist_vecs)} artists, "
            f"{len(self._user_vecs)} users, {elapsed:.2f}s"
        )

    def layer0_metadata_match(self, new_user_profile: dict, top_k: int = 10) -> List[NewUserRecommendation]:
        scores = []
        for artist_id, artist in self._artist_data.items():
            score = self._metadata_match_score(new_user_profile, artist, city_weight=0.05)
            if score > 0:
                scores.append((artist_id, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return [
            NewUserRecommendation(
                artist_id=artist_id,
                score=round(float(score), 4),
                reason=f"Matches your selected cultural preferences - {int(score * 100)}% metadata alignment",
                layer="preference_match",
            )
            for artist_id, score in scores[:top_k]
        ]

    def event_metadata_recommendations(self, new_user_profile: dict, top_k: int = 5) -> List[dict]:
        ART_FORMS = {"music", "dance", "film", "drama"}
        preferred_forms = (
            set(new_user_profile.get("art_interests") or new_user_profile.get("art_forms") or [])
            | set(new_user_profile.get("categories") or [])
        ) & ART_FORMS

        in_form: List[tuple] = []
        off_form: List[tuple] = []
        for event_id, event in self._event_data.items():
            score = self._metadata_match_score(new_user_profile, event, city_weight=0.18)
            if score <= 0:
                continue
            event_forms = set(event.get("art_forms") or event.get("art_interests") or []) & ART_FORMS
            if preferred_forms and event_forms and not (event_forms & preferred_forms):
                off_form.append((event_id, score))
            else:
                in_form.append((event_id, score))

        in_form.sort(key=lambda item: item[1], reverse=True)
        off_form.sort(key=lambda item: item[1], reverse=True)
        scores = in_form[:top_k]
        if len(scores) < 2:
            scores += off_form[: 2 - len(scores)]

        return [
            {
                "id": event_id,
                "event_id": event_id,
                "type": "EVENT",
                "recommendedId": event_id,
                "score": round(float(score), 4),
                "reason": f"Matched to your preferences and location - {int(score * 100)}% metadata alignment",
                "layer": "event_preference_match",
            }
            for event_id, score in scores[:top_k]
        ]

    def layer1_cultural_dna(
        self,
        new_user_profile: dict,
        top_k_artists: int = 10,
        top_k_users: int = 30,
    ) -> List[NewUserRecommendation]:
        new_vec = self._encode_profile(new_user_profile)
        if np.linalg.norm(new_vec) < 1e-8:
            return []

        similarities = []
        for user_id, (user_vec, user_data) in self._user_vecs.items():
            similarity = float(np.dot(new_vec, user_vec))
            if similarity > 0.1:
                similarities.append((user_id, similarity, user_data))

        similarities.sort(key=lambda item: item[1], reverse=True)
        top_similar = similarities[:top_k_users]

        if not top_similar:
            return self._direct_artist_dna_match(new_vec, top_k_artists, "Cultural DNA direct match")

        artist_scores: Dict[str, float] = {}
        for user_id, similarity, _ in top_similar:
            for artist_id in self._follows_map.get(user_id, set()):
                artist_scores[artist_id] = artist_scores.get(artist_id, 0.0) + similarity

        if not artist_scores:
            return self._direct_artist_dna_match(new_vec, top_k_artists, "Cultural DNA direct match")

        max_score = max(artist_scores.values())
        ranked = sorted(
            ((artist_id, score / max_score) for artist_id, score in artist_scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )

        results = []
        for artist_id, score in ranked[:top_k_artists]:
            drivers = sum(1 for user_id, _, _ in top_similar if artist_id in self._follows_map.get(user_id, set()))
            results.append(
                NewUserRecommendation(
                    artist_id=artist_id,
                    score=round(float(score), 4),
                    reason=(
                        f"Followed by {drivers} users with similar cultural taste - "
                        f"{int(score * 100)}% match to your preferences"
                    ),
                    layer="cultural_dna",
                )
            )
        return results

    def _direct_artist_dna_match(
        self, user_vec_n: np.ndarray, top_k: int, reason_prefix: str
    ) -> List[NewUserRecommendation]:
        scores = []
        for artist_id, artist_vec in self._artist_vecs.items():
            similarity = float(np.dot(user_vec_n, artist_vec))
            score = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            scores.append((artist_id, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return [
            NewUserRecommendation(
                artist_id=artist_id,
                score=round(float(score), 4),
                reason=f"{reason_prefix} - {int(score * 100)}% cultural alignment",
                layer="cultural_dna",
            )
            for artist_id, score in scores[:top_k]
        ]

    def layer2_follow_amplification(
        self,
        new_user_profile: dict,
        followed_artist_ids: List[str],
        already_following: Set[str],
        top_k: int = 10,
    ) -> List[NewUserRecommendation]:
        if not followed_artist_ids:
            return []

        new_vec = self._encode_profile(new_user_profile)
        if np.linalg.norm(new_vec) < 1e-8:
            return []

        co_follower_pool = set()
        for artist_id in followed_artist_ids:
            co_follower_pool.update(self._artist_followers.get(artist_id, set()))
        if not co_follower_pool:
            return []

        scored_co_followers = []
        for user_id in co_follower_pool:
            user_entry = self._user_vecs.get(user_id)
            if not user_entry:
                continue
            similarity = float(np.dot(new_vec, user_entry[0]))
            if similarity > 0.05:
                scored_co_followers.append((user_id, similarity))
        scored_co_followers.sort(key=lambda item: item[1], reverse=True)
        top_co_followers = scored_co_followers[:50]

        artist_scores: Dict[str, float] = {}
        for user_id, similarity in top_co_followers:
            for artist_id in self._follows_map.get(user_id, set()):
                if artist_id in already_following:
                    continue
                artist_scores[artist_id] = artist_scores.get(artist_id, 0.0) + similarity
        if not artist_scores:
            return []

        max_score = max(artist_scores.values())
        ranked = sorted(
            ((artist_id, score / max_score) for artist_id, score in artist_scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )

        results = []
        for artist_id, score in ranked[:top_k]:
            drivers = sum(1 for user_id, _ in top_co_followers if artist_id in self._follows_map.get(user_id, set()))
            results.append(
                NewUserRecommendation(
                    artist_id=artist_id,
                    score=round(float(score), 4),
                    reason=(
                        "Users who also follow your artists like this - "
                        f"recommended by {drivers} culturally similar listeners"
                    ),
                    layer="follow_signal",
                )
            )
        return results

    def layer3_inject_user_to_graph(
        self,
        followed_artist_ids: List[str],
        z_dict: dict,
    ) -> Optional[np.ndarray]:
        if not followed_artist_ids or "artist" not in z_dict or not self.graph_builder:
            return None

        artist_nodes = self.graph_builder.node_mappings.get("artist", {})
        artist_embeddings = []
        for artist_id in followed_artist_ids:
            index = artist_nodes.get(artist_id)
            if index is not None and index < z_dict["artist"].shape[0]:
                artist_embeddings.append(z_dict["artist"][index].detach().cpu().numpy())
        if not artist_embeddings:
            return None

        pseudo_user_emb = np.nan_to_num(
            np.mean(artist_embeddings, axis=0).astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        pseudo_user_emb = np.clip(pseudo_user_emb, -1e6, 1e6).astype(np.float64)
        norm = np.linalg.norm(pseudo_user_emb)
        return (pseudo_user_emb / (norm + 1e-8) if norm > 1e-8 else pseudo_user_emb).astype(np.float32)

    def gnn_recs_from_embedding(
        self,
        user_embedding: np.ndarray,
        z_dict: dict,
        already_following: Set[str],
        top_k: int = 10,
    ) -> List[NewUserRecommendation]:
        if not self.graph_builder or "artist" not in z_dict:
            return []

        artist_embs = np.nan_to_num(
            z_dict["artist"].detach().cpu().numpy().astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        artist_embs = np.clip(artist_embs, -1e6, 1e6).astype(np.float64)
        norms = np.linalg.norm(artist_embs, axis=1, keepdims=True)
        artist_embs = artist_embs / (norms + 1e-8)
        artist_embs = np.nan_to_num(artist_embs, nan=0.0, posinf=0.0, neginf=0.0)
        user_vec = np.nan_to_num(
            np.array(user_embedding, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        user_vec = np.clip(user_vec, -1e6, 1e6).astype(np.float64)
        user_norm = np.linalg.norm(user_vec)
        if user_norm > 1e-8:
            user_vec = user_vec / (user_norm + 1e-8)
        user_vec = np.nan_to_num(user_vec, nan=0.0, posinf=0.0, neginf=0.0)
        sims = np.sum(artist_embs * user_vec.reshape(1, -1), axis=1)
        scores = np.nan_to_num(np.clip((sims + 1.0) / 2.0, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)

        idx_to_artist = {idx: artist_id for artist_id, idx in self.graph_builder.node_mappings.get("artist", {}).items()}
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for index in ranked_indices:
            artist_id = idx_to_artist.get(int(index))
            if artist_id and artist_id not in already_following:
                score = float(scores[index])
                results.append(
                    NewUserRecommendation(
                        artist_id=artist_id,
                        score=round(score, 4),
                        reason=f"Recommended by AI based on your followed artists - {int(score * 100)}% match",
                        layer="graph_injection",
                    )
                )
            if len(results) >= top_k:
                break
        return results

    def get_recommendations_for_new_user(
        self,
        new_user_profile: dict,
        followed_artist_ids: List[str],
        z_dict: Optional[dict] = None,
        top_k: int = 10,
    ) -> dict:
        already_following = set(followed_artist_ids)
        all_scores: Dict[str, float] = {}
        all_reasons: Dict[str, str] = {}
        all_layers: Dict[str, str] = {}

        for rec in self.layer0_metadata_match(new_user_profile, top_k=top_k * 2):
            all_scores[rec.artist_id] = all_scores.get(rec.artist_id, 0.0) + 0.35 * rec.score
            all_reasons[rec.artist_id] = rec.reason
            all_layers[rec.artist_id] = rec.layer

        for rec in self.layer1_cultural_dna(new_user_profile, top_k_artists=top_k * 2):
            all_scores[rec.artist_id] = all_scores.get(rec.artist_id, 0.0) + 0.20 * rec.score
            all_reasons[rec.artist_id] = rec.reason
            all_layers[rec.artist_id] = rec.layer

        if followed_artist_ids:
            for rec in self.layer2_follow_amplification(new_user_profile, followed_artist_ids, already_following, top_k=top_k * 2):
                all_scores[rec.artist_id] = all_scores.get(rec.artist_id, 0.0) + 0.30 * rec.score
                all_reasons[rec.artist_id] = rec.reason
                all_layers[rec.artist_id] = rec.layer

        if followed_artist_ids and z_dict is not None:
            user_emb = self.layer3_inject_user_to_graph(followed_artist_ids, z_dict)
            if user_emb is not None:
                for rec in self.gnn_recs_from_embedding(user_emb, z_dict, already_following, top_k=top_k * 2):
                    all_scores[rec.artist_id] = all_scores.get(rec.artist_id, 0.0) + 0.50 * rec.score
                    all_reasons[rec.artist_id] = rec.reason
                    all_layers[rec.artist_id] = rec.layer

        for artist_id in already_following:
            all_scores.pop(artist_id, None)

        ranked = sorted(all_scores.items(), key=lambda item: item[1], reverse=True)
        artists = [
            {
                "id": artist_id,
                "artist_id": artist_id,
                "type": "ARTIST",
                "recommendedId": artist_id,
                "score": round(float(score), 4),
                "reason": all_reasons.get(artist_id, "Recommended for you"),
                "layer": all_layers.get(artist_id, "unknown"),
            }
            for artist_id, score in ranked[:top_k]
        ]
        events = self.event_metadata_recommendations(new_user_profile, top_k=5)
        return {
            "artists": artists,
            "events": events,
            "new_user": True,
            "layers_active": sorted({item["layer"] for item in [*artists, *events]}),
        }
