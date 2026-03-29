"""
Backend connector - Enhanced Collaborative Filtering Recommendation System
Implements: Content-based + Collaborative + Discovery (like Facebook/Instagram/YouTube)
"""
import sys
from pathlib import Path
import json
import pickle
from datetime import datetime, timedelta
import random
from collections import defaultdict

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, parent_dir)


def load_model_and_data():
    """Load trained model and dataset (NEW REAL ARTISTS DATA)"""
    import pandas as pd
    
    # Try loading from new CSV files first
    try:
        csv_dir = Path(__file__).parent.parent.parent / 'data' / 'sample_dataset' / 'csv_export_updated_real'
        
        users_df = pd.read_csv(csv_dir / 'users.csv')
        artists_df = pd.read_csv(csv_dir / 'artists.csv')
        events_df = pd.read_csv(csv_dir / 'events.csv')
        follows_df = pd.read_csv(csv_dir / 'follows.csv')
        attends_df = pd.read_csv(csv_dir / 'attends.csv')
        
        dataset = {
            'users': users_df.to_dict('records'),
            'artists': artists_df.to_dict('records'),
            'events': events_df.to_dict('records'),
            'follows': follows_df.to_dict('records'),
            'attends': attends_df.to_dict('records')
        }
        return None, dataset, None
    except Exception as e:
        print(f"Error loading CSV dataset: {e}")
        # Fallback to pickle
        dataset_path = Path(__file__).parent.parent.parent / 'data' / 'sample_dataset' / 'rasaswadaya_dataset.pkl'
        
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # For now, we don't load the model - just use scoring functions
        return None, dataset, None


def calculate_user_similarity(user1_data, user2_data):
    """
    Calculate similarity between two users based on:
    - Shared interests (art forms)
    - Similar genres and moods
    - Similar artists followed
    - Geographic proximity
    
    Returns: similarity score 0-1
    (UPDATED - Handles multiple art_interests and genres)
    """
    import ast
    
    def parse_list(val):
        if isinstance(val, list):
            return set(val)
        if isinstance(val, str):
            try:
                return set(ast.literal_eval(val))
            except:
                return {val}
        return set()
    
    score = 0.0
    
    # Art Interest similarity (40%)
    interests1 = parse_list(user1_data.get('art_interests', user1_data.get('interests', [])))
    interests2 = parse_list(user2_data.get('art_interests', user2_data.get('interests', [])))
    if interests1 and interests2:
        interest_sim = len(interests1 & interests2) / len(interests1 | interests2)
        score += interest_sim * 0.4
    
    # Genre similarity (15%)
    genres1 = parse_list(user1_data.get('genres', []))
    genres2 = parse_list(user2_data.get('genres', []))
    if genres1 and genres2:
        genre_sim = len(genres1 & genres2) / len(genres1 | genres2)
        score += genre_sim * 0.15
    
    # Mood/vibe similarity (15%)
    moods1 = parse_list(user1_data.get('moods', []))
    moods2 = parse_list(user2_data.get('moods', []))
    if moods1 and moods2:
        mood_sim = len(moods1 & moods2) / len(moods1 | moods2)
        score += mood_sim * 0.15
    
    # Followed artists overlap (25%)
    follows1 = set(user1_data.get('follows', []))
    follows2 = set(user2_data.get('follows', []))
    if follows1 and follows2:
        follow_sim = len(follows1 & follows2) / len(follows1 | follows2)
        score += follow_sim * 0.25
    
    # City proximity boost (5%)
    if user1_data.get('city') == user2_data.get('city'):
        score += 0.05
    
    return min(score, 1.0)


def find_similar_users(user_data, dataset, top_k=10):
    """
    Find users most similar to the given user
    Returns: List of (user_dict, similarity_score) tuples
    (NEW - Updated for real artists CSV data)
    """
    import ast
    
    similar_users = []
    
    # Helper to parse list fields from CSV (string representations)
    def parse_list(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str) and val:
            try:
                return ast.literal_eval(val)
            except:
                return [val]
        return []
    
    # Build follows mapping from dataset
    follows_map = defaultdict(list)
    for follow in dataset.get('follows', []):
        user_id = follow.get('user_id')
        artist_id = follow.get('artist_id')
        if user_id and artist_id:
            follows_map[user_id].append(artist_id)
    
    # Validate input user_data has required fields
    if 'user_id' not in user_data or not user_data.get('follows'):
        print(f"WARNING: user_data missing 'follows' field. Collaborative filtering will return empty results.")
    
    for user in dataset['users']:
        user_id_val = user.get('user_id') if isinstance(user, dict) else user['user_id']
        if str(user_id_val) == str(user_data.get('user_id')):
            continue  # Skip self
        
        # Build comparable user data - PROPERLY PARSE CSV STRING FIELDS
        other_user_data = {
            'user_id': user_id_val,
            'name': user.get('name', ''),
            'city': user.get('city', ''),
            'art_interests': parse_list(user.get('art_interests', [])),  # Parse from CSV string
            'interests': parse_list(user.get('interests', [])),  # Parse from CSV string
            'moods': parse_list(user.get('moods', [])),  # Parse from CSV string
            'genres': parse_list(user.get('genres', [])),  # Parse from CSV string
            'follows': follows_map.get(user_id_val, [])
        }
        
        similarity = calculate_user_similarity(user_data, other_user_data)
        
        if similarity > 0.1:  # Only consider meaningful similarities
            similar_users.append((other_user_data, similarity))
    
    # Sort by similarity and return top K
    similar_users.sort(key=lambda x: x[1], reverse=True)
    return similar_users[:top_k]


def get_collaborative_artist_recommendations(user_data, dataset, similar_users):
    """
    Get artist recommendations based on what similar users follow
    (Collaborative Filtering - NEW - Updated for CSV data)
    """
    import ast
    
    collaborative_scores = defaultdict(float)
    user_follows = set(user_data.get('follows', []))
    
    # Aggregate what similar users follow
    for similar_user, similarity_score in similar_users:
        for artist_id in similar_user['follows']:
            if artist_id not in user_follows:
                collaborative_scores[artist_id] += similarity_score
    
    # Helper to parse list fields from CSV
    def parse_list(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except:
                return [str(val)] if val else []
        return []
    
    # Convert to artist recommendations
    recommendations = []
    for artist in dataset['artists']:
        artist_id = artist.get('artist_id')
        if artist_id in collaborative_scores:
            art_form = artist.get('art_form', 'music')
            art_forms = parse_list(artist.get('art_forms', [art_form]))
            genres = parse_list(artist.get('genres', []))
            styles = parse_list(artist.get('styles', []))
            
            recommendations.append({
                'id': artist_id,
                'name': artist.get('name', ''),
                'art_form': art_form,
                'art_forms': art_forms,
                'genres': genres,
                'styles': styles,
                'genre': art_form,
                'city': artist.get('city', ''),
                'score': collaborative_scores[artist_id],
                'reason': 'collaborative'
            })
    
    return recommendations


def get_content_based_artist_recommendations(user_data, dataset):
    """
    Get artist recommendations based on user's own interests
    (Content-Based Filtering - NEW - Updated for CSV data)
    """
    import ast
    
    # Helper to parse list fields from CSV
    def parse_list(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except:
                return [str(val)] if val else []
        return []
    
    recommendations = []
    user_follows = set(user_data.get('follows', []))
    
    for artist in dataset['artists']:
        artist_id = artist.get('artist_id')
        if artist_id not in user_follows:
            score = calculate_artist_score(user_data, artist, dataset)
            art_form = artist.get('art_form', 'music')
            art_forms = parse_list(artist.get('art_forms', [art_form]))
            genres = parse_list(artist.get('genres', []))
            styles = parse_list(artist.get('styles', []))
            
            recommendations.append({
                'id': artist_id,
                'name': artist.get('name', ''),
                'art_form': art_form,
                'art_forms': art_forms,
                'genres': genres,
                'styles': styles,
                'genre': art_form,
                'city': artist.get('city', ''),
                'score': score,
                'reason': 'content'
            })
    
    return recommendations


def get_discovery_recommendations(user_data, dataset):
    """
    Get trending/popular items for discovery (exploration factor)
    Shows popular content even outside user's typical interests
    (NEW - Updated for real artists CSV data)
    """
    import ast
    
    # Helper to parse list fields from CSV
    def parse_list(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except:
                return [str(val)] if val else []
        return []
    
    # Get trending artists
    artist_follows = defaultdict(int)
    for follow in dataset.get('follows', []):
        # Handle new CSV format: artist_id instead of target
        artist_id = follow.get('artist_id')
        if artist_id:
            artist_follows[artist_id] += 1
    
    # Get top trending artists not already followed
    user_follows = set(user_data.get('follows', []))
    trending = []
    
    for artist in dataset['artists']:
        artist_id = artist.get('artist_id')
        if artist_id not in user_follows:
            popularity = artist_follows[artist_id]
            if popularity >= 0:  # Has some followers (>=0 to include new artists)
                art_form = artist.get('art_form', 'music')
                art_forms = parse_list(artist.get('art_forms', [art_form]))
                genres = parse_list(artist.get('genres', []))
                styles = parse_list(artist.get('styles', []))
                
                trending.append({
                    'id': artist_id,
                    'name': artist.get('name', ''),
                    'art_form': art_form,
                    'art_forms': art_forms,
                    'genres': genres,
                    'styles': styles,
                    'genre': art_form,
                    'city': artist.get('city', ''),
                    'score': min((popularity + 1) / 10.0, 1.0),  # Normalize
                    'reason': 'discovery'
                })
    
    return trending


def get_recommendations(user_data):
    """
    ENHANCED RECOMMENDATION SYSTEM
    Combines: Content-based + Collaborative + Discovery
    
    Mix: 50% Content-based + 30% Collaborative + 20% Discovery
    
    Args:
        user_data: dict with user info (name, city, interests, follows)
    
    Returns:
        dict with 'artists', 'events', and 'similar_users' lists
    """
    import ast
    
    try:
        model, dataset, device = load_model_and_data()
        
        # Helper to parse list fields from CSV
        def parse_list(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str) and val:
                try:
                    return ast.literal_eval(val)
                except:
                    return [val]
            return []
        
        # Get user ID from name
        user_id = None
        full_user_data = None
        for u in dataset['users']:
            if u['name'] == user_data['name']:
                user_id = u['user_id']
                full_user_data = u
                break
        
        if user_id is None:
            return {'artists': [], 'events': [], 'similar_users': []}
        
        # BUILD FOLLOWS MAPPING for user enrichment
        follows_map = defaultdict(list)
        for follow in dataset.get('follows', []):
            user_id_val = follow.get('user_id')
            artist_id = follow.get('artist_id')
            if user_id_val and artist_id:
                follows_map[user_id_val].append(artist_id)
        
        # ENRICH user_data with data from CSV and follows
        enriched_user_data = {
            'user_id': user_id,
            'name': full_user_data.get('name', ''),
            'city': full_user_data.get('city', ''),
            'art_interests': parse_list(full_user_data.get('art_interests', [])),
            'interests': parse_list(full_user_data.get('interests', [])),
            'moods': parse_list(full_user_data.get('moods', [])),
            'genres': parse_list(full_user_data.get('genres', [])),
            'follows': follows_map.get(user_id, [])
        }
        
        # Find similar users (Collaborative Filtering) - USE ENRICHED DATA
        similar_users = find_similar_users(enriched_user_data, dataset, top_k=10)
        
        # Get recommendations from different sources - USE ENRICHED DATA
        content_recs = get_content_based_artist_recommendations(enriched_user_data, dataset)
        collaborative_recs = get_collaborative_artist_recommendations(enriched_user_data, dataset, similar_users)
        discovery_recs = get_discovery_recommendations(enriched_user_data, dataset)
        
        # Mix recommendations: 50% content + 30% collaborative + 20% discovery
        content_recs.sort(key=lambda x: x['score'], reverse=True)
        collaborative_recs.sort(key=lambda x: x['score'], reverse=True)
        discovery_recs.sort(key=lambda x: x['score'], reverse=True)
        
        mixed_artists = []
        seen = set()
        
        # Add top content-based (50%)
        for rec in content_recs[:5]:
            if rec['id'] not in seen:
                mixed_artists.append(rec)
                seen.add(rec['id'])
        
        # Add collaborative (30%)
        for rec in collaborative_recs[:3]:
            if rec['id'] not in seen:
                mixed_artists.append(rec)
                seen.add(rec['id'])
        
        # Add discovery (20%)
        for rec in discovery_recs[:2]:
            if rec['id'] not in seen:
                mixed_artists.append(rec)
                seen.add(rec['id'])
        
        # Fill remaining slots with best remaining recommendations
        all_remaining = [r for r in content_recs + collaborative_recs if r['id'] not in seen]
        all_remaining.sort(key=lambda x: x['score'], reverse=True)
        
        for rec in all_remaining:
            if len(mixed_artists) >= 10:
                break
            if rec['id'] not in seen:
                mixed_artists.append(rec)
                seen.add(rec['id'])
        
        # Get event recommendations (collaborative-aware) - USE ENRICHED DATA
        recommended_events = get_event_recommendations(enriched_user_data, dataset, similar_users)
        
        return {
            'artists': mixed_artists[:10],
            'events': recommended_events[:10],
            'similar_users': [{'name': u[0]['name'], 'similarity': round(u[1], 2)} for u in similar_users[:5]]
        }
    
    except Exception as e:
        print(f"Error in get_recommendations: {e}")
        import traceback
        traceback.print_exc()
        return {'artists': [], 'events': [], 'similar_users': []}




def get_event_recommendations(user_data, dataset, similar_users):
    """
    Get event recommendations using collaborative filtering + LOCATION PRIORITY
    Scoring Formula: FINAL_SCORE = (location × 0.50) + (content × 0.30) + (collaborative × 0.10) + (popularity × 0.10)
    
    Considers: 
      - 50% Location proximity (nearest events first) - HIGHEST PRIORITY
      - 30% User interests & followed artists
      - 10% What similar users attend
      - 10% Trending/Popularity
    (Updated for multiple artists per event)
    """
    import ast
    
    event_scores = {}
    user_follows = set(user_data.get('follows', []))
    user_city = user_data.get('city', '')
    
    # Helper to parse artist_ids (handles string representation from CSV)
    def parse_artist_ids(artist_ids_data):
        if isinstance(artist_ids_data, list):
            return artist_ids_data
        if isinstance(artist_ids_data, str):
            try:
                return ast.literal_eval(artist_ids_data)
            except:
                return [artist_ids_data] if artist_ids_data else []
        return []
    
    # Get events user might attend
    for event in dataset['events']:
        score = 0.0
        
        # 1. LOCATION PROXIMITY (50%) - HIGHEST PRIORITY
        distance = calculate_distance(user_city, event.get('city', ''))
        # Closer = higher score. Max distance ~400km in SL, so normalize
        location_score = max(0, 1.0 - (distance / 400.0))  # 0 at 400km, 1 at same city
        score += location_score * 0.50
        
        # 2. CONTENT MATCHING (30%) - User interests and followed artists
        content_score = calculate_event_score(user_data, event, dataset)
        score += content_score * 0.30
        
        # 3. COLLABORATIVE SCORE (10%) - Similar users' interests
        event_artists = set(parse_artist_ids(event.get('artist_ids', [])))
        for similar_user, similarity in similar_users[:5]:
            similar_follows = set(similar_user.get('follows', []))
            artist_overlap = len(event_artists & similar_follows)
            if artist_overlap > 0:
                score += similarity * artist_overlap * 0.10
        
        # 4. POPULARITY BOOST (10%)
        # NEW: attends is directly in dataset, not under 'interactions'
        event_attendees = sum(1 for a in dataset.get('attends', []) if a.get('event_id') == event['event_id'])
        popularity_score = min(event_attendees / 50.0, 1.0)  # Normalize
        score += popularity_score * 0.10
        
        event_scores[event['event_id']] = {
            'id': event['event_id'],
            'name': event['name'],
            'date': event.get('date'),
            'city': event.get('city', ''),
            'art_form': event.get('art_form', 'music'),
            'num_artists': len(parse_artist_ids(event.get('artist_ids', []))),
            'distance': distance,
            'distance_label': f"{distance} km away",
            'score': min(score, 1.0)
        }
    
    # Sort by score (location proximity is now baked in)
    recommended_events = sorted(event_scores.values(), key=lambda x: x['score'], reverse=True)
    return recommended_events


def get_trending_data(user=None, days=30, personalized=True):
    """
    Get trending artists, events, and genres
    (Updated for multiple artists per event)
    
    Returns:
        dict with 'artists', 'events', and 'genres' lists
    """
    import ast
    
    def parse_artist_ids(artist_ids_data):
        """Helper to parse artist_ids from CSV string format"""
        if isinstance(artist_ids_data, list):
            return artist_ids_data
        if isinstance(artist_ids_data, str):
            try:
                return ast.literal_eval(artist_ids_data)
            except:
                return [artist_ids_data] if artist_ids_data else []
        return []
    
    try:
        _, dataset, _ = load_model_and_data()
        
        # Get follows data
        trending_artists = []
        artist_follows = {}
        genre_engagement = {}
        from datetime import datetime, timedelta
        now = datetime.utcnow()
        
        # Create artist lookup for efficiency
        artist_lookup = {artist['artist_id']: artist for artist in dataset.get('artists', [])}
        
        # NEW: follows is directly in dataset, not under 'interactions'
        for follow in dataset.get('follows', []):
            artist_id = follow.get('artist_id')
            if artist_id:
                artist_follows[artist_id] = artist_follows.get(artist_id, 0) + 1
            
            # Count art forms from followed artists (weight: 2 for follows)
            if artist_id in artist_lookup:
                artist = artist_lookup[artist_id]
                art_form = artist.get('art_form')
                if art_form:
                    genre_engagement[art_form] = genre_engagement.get(art_form, 0) + 2
        
        for artist in dataset['artists']:
            follows_count = artist_follows.get(artist['artist_id'], 0)
            trending_artists.append({
                'id': artist['artist_id'],
                'name': artist['name'],
                'follows': follows_count,
                'art_form': artist.get('art_form', 'music')
            })
        
        # Sort by follows
        trending_artists = sorted(trending_artists, key=lambda x: x['follows'], reverse=True)[:10]
        
        # Get trending events
        trending_events = []
        event_attendees = {}
        
        # Create event lookup for efficiency
        event_lookup = {event['event_id']: event for event in dataset.get('events', [])}
        
        # NEW: attends is directly in dataset, not under 'interactions'
        for attend in dataset.get('attends', []):
            event_id = attend.get('event_id')
            if event_id:
                event_attendees[event_id] = event_attendees.get(event_id, 0) + 1

            # Count art forms from event attendances (weight: 1 for attendances)
            if event_id in event_lookup:
                event = event_lookup[event_id]
                art_form = event.get('art_form')
                if art_form:
                    genre_engagement[art_form] = genre_engagement.get(art_form, 0) + 1
        
        for event in dataset['events']:
            attendees_count = event_attendees.get(event['event_id'], 0)
            event_artist_ids = parse_artist_ids(event.get('artist_ids', []))
            
            trending_events.append({
                'id': event['event_id'],
                'name': event['name'],
                'attendees': attendees_count,
                'date': event.get('date'),
                'art_form': event.get('art_form', 'music'),
                'num_artists': len(event_artist_ids),
                'artist_ids': event_artist_ids
            })
        
        # Sort by attendees (global ranking)
        trending_events_global = sorted(trending_events, key=lambda x: x['attendees'], reverse=True)[:10]

        # If personalization requested and a user provided, compute personalized event scores
        trending_events_personalized = []
        if personalized and user:
            # Determine user id and preferences
            user_id = user.get('user_id') if isinstance(user, dict) else user

            # Collect user's preferred art forms from several possible keys
            user_genres = set(user.get('interests', []) or user.get('art_interests', []) or user.get('moods', []))

            # Collect artist ids followed by this user (NEW: uses dataset.get('follows'))
            user_followed_artists = set([f['artist_id'] for f in dataset.get('follows', []) if f.get('user_id') == user_id])

            # Compute personalized score: base = attendees, + artist boost, + genre boost, apply time decay
            for ev in trending_events:
                base = ev['attendees']
                score = float(base)

                # Boost if event features an artist the user follows
                if user_followed_artists and set(ev.get('artist_ids', [])) & user_followed_artists:
                    score += 3.0

                # Boost by number of shared art forms (NEW: art_form instead of genres)
                event_art_form = ev.get('art_form')
                if user_genres and event_art_form and event_art_form in user_genres:
                    score += 2.0

                # Time decay: more recent events weigh more within the window
                try:
                    ev_date = None
                    if ev.get('date'):
                        ev_date = datetime.fromisoformat(ev['date'])
                    if ev_date:
                        days_since = (now - ev_date).days
                        decay = max(0.1, 1.0 - (days_since / float(days)))
                        score *= decay
                except Exception:
                    # if parsing fails, ignore decay
                    pass

                trending_events_personalized.append((ev, score))

            # sort by personalized score
            trending_events_personalized = [x[0] for x in sorted(trending_events_personalized, key=lambda t: t[1], reverse=True)][:10]

        # Choose output: personalized if available else global
        trending_events = trending_events_personalized if trending_events_personalized else trending_events_global
        
        # Get trending genres (NEW!)
        trending_genres = []
        for genre, engagement_score in sorted(genre_engagement.items(), key=lambda x: x[1], reverse=True)[:8]:
            trending_genres.append({
                'name': genre,
                'engagement': engagement_score
            })
        
        return {
            'artists': trending_artists,
            'events': trending_events,
            'events_global': trending_events_global,
            'genres': trending_genres
        }
    
    except Exception as e:
        print(f"Error in get_trending_data: {e}")
        import traceback
        traceback.print_exc()
        return {'artists': [], 'events': [], 'genres': []}


def calculate_artist_score(user_data, artist, dataset):
    """Calculate compatibility score between user and artist (UPDATED - Multiple interests)"""
    import ast
    
    def parse_list(val):
        if isinstance(val, list):
            return set(val)
        if isinstance(val, str):
            try:
                return set(ast.literal_eval(val))
            except:
                return {val}
        return set()
    
    score = 0.3
    
    # Art Form Matching (40%)
    user_interests = parse_list(user_data.get('art_interests', user_data.get('interests', [])))
    artist_art_form = artist.get('art_form', 'music')
    artist_genres = parse_list(artist.get('genres', []))
    
    # Check if user is interested in artist's art form
    if artist_art_form in user_interests:
        score += 0.4
    else:
        # Soft match: check if any user genre matches
        user_genres = parse_list(user_data.get('genres', []))
        if len(user_genres & artist_genres) > 0:
            score += 0.2  # Partial match
    
    # Genre Matching (30%)
    user_genres = parse_list(user_data.get('genres', []))
    shared_genres = len(user_genres & artist_genres)
    if shared_genres > 0:
        score += 0.3 * (shared_genres / max(len(user_genres), 1))
    
    # Mood Matching (20%)
    user_moods = parse_list(user_data.get('moods', []))
    artist_moods = parse_list(artist.get('mood_tags', []))
    shared_moods = len(user_moods & artist_moods)
    if shared_moods > 0:
        score += 0.2 * (shared_moods / max(len(user_moods), 1))
    
    # City proximity (5%)
    if user_data.get('city') == artist.get('city'):
        score += 0.05
    else:
        distance = calculate_distance(user_data.get('city', ''), artist.get('city', ''))
        score += 0.05 * max(0, 1 - distance / 300)  # 300km = 0 score
    
    # Popularity factor (10%)
    follows = artist.get('follower_count', 0)
    popularity = min(follows / 500000.0, 1.0)  # Normalize for high follower counts
    score += popularity * 0.1
    
    return min(score, 1.0)


def get_trending_data(user=None, days=30, personalized=True):
    """
    Get trending artists and events (NEW - CSV data)
    
    Returns:
        dict with 'artists' and 'events' lists
    """
    try:
        _, dataset, _ = load_model_and_data()
        
        # Get follows data
        trending_artists = []
        artist_follows = {}
        
        for follow in dataset.get('follows', []):
            # Handle CSV format: artist_id instead of target
            artist_id = follow.get('artist_id')
            if artist_id:
                artist_follows[artist_id] = artist_follows.get(artist_id, 0) + 1
        
        for artist in dataset['artists']:
            artist_id = artist.get('artist_id')
            follows_count = artist_follows.get(artist_id, 0)
            trending_artists.append({
                'id': artist_id,
                'name': artist.get('name', 'Unknown'),
                'follows': follows_count,
                'art_form': artist.get('art_form', 'music')
            })
        
        # Sort by follows
        trending_artists = sorted(trending_artists, key=lambda x: x['follows'], reverse=True)[:10]
        
        # Get trending events
        trending_events = []
        event_attendees = {}
        
        for attend in dataset.get('attends', []):
            event_id = attend.get('event_id')  # NEW: CSV format uses event_id, not target
            if event_id:
                event_attendees[event_id] = event_attendees.get(event_id, 0) + 1
        
        for event in dataset['events']:
            attendees_count = event_attendees.get(event['event_id'], 0)
            trending_events.append({
                'id': event['event_id'],
                'name': event['name'],
                'attendees': attendees_count,
                'date': event.get('date'),
                'art_form': event.get('art_form', 'music'),
                'num_artists': len(event.get('artist_ids', []))
            })
        
        # Sort by attendees
        trending_events = sorted(trending_events, key=lambda x: x['attendees'], reverse=True)[:10]
        
        return {
            'artists': trending_artists,
            'events': trending_events
        }
    
    except Exception as e:
        print(f"Error in get_trending_data: {e}")
        return {'artists': [], 'events': []}


def calculate_artist_score(user_data, artist, dataset):
    """Calculate compatibility score between user and artist"""
    score = random.uniform(0.3, 0.9)
    
    # Boost score if genres match
    user_interests = set(user_data.get('interests', []))
    artist_genres = set(artist.get('genres', []))
    
    if user_interests & artist_genres:
        score += 0.1
    
    # Boost score if city matches
    if user_data.get('city') == artist.get('city'):
        score += 0.05
    
    return min(score, 1.0)


def calculate_event_score(user_data, event, dataset):
    """Calculate event relevance score (NEW - CSV data with multiple artists)"""
    import ast
    
    score = 0.5
    
    # Distance scoring (40%)
    user_city = user_data.get('city', 'Colombo')
    event_city = event.get('city', 'Colombo')
    try:
        distance = calculate_distance(user_city, event_city)
        distance_score = max(0, 1 - (distance / 500))  # 500km = 0 score
    except:
        distance_score = 0.5  # Default if distance fails
    score += distance_score * 0.4
    
    # Artist scoring (35%) - Handles multiple artists per event
    user_follows = set(user_data.get('follows', []))
    event_artist_ids = event.get('artist_ids', [])
    
    # Parse artist_ids if it's a string (from CSV) or already a list
    if isinstance(event_artist_ids, str):
        try:
            event_artist_ids = ast.literal_eval(event_artist_ids)
        except:
            event_artist_ids = [event_artist_ids]
    elif not isinstance(event_artist_ids, list):
        event_artist_ids = [event_artist_ids]
    
    event_artists = set(event_artist_ids)
    
    if event_artists:
        artist_match = len(user_follows & event_artists) / len(event_artists)
        score += artist_match * 0.35
    
    # Art form scoring (25%)
    user_interests = set(user_data.get('interests', []))
    event_art_form = event.get('art_form', 'music')
    
    # Match against user interests or try to find matching artist in dataset
    art_form_match = 0.3  # Base score
    if event_art_form in user_interests or event_art_form.lower() in [i.lower() for i in user_interests]:
        art_form_match = 1.0
    
    score += art_form_match * 0.25
    
    return min(score, 1.0)


def calculate_distance(city1, city2):
    """Calculate distance between two cities"""
    # Haversine coordinates for Sri Lankan cities
    city_coords = {
        'colombo': (6.9271, 79.8612),
        'kandy': (7.2906, 80.6337),
        'galle': (6.0535, 80.2210),
        'jaffna': (9.6615, 80.0255),
        'ratnapura': (6.6872, 80.3903),
        'matara': (5.9466, 80.5395),
        'anuradhapura': (8.3142, 80.4137),
        'polonnaruwa': (7.9366, 80.9831),
        'trincomalee': (8.5874, 81.2346),
        'batticaloa': (7.7102, 81.6924),
        'negombo': (7.2089, 79.8589),
        'kalutara': (6.5881, 80.0036),
        'beruwala': (6.4719, 79.9889),
        'hikkaduwa': (6.1482, 80.1043),
        'unawatuna': (6.0232, 80.2456),
        'nuwara_eliya': (6.9497, 80.7850),
        'ella': (6.8654, 81.0480),
        'bandarawela': (6.8288, 80.9929),
        'dambulla': (7.8673, 80.6547),
        'sigiriya': (7.9426, 80.7618),
        'kegalle': (7.2554, 80.3506),
        'peradeniya': (7.2686, 80.5931),
        'puttalam': (8.0328, 79.8346),
        'kurunegala': (7.4788, 80.3637),
        'badulla': (6.9934, 81.0553),
        'gampaha': (7.0878, 80.0027),
        'matale': (7.4680, 80.6240),
        'monaragala': (6.8726, 81.3486),
        'hambantota': (6.1246, 81.1185),
        'ampara': (7.2991, 81.6740),
        'vavuniya': (8.7514, 80.4970),
        'mullaitivu': (9.2674, 80.8137),
        'kilinochchi': (9.3966, 80.3957),
        'mannar': (8.9818, 79.9018),
        'chilaw': (7.5759, 79.7964),
        'ambalangoda': (6.2348, 80.0559),
        'tangalle': (6.0244, 80.7959),
        'embilipitiya': (6.3411, 80.8448),
        'balangoda': (6.6483, 80.6942),
        'tissamaharama': (6.2843, 81.2921),
        'wellawaya': (6.7333, 81.1000),
        'haputale': (6.7653, 80.9531),
        'nittambuwa': (7.1589, 80.0872),
        'avissawella': (6.9350, 80.2150),
        'horana': (6.7156, 80.0614),
        'panadura': (6.7128, 79.9036),
        'moratuwa': (6.7728, 79.8820),
        'dehiwala': (6.8356, 79.8656),
        'kaduwela': (6.9303, 79.9775),
        'kotte': (6.8929, 79.8989),
        'maharagama': (6.8481, 79.9264),
        'piliyandala': (6.7994, 79.9203),
        'rajagiriya': (6.9078, 79.9003),
        'nugegoda': (6.8703, 79.8991),
        'boralesgamuwa': (6.8356, 79.8994),
    }
    
    # Normalize city names: lowercase and replace spaces with underscores
    city1_normalized = city1.lower().replace(' ', '_') if city1 else ''
    city2_normalized = city2.lower().replace(' ', '_') if city2 else ''
    
    # Handle missing cities gracefully
    if city1_normalized not in city_coords or city2_normalized not in city_coords:
        print(f"WARNING: City not in coordinates database. city1='{city1_normalized}', city2='{city2_normalized}'")
        return 100.0  # Default distance instead of random
    
    if city1_normalized == city2_normalized:
        return 0.0
    
    # Calculate Haversine distance
    lat1, lon1 = city_coords[city1_normalized]
    lat2, lon2 = city_coords[city2_normalized]
    
    distance = ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5 * 111  # km per degree
    return round(distance, 1)
