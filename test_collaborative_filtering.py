#!/usr/bin/env python3
"""
Test script to verify collaborative filtering fixes
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / 'rasaswadaya_gnn'))
sys.path.insert(0, str(Path(__file__).parent / 'rasaswadaya_gnn' / 'streamlit_app'))

from streamlit_app.utils.recommender import (
    load_model_and_data,
    find_similar_users,
    get_recommendations,
    calculate_user_similarity
)

def test_collaborative_filtering():
    """Test the collaborative filtering system"""
    print("=" * 60)
    print("TESTING COLLABORATIVE FILTERING SYSTEM")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    try:
        model, dataset, device = load_model_and_data()
        print(f"   ✓ Dataset loaded")
        print(f"   - Users: {len(dataset['users'])}")
        print(f"   - Artists: {len(dataset['artists'])}")
        print(f"   - Follows: {len(dataset.get('follows', []))}")
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        return
    
    # Test with first user
    if not dataset['users']:
        print("   ✗ No users in dataset")
        return
    
    test_user = dataset['users'][0]
    user_name = test_user['name']
    
    print(f"\n2. Testing with user: {user_name}")
    print(f"   - User ID: {test_user['user_id']}")
    print(f"   - City: {test_user.get('city', 'N/A')}")
    print(f"   - Interests: {test_user.get('interests', 'N/A')}")
    
    # Test get_recommendations (which includes collaborative filtering)
    print(f"\n3. Running get_recommendations()...")
    try:
        recommendations = get_recommendations({'name': user_name})
        
        print(f"   ✓ Recommendations generated")
        print(f"   - Artists recommended: {len(recommendations.get('artists', []))}")
        print(f"   - Events recommended: {len(recommendations.get('events', []))}")
        print(f"   - Similar users found: {len(recommendations.get('similar_users', []))}")
        
        if recommendations.get('similar_users'):
            print(f"\n   Similar users:")
            for user in recommendations['similar_users'][:3]:
                print(f"     - {user['name']}: {user['similarity']:.2f} similarity")
        else:
            print(f"   ⚠ WARNING: No similar users found!")
        
        if recommendations.get('artists'):
            collab_artists = [a for a in recommendations['artists'] if a.get('reason') == 'collaborative']
            print(f"\n   Collaborative recommendations: {len(collab_artists)}")
            for artist in collab_artists[:3]:
                print(f"     - {artist['name']} (score: {artist['score']:.2f})")
        
    except Exception as e:
        print(f"   ✗ Error in get_recommendations: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test find_similar_users directly
    print(f"\n4. Testing find_similar_users() directly...")
    try:
        # Build enriched user data
        import ast
        from collections import defaultdict
        
        def parse_list(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str) and val:
                try:
                    return ast.literal_eval(val)
                except:
                    return [val]
            return []
        
        follows_map = defaultdict(list)
        for follow in dataset.get('follows', []):
            user_id_val = follow.get('user_id')
            artist_id = follow.get('artist_id')
            if user_id_val and artist_id:
                follows_map[user_id_val].append(artist_id)
        
        enriched_user_data = {
            'user_id': test_user['user_id'],
            'name': test_user['name'],
            'city': test_user.get('city', ''),
            'art_interests': parse_list(test_user.get('art_interests', [])),
            'interests': parse_list(test_user.get('interests', [])),
            'moods': parse_list(test_user.get('moods', [])),
            'genres': parse_list(test_user.get('genres', [])),
            'follows': follows_map.get(test_user['user_id'], [])
        }
        
        print(f"   Enriched user data:")
        print(f"     - Follows: {len(enriched_user_data['follows'])} artists")
        print(f"     - Interests: {enriched_user_data['interests']}")
        print(f"     - Moods: {enriched_user_data['moods']}")
        
        similar_users = find_similar_users(enriched_user_data, dataset, top_k=5)
        
        if similar_users:
            print(f"   ✓ Found {len(similar_users)} similar users")
            for user, sim_score in similar_users[:3]:
                print(f"     - {user['name']}: {sim_score:.2f}")
        else:
            print(f"   ✗ No similar users found")
    
    except Exception as e:
        print(f"   ✗ Error in find_similar_users: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    test_collaborative_filtering()
