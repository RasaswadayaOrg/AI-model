import sys
sys.path.insert(0, 'rasaswadaya_gnn/streamlit_app')
from utils.data_loader import load_users_list
from utils.recommender import get_recommendations

users = load_users_list()
print('Total users loaded:', len(users))
amila = next((u for u in users if 'Amila Wickremasinghe' in u['name']), None)
print('Amila found:', amila['name'] if amila else 'NOT FOUND')

if amila:
    print('City:', amila['city'])
    print('Interests:', amila['interests'])
    print('Moods:', amila['moods'])
    result = get_recommendations(amila)
    print()
    print('Artists recommended:', len(result['artists']))
    print('Events recommended:', len(result['events']))
    print('Similar users:', len(result['similar_users']))
    if result['artists']:
        print('Top 3 artists:', [(a['name'], round(a['score'],2)) for a in result['artists'][:3]])
    if result['events']:
        print('Top 3 events:', [(e['name'], e.get('city','')) for e in result['events'][:3]])
