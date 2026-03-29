import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from utils.data_loader import load_dataset, load_users_list
from utils.recommender import get_recommendations, get_trending_data
from utils.helpers import format_artist, format_event, format_trending_artist, format_trending_event
from utils.graph_visualizer import create_gnn_visualization, create_gnn_architecture_diagram

# Page config
st.set_page_config(
    page_title="Rasaswadaya - Entertainment Discovery",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern Design System CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
    
    /* Main container */
    .main {
        max-width: 1280px;
        margin: 0 auto;
    }
    
    /* Top bar styling */
    .top-bar {
        background: linear-gradient(135deg, #ffffff 0%, #f9f9f9 100%);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid #e5e5e5;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        border-radius: 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }
    
    /* Section title */
    .section-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #f0f0f0;
    }
    
    /* Card styling */
    .card-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #f0f0f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .card-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(208, 0, 0, 0.1);
        border-color: #D00000;
    }
    
    /* Artist/Event card */
    .item-card {
        background: #ffffff;
        border: 1px solid #f0f0f0;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .item-card:hover {
        border-color: #D00000;
        box-shadow: 0 8px 16px rgba(208, 0, 0, 0.08);
        transform: translateY(-2px);
    }
    
    .item-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    
    .item-meta {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.75rem;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    
    .item-score {
        font-size: 1.25rem;
        font-weight: 700;
        color: #D00000;
        margin: 0.5rem 0;
    }
    
    /* Trending card */
    .trending-card {
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 100%);
        border: 1px solid #f0f0f0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .trending-card:hover {
        border-color: #D00000;
        box-shadow: 0 4px 12px rgba(208, 0, 0, 0.08);
    }
    
    .trending-rank {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    
    .trending-value {
        font-size: 0.9rem;
        color: #D00000;
        font-weight: 600;
    }
    
    /* Primary button */
    .btn-primary {
        background: #D00000 !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 15px rgba(208, 0, 0, 0.2) !important;
    }
    
    .btn-primary:hover {
        background: #B80000 !important;
        box-shadow: 0 6px 25px rgba(208, 0, 0, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        background: rgba(208, 0, 0, 0.1);
        color: #D00000;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    /* Divider */
    hr.modern {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #e5e5e5, transparent);
        margin: 1.5rem 0;
    }
    
    /* Footer */
    .footer {
        background: #0f172a;
        color: #e5e7eb;
        padding: 3rem 2rem;
        margin-top: 3rem;
        border-top: 1px solid #1e293b;
        text-align: center;
        font-size: 0.9rem;
    }
    
    .footer-divider {
        border: none;
        height: 1px;
        background: #1e293b;
        margin: 1.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_user' not in st.session_state:
    st.session_state.selected_user = None
if 'show_more_artists' not in st.session_state:
    st.session_state.show_more_artists = False
if 'show_more_events' not in st.session_state:
    st.session_state.show_more_events = False
if 'show_more_trending' not in st.session_state:
    st.session_state.show_more_trending = False

# Load data
@st.cache_resource
def load_all_data():
    users, artists, events, follows, attends = load_dataset()
    users_list = load_users_list()
    return users, artists, events, follows, attends, users_list

users, artists, events, follows, attends, users_list = load_all_data()

# ===== TOP BAR =====
st.markdown('<div class="top-bar">', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center;'>
    <h1 style='margin: 0; padding: 0; font-size: 2rem;'>Rasaswadaya</h1>
    <div style='font-size: 0.9rem; color: #666; margin-top: 0.25rem;'>
        Entertainment Discovery Platform
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ===== USER DISCOVERY SECTION =====
st.markdown('<hr class="modern">', unsafe_allow_html=True)
st.markdown("<h2 class='section-title'>Discover & Explore</h2>", unsafe_allow_html=True)

selected_name = st.selectbox(
    "Select a user:",
    users_list,
    format_func=lambda x: f"{x['name']} ({x['user_id']})",
    key="user_select"
)
st.session_state.selected_user = selected_name

# Display user profile card
if st.session_state.selected_user:
    user_data = st.session_state.selected_user
    interests_badges = " ".join([f'<span class="badge">{interest}</span>' for interest in user_data['interests'][:4]])
    moods_badges = " ".join([f'<span class="badge">{mood}</span>' for mood in user_data['moods'][:4]])
    
    st.markdown(f"""
    <div class="card-container">
        <div style='display: flex; justify-content: space-between; align-items: start;'>
            <div>
                <div style='font-family: Outfit; font-size: 1.3rem; font-weight: 700; color: #1a1a1a;'>{user_data['name']}</div>
                <div style='color: #D00000; font-weight: 600; font-size: 0.95rem; margin-top: 0.25rem;'>{user_data['city']}</div>
            </div>
            <div style='text-align: right;'>
                <div style='font-size: 0.8rem; color: #999; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Interests</div>
                <div style='margin-bottom: 1rem;'>
                    {interests_badges}
                </div>
                <div style='font-size: 0.8rem; color: #999; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Genres</div>
                <div>
                    {moods_badges}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="modern">', unsafe_allow_html=True)

# Get recommendations
recommendations = None
trending = None
if st.session_state.selected_user:
    recommendations = get_recommendations(st.session_state.selected_user)
    # Pass selected user to get personalized trending events/genres
    trending = get_trending_data(user=st.session_state.selected_user)
    
    # Show similar users section
    if recommendations.get('similar_users'):
        st.markdown("<h2 class='section-title'>👥 Users With Similar Tastes</h2>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.9rem; color: #666; margin-bottom: 1rem;'>Discover users who share your interests - recommendations are influenced by what they enjoy</div>", unsafe_allow_html=True)
        
        similar_cols = st.columns(5)
        for idx, similar_user in enumerate(recommendations['similar_users'][:5]):
            with similar_cols[idx]:
                similarity_pct = int(similar_user['similarity'] * 100)
                st.markdown(f"""
                <div style='background: #f5f5f5; border-radius: 8px; padding: 1rem; text-align: center; border: 2px solid #e0e0e0;'>
                    <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>👤</div>
                    <div style='font-size: 0.85rem; font-weight: 600; color: #1a1a1a; margin-bottom: 0.5rem;'>{similar_user['name'].split()[0]}</div>
                    <div style='font-size: 0.75rem; color: #D00000; font-weight: 600;'>{similarity_pct}% match</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('<hr class="modern">', unsafe_allow_html=True)
    
    # Main three-column layout
    col1, col2, col3 = st.columns(3, gap="medium")
    
    # ===== COLUMN 1: ARTISTS =====
    with col1:
        st.markdown("<h2 class='section-title'>Recommended Artists</h2>", unsafe_allow_html=True)
        
        num_artists = len(recommendations['artists']) if not st.session_state.show_more_artists else 10
        display_artists = recommendations['artists'][:num_artists]
        
        for i, artist in enumerate(display_artists, 1):
            # Render opening of card so the button can be visually placed inside the same card
            art_forms = ", ".join(artist.get('art_forms', []))
            genres_str = ", ".join(artist.get('genres', []))
            styles_str = ", ".join(artist.get('styles', []))
            
            # Recommendation reason badge
            reason = artist.get('reason', 'content')
            if reason == 'collaborative':
                reason_badge = '<span class="badge" style="background: rgba(74, 144, 226, 0.15); color: #4A90E2;">👥 Similar Users</span>'
            elif reason == 'discovery':
                reason_badge = '<span class="badge" style="background: rgba(255, 193, 7, 0.15); color: #F57C00;">🔥 Trending</span>'
            else:
                reason_badge = '<span class="badge" style="background: rgba(76, 175, 80, 0.15); color: #4CAF50;">✨ For You</span>'
            
            st.markdown(f"""
            <div class="item-card">
                <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;'>
                    <div class="item-title" style='margin-bottom: 0;'>{i}. {artist['name']}</div>
                    {reason_badge}
                </div>
                <div class="item-meta" style='font-weight: 600;'>🎭 {artist.get('art_form', artist.get('genre', 'music'))}</div>
                <div class="item-meta" style='font-size: 0.85rem; color: #666;'><strong>Forms:</strong> {art_forms}</div>
                <div class="item-meta" style='font-size: 0.85rem; color: #666;'><strong>Genres:</strong> {genres_str}</div>
                <div class="item-meta" style='font-size: 0.85rem; color: #666;'><strong>Styles:</strong> {styles_str}</div>
            """, unsafe_allow_html=True)

            # Follow button placed while card div is still open to appear inside the card
            if st.button("Follow", key=f"follow_artist_{i}_{artist['id']}", use_container_width=True):
                st.success(f"Following {artist['name']}!")

            # Close the card container
            st.markdown("</div>", unsafe_allow_html=True)
        
        if len(recommendations['artists']) > 5 and not st.session_state.show_more_artists:
            if st.button("Show More Artists", use_container_width=True, key="more_artists"):
                st.session_state.show_more_artists = True
                st.rerun()
        
        if st.session_state.show_more_artists and st.button("Show Less", use_container_width=True, key="less_artists"):
            st.session_state.show_more_artists = False
            st.rerun()
    
    # ===== COLUMN 2: EVENTS =====
    with col2:
        st.markdown("<h2 class='section-title'>Recommended Events</h2>", unsafe_allow_html=True)
        
        # Information about location-based recommendations
        with st.expander("📍 Sorted by Proximity to Your Location", expanded=False):
            st.markdown("""
            **Events are ranked by location first!**
            
            This means:
            - Events closest to your city appear first (50% of ranking)
            - Combined with your interests (30%) and what similar users attend (10%)
            - Popular events boost matters (10%)
            
            Find events you can actually attend nearby! 🎭
            """)
        
        num_events = len(recommendations['events']) if not st.session_state.show_more_events else 10
        display_events = recommendations['events'][:num_events]
        
        for i, event in enumerate(display_events, 1):
            distance_km = event.get('distance', 0)
            proximity_color = "#28a745" if distance_km < 50 else "#ffc107" if distance_km < 150 else "#dc3545"
            proximity_label = "Very Close" if distance_km < 50 else "Nearby" if distance_km < 150 else "Moderate Distance"
            
            # render event card with location and distance
            st.markdown(f"""
            <div class="item-card">
                <div class="item-title">{i}. {event['name']}</div>
                <div class="item-meta" style='margin-top: 0.75rem;'>
                    <div style='color: #666;'>{event['city']}</div>
                    <div style='margin-top: 0.5rem; color: {proximity_color}; font-weight: 600; font-size: 0.8rem;'>
                        📍 {distance_km} km • {proximity_label}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Place Attend button inside the card
            if st.button("Attend", key=f"attend_event_{i}_{event['id']}", use_container_width=True):
                st.success(f"Marked attending: {event['name']}!")

            # close card container
            st.markdown("</div>", unsafe_allow_html=True)
        
        if len(recommendations['events']) > 5 and not st.session_state.show_more_events:
            if st.button("Show More Events", use_container_width=True, key="more_events"):
                st.session_state.show_more_events = True
                st.rerun()
        
        if st.session_state.show_more_events and st.button("Show Less", use_container_width=True, key="less_events"):
            st.session_state.show_more_events = False
            st.rerun()
    
    # ===== COLUMN 3: TRENDING =====
    with col3:
        st.markdown("<h2 class='section-title'>Trending Now</h2>", unsafe_allow_html=True)
        
        # Trending Artists
        st.markdown("<div style='font-family: Outfit; font-weight: 600; font-size: 1.05rem; margin-bottom: 1rem; color: #1a1a1a;'>Top Artists</div>", unsafe_allow_html=True)
        
        if trending['artists']:
            max_follows = max([a['follows'] for a in trending['artists']])
            num_trending_artists = len(trending['artists']) if st.session_state.show_more_trending else 3
            
            for i, artist in enumerate(trending['artists'][:num_trending_artists], 1):
                st.markdown(f"""
                <div class="trending-card">
                    <div class="trending-rank">#{i} {artist['name']}</div>
                    <div class="item-meta" style='margin-top: 0.5rem; font-size: 0.85rem;'>{artist.get('art_form', artist.get('genre', 'music'))}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('<hr class="modern">', unsafe_allow_html=True)
        
        # Trending Genres (NEW!)
        st.markdown("<div style='font-family: Outfit; font-weight: 600; font-size: 1.05rem; margin-bottom: 1rem; color: #1a1a1a;'>🎨 Hot Genres</div>", unsafe_allow_html=True)
        
        if trending.get('genres'):
            max_engagement = max([g['engagement'] for g in trending['genres']])
            num_trending_genres = len(trending['genres']) if st.session_state.show_more_trending else 3
            
            for i, genre in enumerate(trending['genres'][:num_trending_genres], 1):
                # Calculate percentage of max engagement for visual bar
                engagement_pct = (genre['engagement'] / max_engagement) * 100 if max_engagement > 0 else 0
                st.markdown(f"""
                <div class="trending-card">
                    <div class="trending-rank">#{i} {genre['name'].replace('_', ' ').title()}</div>
                    <div style='margin-top: 0.5rem; font-size: 0.75rem; color: #999;'>Engagement: {genre['engagement']}</div>
                    <div style='margin-top: 0.3rem; width: 100%; background: #f0f0f0; border-radius: 4px; height: 4px;'>
                        <div style='background: linear-gradient(90deg, #D00000, #FF6666); width: {engagement_pct}%; height: 4px; border-radius: 4px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('<hr class="modern">', unsafe_allow_html=True)
        
        # Trending Events
        st.markdown("<div style='font-family: Outfit; font-weight: 600; font-size: 1.05rem; margin-bottom: 1rem; color: #1a1a1a;'>Top Events</div>", unsafe_allow_html=True)
        
        if trending['events']:
            max_attendees = max([e['attendees'] for e in trending['events']])
            num_trending_events = len(trending['events']) if st.session_state.show_more_trending else 3
            
            for i, event in enumerate(trending['events'][:num_trending_events], 1):
                # NEW: Use art_form instead of genres list
                art_form = event.get('art_form', event.get('genre', 'music'))
                art_form_text = art_form if isinstance(art_form, str) else 'music'
                genre_display = f"<div class='item-meta' style='margin-top: 0.5rem; font-size: 0.85rem;'>{art_form_text}</div>"
                st.markdown(f"""
                <div class="trending-card">
                    <div class="trending-rank">#{i} {event['name']}</div>
                    {genre_display}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('<hr class="modern">', unsafe_allow_html=True)
        
        if not st.session_state.show_more_trending:
            if st.button("View All Trending", use_container_width=True, key="more_trending"):
                st.session_state.show_more_trending = True
                st.rerun()
        else:
            if st.button("Show Less", use_container_width=True, key="less_trending"):
                st.session_state.show_more_trending = False
                st.rerun()

st.markdown('<hr class="modern">', unsafe_allow_html=True)

# ===== GNN VISUALIZATION SECTION =====
st.markdown("<h2 class='section-title'>🧠 How Collaborative Filtering Works</h2>", unsafe_allow_html=True)
st.markdown("<div style='font-size: 0.95rem; color: #666; margin-bottom: 1.5rem;'>Understand how the system combines YOUR preferences with insights from SIMILAR USERS to create personalized recommendations</div>", unsafe_allow_html=True)

# Create two tabs for different visualizations
viz_tab1, viz_tab2 = st.tabs(["🔗 Collaborative Graph", "🏗️ System Architecture"])

with viz_tab1:
    if st.session_state.selected_user and recommendations and recommendations.get('artists'):
        st.markdown("<div style='font-size: 0.95rem; color: #666; margin-bottom: 1.5rem;'>🔍 Interactive visualization showing how collaborative filtering connects you with similar users and personalized recommendations</div>", unsafe_allow_html=True)
        graph_fig = create_gnn_visualization(st.session_state.selected_user, recommendations)
        st.plotly_chart(graph_fig, use_container_width=True, key="gnn_graph")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 1.5rem; border-radius: 12px; margin-top: 1.5rem; font-size: 0.95rem; color: #333; border-left: 4px solid #D00000;'>
        <b style='font-size: 1.1rem; color: #1a1a1a;'>📊 Understanding the Collaborative Graph:</b>
        <div style='margin-top: 1rem; line-height: 1.8;'>
            <div style='margin-bottom: 0.5rem;'><b>🔴 Red Star</b> = You (the current user at the center)</div>
            <div style='margin-bottom: 0.5rem;'><b>🟣 Purple Hexagons</b> = Users with similar tastes (collaborative filtering)</div>
            <div style='margin-bottom: 0.5rem;'><b>🔵 Blue Circles</b> = Recommended Artists (mix of your interests + similar users' favorites + trending)</div>
            <div style='margin-bottom: 0.5rem;'><b>💎 Green Diamonds</b> = Events featuring these artists</div>
            <div style='margin-bottom: 0.5rem;'><b>Purple Dashed Lines</b> = User similarity connections (collaborative filtering)</div>
            <div style='margin-bottom: 0.5rem;'><b>Red Solid Lines</b> = Your personalized recommendations</div>
            <div><b>Blue Dotted Lines</b> = Artist performs at event</div>
        </div>
        <div style='margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.5); font-style: italic; color: #555;'>
            The GNN analyzes <b>your preferences</b> AND learns from <b>users with similar tastes</b> to recommend both familiar content and new discoveries - just like Facebook, Instagram, and YouTube!
        </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background: #f5f5f5; border-radius: 12px; margin: 2rem 0;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>📊</div>
            <div style='font-size: 1.2rem; color: #666; font-weight: 600;'>Select a user above to visualize the recommendation graph</div>
            <div style='font-size: 0.9rem; color: #999; margin-top: 0.5rem;'>See how the GNN connects users to personalized content</div>
        </div>
        """, unsafe_allow_html=True)

with viz_tab2:
    st.markdown("<div style='font-size: 0.9rem; color: #666; margin-bottom: 1rem;'>The collaborative filtering architecture that powers smart recommendations</div>", unsafe_allow_html=True)
    arch_fig = create_gnn_architecture_diagram()
    st.plotly_chart(arch_fig, use_container_width=True, key="gnn_arch")
    
    st.markdown("""
    <div style='background: #f5f5f5; padding: 1.5rem; border-radius: 12px; margin-top: 1rem; font-size: 0.95rem; color: #555;'>
    <b style='font-size: 1.05rem; color: #1a1a1a;'>⚙️ System Components:</b>
    <ul style='margin: 0.75rem 0; line-height: 1.8;'>
        <li><b>Input Layer:</b> Your cultural DNA (interests, art forms, moods, preferences)</li>
        <li><b>User Similarity:</b> Find users with matching tastes using similarity algorithms</li>
        <li><b>GraphSAGE Conv:</b> Neural network learns relationships between all nodes</li>
        <li><b>Collaborative Filter:</b> Aggregate what similar users like that you haven't tried</li>
        <li><b>Output:</b> Hybrid recommendations mixing content-based + collaborative + discovery</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%); padding: 1.5rem; border-radius: 12px; margin-top: 1.5rem; border-left: 4px solid #00BCD4;'>
        <b style='font-size: 1.1rem; color: #1a1a1a;'>🔄 The Complete Recommendation Flow:</b>
        <div style='margin-top: 1rem; line-height: 1.8; color: #333;'>
            <div style='margin-bottom: 0.75rem;'><b>Step 1 - Similarity Matching:</b> Calculate similarity score with every other user based on shared interests, moods, and followed artists</div>
            <div style='margin-bottom: 0.75rem;'><b>Step 2 - Collaborative Discovery:</b> Find artists that similar users follow but you haven't discovered yet (weighted by similarity score)</div>
            <div style='margin-bottom: 0.75rem;'><b>Step 3 - Content Matching:</b> Score artists based on YOUR explicit preferences and cultural attributes</div>
            <div style='margin-bottom: 0.75rem;'><b>Step 4 - Trending Boost:</b> Add popular/trending artists to help you discover what's hot</div>
            <div style='margin-bottom: 0.75rem;'><b>Step 5 - Smart Mixing:</b> Combine 50% content-based + 30% collaborative + 20% discovery</div>
            <div><b>Step 6 - Personalization:</b> Rank final recommendations by combined scores and your engagement patterns</div>
        </div>
        <div style='margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid rgba(0,0,0,0.1); font-weight: 600; color: #00838F;'>
            ✨ Result: A perfectly balanced feed with familiar favorites AND exciting new discoveries!
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="modern">', unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("""
<div class="footer">
    <h3 style='font-family: Outfit; margin-top: 0;'>Rasaswadaya</h3>
    <p>Entertainment Discovery Platform • Powered by Collaborative Filtering GNN</p>
    <hr class="footer-divider">
    <p style='font-size: 0.85rem; opacity: 0.8;'>Discover Sri Lankan cultural artists and events through personalized + community-driven recommendations</p>
    <p style='font-size: 0.75rem; opacity: 0.7; margin-top: 0.5rem;'>Using advanced AI to connect you with both familiar favorites and exciting new discoveries</p>
</div>
""", unsafe_allow_html=True)
