import sys
import argparse
import os
import psycopg2
from pprint import pprint
from dotenv import load_dotenv


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

DB_URL = os.environ.get("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL environment variable not set")

# Connect to database directly to check recommendations
def get_user_recommendations(user_id):
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        
        # SQL query to get recommendations with joined artist/event names
        query = """
        SELECT 
            r.id, 
            r."recommendedType",
            r."score",
            CASE 
                WHEN r."recommendedType" = 'ARTIST' THEN a.name
                WHEN r."recommendedType" = 'EVENT' THEN e.title
                ELSE 'Unknown'
            END as title
        FROM "Recommendation" r
        LEFT JOIN "Artist" a ON r."recommendedId" = a.id AND r."recommendedType" = 'ARTIST'
        LEFT JOIN "Event" e ON r."recommendedId" = e.id AND r."recommendedType" = 'EVENT'
        WHERE r."userId" = %s
        ORDER BY r.score DESC;
        """
        
        cur.execute(query, (user_id,))
        results = cur.fetchall()
        
        if not results:
            print(f"\n❌ No recommendations found for user ID: {user_id}")
            print("Make sure this user exists and has AI recommendations generated.")
            return

        print(f"\n=======================================================")
        print(f"🤖 AI RECOMMENDATIONS FOR USER: {user_id}")
        print(f"=======================================================\n")
        
        artists = []
        events = []
        
        for row in results:
            rec_type = row[1]
            score = row[2]
            title = row[3]
            explanation = "AI Match Based on similar interests"
            
            if rec_type == 'ARTIST':
                artists.append((title, score, explanation))
            elif rec_type == 'EVENT':
                events.append((title, score, explanation))
                
        print(f"🎵 TOP ARTISTS:")
        for i, (title, score, exp) in enumerate(artists[:5], 1):
            sc = float(score) * 100 if score else 0
            print(f"  {i}. {title} [Match: {sc:.1f}%]")
            print(f"     Reason: {exp}")
            print("")
            
        print(f"🎟️ TOP EVENTS:")
        for i, (title, score, exp) in enumerate(events[:5], 1):
            sc = float(score) * 100 if score else 0
            print(f"  {i}. {title} [Match: {sc:.1f}%]")
            print(f"     Reason: {exp}")
            print("")

        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Database error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Prompt if no argument given
        user_id = input("Enter User ID to view recommendations: ")
    else:
        user_id = sys.argv[1]
        
    get_user_recommendations(user_id)
