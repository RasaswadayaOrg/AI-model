import os

import psycopg2
from dotenv import load_dotenv


BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

DB_URL = os.environ.get("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL environment variable not set")


def cleanup_orphan_recommendations():
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute('UPDATE "Recommendation" SET reason = %s WHERE reason IS NULL', ("",))
            cur.execute(
                '''
                DELETE FROM "Recommendation" r
                WHERE r."recommendedType" = 'ARTIST'
                  AND NOT EXISTS (
                    SELECT 1 FROM "Artist" a WHERE a.id = r."recommendedId"
                  )
                '''
            )
            orphan_artists = cur.rowcount
            cur.execute(
                '''
                DELETE FROM "Recommendation" r
                WHERE r."recommendedType" = 'EVENT'
                  AND NOT EXISTS (
                    SELECT 1 FROM "Event" e WHERE e.id = r."recommendedId"
                  )
                '''
            )
            orphan_events = cur.rowcount
        conn.commit()
    print({"deleted_artist_recs": orphan_artists, "deleted_event_recs": orphan_events})


if __name__ == "__main__":
    cleanup_orphan_recommendations()