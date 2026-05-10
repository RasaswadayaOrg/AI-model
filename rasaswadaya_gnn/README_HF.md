---
title: Rasaswadaya GNN API
emoji: 🎭
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Cultural-arts recommendation GNN (FastAPI)
---

# Rasaswadaya GNN Recommendation API

FastAPI service exposing the trained heterogeneous-graph recommendation model for the Rasaswadaya cultural-arts platform.

## Endpoints

- `GET /recommend/{user_id}` — top-K personalised artists & events
- `POST /refresh` — re-fetch live dataset from Supabase and rebuild the graph

## Configuration (Space Secrets)

| Name           | Required | Notes                                                     |
| -------------- | -------- | --------------------------------------------------------- |
| `DATABASE_URL` | yes      | Supabase Postgres connection string used by `/refresh`    |

The startup pipeline loads a bundled pilot dataset (`data/sample_dataset/rasaswadaya_dataset.pkl`) so the Space is functional out of the box; calling `/refresh` swaps to the live DB graph.
