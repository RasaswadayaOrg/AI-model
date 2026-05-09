"""Canonical preference mapping shared by the AI runtime.

The web app stores compact onboarding IDs. This module expands those IDs into
the art forms, genres, styles, moods, and city tokens used by the generated AI
dataset, while keeping backward compatibility with older saved values.
"""

import re
from typing import Dict, Iterable, List, Optional, Union


ART_FORMS = {"music", "dance", "film", "drama"}

CATEGORY_TO_ART_FORM = {
    # Music — only actual music/audio performance events
    "music": "music",
    "concert": "music",
    "live music": "music",
    "baila": "music",
    "classical": "music",
    "carnatic": "music",
    "orchestra": "music",
    "choir": "music",
    "folk music": "music",
    "band": "music",
    "opera": "music",

    # Dance — only actual dance performance events
    "dance": "dance",
    "kandyan": "dance",
    "kandyan dance": "dance",
    "ballet": "dance",
    "contemporary dance": "dance",
    "folk dance": "dance",
    "low country": "dance",
    "sabaragamuwa": "dance",
    "bharatanatyam": "dance",

    # Drama / Theatre
    "drama": "drama",
    "theatre": "drama",
    "theater": "drama",
    "play": "drama",
    "nadagam": "drama",
    "kolam": "drama",
    "nurthi": "drama",
    "stage": "drama",
    "teledrama": "drama",

    # Film — ONLY actual cinema events
    "film": "film",
    "cinema": "film",
    "movie": "film",
    "screening": "film",
    "documentary": "film",
    "short film": "film",
    "film festival": "film",

    # Visual arts / crafts / exhibitions are not performing-art forms.
    "art": None,
    "arts": None,
    "exhibition": None,
    "exhibitions": None,
    "gallery": None,
    "craft": None,
    "crafts": None,
    "handicraft": None,
    "batik": None,
    "handloom": None,
    "cultural": None,
    "heritage": None,
    "festival": None,
    "workshop": None,
    "seminar": None,
    "fair": None,
    "expo": None,
    "general": None,
}

CATEGORY_TO_GENRE = {
    "art": "visual_arts",
    "arts": "visual_arts",
    "exhibition": "visual_arts",
    "exhibitions": "visual_arts",
    "craft": "traditional_crafts",
    "crafts": "traditional_crafts",
    "batik": "traditional_crafts",
    "handloom": "traditional_crafts",
    "cultural": "cultural_heritage",
    "heritage": "cultural_heritage",
    "festival": "festival",
    "folk": "folk_tradition",
}


def get_art_form(category: str):
    if not category:
        return None
    key = str(category).lower().strip()
    return CATEGORY_TO_ART_FORM.get(key, CATEGORY_TO_ART_FORM.get(normalise_token(category), None))


def get_genre(category: str):
    if not category:
        return None
    key = str(category).lower().strip()
    return CATEGORY_TO_GENRE.get(key, CATEGORY_TO_GENRE.get(normalise_token(category), None))


PreferenceExpansion = Dict[str, Union[str, List[str]]]

PREFERENCE_EXPANSIONS: Dict[str, PreferenceExpansion] = {
    "sinhala_classical": {
        "canonical": "sinhala_classical",
        "categories": ["music"],
        "genres": ["sinhala_classical", "light_classical", "classical_vocal", "classical_instrumental"],
        "styles": ["classical_semi_classical"],
        "culture": ["traditional"],
        "moods": ["spiritual", "devotional", "patriotic"],
    },
    "classical_music": {
        "canonical": "sinhala_classical",
        "categories": ["music"],
        "genres": ["sinhala_classical", "light_classical", "classical_vocal", "classical_instrumental"],
        "styles": ["classical_semi_classical"],
        "culture": ["traditional"],
        "moods": ["spiritual", "devotional", "patriotic"],
    },
    "folk_fusion": {
        "canonical": "folk_fusion",
        "categories": ["music"],
        "genres": ["folk_fusion", "jana_kavi", "harvest_songs", "virindu"],
        "styles": ["traditional_indigenous", "fusion"],
        "culture": ["traditional", "fusion"],
        "moods": ["traditional", "rural_village", "cultural_pride"],
    },
    "folk_music": {
        "canonical": "folk_fusion",
        "categories": ["music"],
        "genres": ["folk_fusion", "jana_kavi", "harvest_songs", "virindu"],
        "styles": ["traditional_indigenous", "fusion"],
        "culture": ["traditional", "fusion"],
        "moods": ["traditional", "rural_village", "cultural_pride"],
    },
    "light_classical": {
        "canonical": "light_classical",
        "categories": ["music"],
        "genres": ["light_classical", "sinhala_pop", "ballads", "love_songs"],
        "styles": ["classical_semi_classical", "sinhala_commercial"],
        "culture": ["traditional", "contemporary"],
        "moods": ["emotional", "peaceful", "romantic"],
    },
    "sarala_gee": {
        "canonical": "light_classical",
        "categories": ["music"],
        "genres": ["light_classical", "sinhala_pop", "ballads", "love_songs"],
        "styles": ["classical_semi_classical", "sinhala_commercial"],
        "culture": ["traditional", "contemporary"],
        "moods": ["emotional", "peaceful", "romantic"],
    },
    "classical_fusion": {
        "canonical": "classical_fusion",
        "categories": ["music"],
        "genres": ["classical_fusion", "folk_fusion", "ethnic_electronic", "traditional_hip_hop_fusion"],
        "styles": ["fusion"],
        "culture": ["fusion", "contemporary"],
        "moods": ["fusion_energy", "energetic", "uplifting"],
    },
    "fusion_modern": {
        "canonical": "classical_fusion",
        "categories": ["music"],
        "genres": ["classical_fusion", "folk_fusion", "ethnic_electronic", "traditional_hip_hop_fusion"],
        "styles": ["fusion"],
        "culture": ["fusion", "contemporary"],
        "moods": ["fusion_energy", "energetic", "uplifting"],
    },
    "orchestral_scores": {
        "canonical": "orchestral_scores",
        "categories": ["music"],
        "genres": ["orchestral_scores", "background_scores", "classical_vocal"],
        "styles": ["classical_semi_classical", "film_music"],
        "culture": ["traditional"],
        "moods": ["peaceful", "meditative", "spiritual"],
    },
    "classical_instrumental": {
        "canonical": "orchestral_scores",
        "categories": ["music"],
        "genres": ["orchestral_scores", "background_scores", "classical_vocal"],
        "styles": ["classical_semi_classical", "film_music"],
        "culture": ["traditional"],
        "moods": ["peaceful", "meditative", "spiritual"],
    },
    "instrumental": {
        "canonical": "orchestral_scores",
        "categories": ["music"],
        "genres": ["orchestral_scores", "background_scores", "classical_vocal"],
        "styles": ["classical_semi_classical", "film_music"],
        "culture": ["traditional"],
        "moods": ["peaceful", "meditative", "spiritual"],
    },
    "ves_dance": {
        "canonical": "ves_dance",
        "categories": ["dance"],
        "genres": ["ves_dance", "naiyandi", "pantheru", "uddekki", "vannam_dance"],
        "styles": ["kandyan_dance"],
        "culture": ["traditional"],
        "moods": ["ritualistic", "sacred", "graceful"],
    },
    "upcountry_dance": {
        "canonical": "ves_dance",
        "categories": ["dance"],
        "genres": ["ves_dance", "naiyandi", "pantheru", "uddekki", "vannam_dance"],
        "styles": ["kandyan_dance"],
        "culture": ["traditional"],
        "moods": ["ritualistic", "sacred", "graceful"],
    },
    "kolam_dance": {
        "canonical": "kolam_dance",
        "categories": ["dance"],
        "genres": ["kolam_dance", "sanni_yakuma", "devil_dance", "ritual_based", "drum_centered"],
        "styles": ["low_country_dance"],
        "culture": ["traditional", "ritual"],
        "moods": ["ritualistic", "fierce", "theatrical"],
    },
    "lowcountry_dance": {
        "canonical": "kolam_dance",
        "categories": ["dance"],
        "genres": ["kolam_dance", "sanni_yakuma", "devil_dance", "ritual_based", "drum_centered"],
        "styles": ["low_country_dance"],
        "culture": ["traditional", "ritual"],
        "moods": ["ritualistic", "fierce", "theatrical"],
    },
    "harvest_dances": {
        "canonical": "harvest_dances",
        "categories": ["dance"],
        "genres": ["harvest_dances", "village_festival_dances", "new_year_dances"],
        "styles": ["folk_dance"],
        "culture": ["traditional", "festival_specific"],
        "moods": ["festive", "joyful", "rural_village"],
    },
    "sabaragamuwa_dance": {
        "canonical": "harvest_dances",
        "categories": ["dance"],
        "genres": ["harvest_dances", "village_festival_dances", "new_year_dances"],
        "styles": ["folk_dance"],
        "culture": ["traditional", "festival_specific"],
        "moods": ["festive", "joyful", "rural_village"],
    },
    "sri_lankan_contemporary": {
        "canonical": "sri_lankan_contemporary",
        "categories": ["dance"],
        "genres": ["sri_lankan_contemporary", "interpretative", "theatrical_dance"],
        "styles": ["contemporary_modern"],
        "culture": ["contemporary"],
        "moods": ["expressive", "graceful", "dramatic"],
    },
    "contemporary_dance": {
        "canonical": "sri_lankan_contemporary",
        "categories": ["dance"],
        "genres": ["sri_lankan_contemporary", "interpretative", "theatrical_dance"],
        "styles": ["contemporary_modern"],
        "culture": ["contemporary"],
        "moods": ["expressive", "graceful", "dramatic"],
    },
    "social_realism": {
        "canonical": "social_realism",
        "categories": ["film"],
        "genres": ["social_realism", "realism", "social_justice", "ethnic_conflict"],
        "styles": ["art_parallel_cinema", "political_cinema"],
        "culture": ["contemporary", "educational"],
        "moods": ["social_commentary", "serious", "reflective"],
    },
    "romantic_drama": {
        "canonical": "romantic_drama",
        "categories": ["film", "drama"],
        "genres": ["romantic_drama", "romantic_comedy", "family_drama", "romance_series"],
        "styles": ["romantic", "modern_stage_drama"],
        "culture": ["contemporary"],
        "moods": ["romantic", "emotional", "dramatic"],
    },
    "ancient_sri_lanka": {
        "canonical": "ancient_sri_lanka",
        "categories": ["film"],
        "genres": ["ancient_sri_lanka", "colonial_era", "biographical", "jataka_tales"],
        "styles": ["historical_period", "religious_mythological"],
        "culture": ["traditional", "educational"],
        "moods": ["historical_nostalgia", "heroic", "mythological"],
    },
    "low_budget_indie": {
        "canonical": "low_budget_indie",
        "categories": ["film"],
        "genres": ["low_budget_indie", "avant_garde", "festival_cinema", "symbolic_cinema"],
        "styles": ["experimental_independent", "art_parallel_cinema"],
        "culture": ["contemporary"],
        "moods": ["reflective", "social_commentary", "satirical"],
    },
    "nadagam": {
        "canonical": "nadagam",
        "categories": ["drama"],
        "genres": ["nadagam", "kolam_theatre", "sokari"],
        "styles": ["traditional_theatre"],
        "culture": ["traditional"],
        "moods": ["theatrical", "mythological", "traditional"],
    },
    "stylized_drama": {
        "canonical": "nadagam",
        "categories": ["drama"],
        "genres": ["nadagam", "kolam_theatre", "sokari"],
        "styles": ["traditional_theatre"],
        "culture": ["traditional"],
        "moods": ["theatrical", "mythological", "traditional"],
    },
    "social_drama": {
        "canonical": "social_drama",
        "categories": ["drama"],
        "genres": ["social_drama", "literary_adaptation", "family_drama"],
        "styles": ["modern_stage_drama"],
        "culture": ["contemporary"],
        "moods": ["dramatic", "social_awareness", "emotional"],
    },
    "realistic_drama": {
        "canonical": "social_drama",
        "categories": ["drama"],
        "genres": ["social_drama", "literary_adaptation", "family_drama"],
        "styles": ["modern_stage_drama"],
        "culture": ["contemporary"],
        "moods": ["dramatic", "social_awareness", "emotional"],
    },
    "stage_musicals": {
        "canonical": "stage_musicals",
        "categories": ["drama"],
        "genres": ["stage_musicals", "opera", "physical_theatre"],
        "styles": ["musical_drama"],
        "culture": ["contemporary", "entertainment"],
        "moods": ["joyful", "theatrical", "expressive"],
    },
    "comedy_drama": {
        "canonical": "stage_musicals",
        "categories": ["drama"],
        "genres": ["stage_musicals", "family_drama", "romance_series"],
        "styles": ["musical_drama", "modern_stage_drama"],
        "culture": ["entertainment"],
        "moods": ["comedy", "humorous", "satirical"],
    },
    "political_theatre": {
        "canonical": "political_theatre",
        "categories": ["drama"],
        "genres": ["political_theatre", "social_drama", "physical_theatre"],
        "styles": ["modern_stage_drama", "experimental_avant_garde"],
        "culture": ["contemporary", "educational"],
        "moods": ["political_awareness", "social_awareness", "urban_street"],
    },
    "street_drama": {
        "canonical": "political_theatre",
        "categories": ["drama"],
        "genres": ["political_theatre", "social_drama", "physical_theatre"],
        "styles": ["modern_stage_drama", "experimental_avant_garde"],
        "culture": ["contemporary", "educational"],
        "moods": ["political_awareness", "social_awareness", "urban_street"],
    },
}


def normalise_token(value: Optional[str]) -> str:
    if value is None:
        return ""
    token = str(value).strip().lower().replace("&", "and")
    token = re.sub(r"[^a-z0-9]+", "_", token)
    return token.strip("_")


def normalise_city(value: Optional[str]) -> str:
    return normalise_token(value) or "colombo"


def append_unique(target: List[str], values: Iterable[str]):
    for raw_value in values or []:
        value = normalise_token(raw_value)
        if value and value not in target:
            target.append(value)


def add_term(terms: List[str], value: Optional[str]):
    token = normalise_token(value)
    if not token or token in terms:
        return
    terms.append(token)
    for part in token.split("_"):
        if len(part) > 2 and part not in terms:
            terms.append(part)


def build_preference_profile(categories=None, interests=None, city=None) -> dict:
    art_forms: List[str] = []
    canonical_interests: List[str] = []
    genres: List[str] = []
    styles: List[str] = []
    culture_preferences: List[str] = []
    mood_preferences: List[str] = []
    match_terms: List[str] = []

    for raw_category in categories or []:
        token = normalise_token(raw_category)
        category = get_art_form(raw_category)
        if category in ART_FORMS:
            append_unique(art_forms, [category])
            add_term(match_terms, category)
            continue
        genre = get_genre(raw_category)
        if genre:
            append_unique(genres, [genre])
            add_term(match_terms, genre)
        elif token:
            add_term(match_terms, token)

    for raw_interest in interests or []:
        token = normalise_token(raw_interest)
        if not token:
            continue
        category = get_art_form(raw_interest)
        if category in ART_FORMS:
            append_unique(art_forms, [category])
            add_term(match_terms, category)
            continue

        genre = get_genre(raw_interest)
        if genre:
            append_unique(canonical_interests, [token])
            append_unique(genres, [genre])
            add_term(match_terms, genre)
            add_term(match_terms, token)
            continue

        expansion = PREFERENCE_EXPANSIONS.get(token)
        if expansion:
            append_unique(canonical_interests, [str(expansion["canonical"])])
            append_unique(art_forms, expansion.get("categories", []))
            append_unique(genres, expansion.get("genres", []))
            append_unique(styles, expansion.get("styles", []))
            append_unique(culture_preferences, expansion.get("culture", []))
            append_unique(mood_preferences, expansion.get("moods", []))
            for expanded_value in [
                str(expansion["canonical"]),
                *expansion.get("categories", []),
                *expansion.get("genres", []),
                *expansion.get("styles", []),
                *expansion.get("culture", []),
                *expansion.get("moods", []),
            ]:
                add_term(match_terms, expanded_value)
        else:
            append_unique(canonical_interests, [token])
            append_unique(genres, [token])
            add_term(match_terms, token)

    city_token = normalise_city(city)
    add_term(match_terms, city_token)

    return {
        "city": city_token,
        "art_interests": art_forms or ["music"],
        "interests": canonical_interests,
        "genres": genres,
        "styles": styles,
        "culture_preferences": culture_preferences or ["contemporary"],
        "mood_preferences": mood_preferences or ["energetic"],
        "match_terms": match_terms,
    }
