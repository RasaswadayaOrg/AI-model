"""
Large-Scale Dataset Generator for Rasaswadaya.lk
==================================================
Generates a realistic, high-volume Sri Lankan cultural platform dataset.

Scale targets:
  - 10,000 users      (realistic Sri Lankan names, demographics)
  - 500 artists       (rich cultural metadata, 15 real + 485 synthetic)
  - 2,000 events      (linked to artists, venues, cities)
  - ~84,000 follows   (avg 8.4 per user)
  - ~48,000 attends   (avg 4.8 per user)
"""

import random
import json
import pickle
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# NAME POOLS  (large, realistic Sri Lankan name pool)
# ─────────────────────────────────────────────────────────────────────────────

SINHALA_FIRST_NAMES = [
    "Kasun","Nuwan","Chaminda","Mahela","Dilshan","Lasith","Angelo","Sanduni",
    "Dilini","Chathurika","Nadeeka","Thilini","Buddhika","Ravi","Priya","Amila",
    "Danushka","Saman","Lakshman","Nalaka","Nimesh","Sachini","Tharindu","Madhavi",
    "Isuru","Nadeesha","Kavinda","Yashodha","Damith","Ayesha","Ruwan","Hiruni",
    "Shehan","Hashini","Chathura","Anusha","Gayan","Sandani","Thisara","Vindya",
    "Roshan","Chandima","Dinesh","Malsha","Ajith","Samanthi","Lahiru","Nishani",
    "Eranga","Pasan","Supun","Dulani","Chameera","Sachith","Kalani","Devindi",
    "Bhanuka","Asela","Ishan","Keshara","Minuli","Osanda","Pasindu","Rasika",
    "Senuri","Tharaka","Udara","Vimukthi","Wathsala","Xian","Yanuka","Zoysa",
    "Akila","Binara","Charith","Dinuka","Enara","Farida","Gimhana","Hasitha",
    "Inoka","Janith","Kithsiri","Lochana","Menaka","Nithila","Oneli","Pamoda",
    "Quarshi","Renuka","Sajith","Thilina","Umanga","Vindula","Waruni","Hasini",
    "Yasas","Zuha","Amali","Buddhima","Chanaka","Dinesha","Eshan","Fathima",
    "Gayani","Hesara","Imesha","Janaka","Kushani","Lihini","Manuja","Nayomi",
    "Omali","Praveen","Qadeeja","Ruwantha","Sithara","Thathsarani","Ushara",
    "Vihanga","Wasana","Hirantha","Yohan","Zainab","Akalanka","Bimsara",
    "Chamodi","Deshapriya","Ekamali","Fonseka","Haritha","Imali","Jaliya",
    "Kalindu","Laksitha","Maduranga","Navodya","Oshadhi","Piumi","Rangi",
    "Sulochana","Thisuri","Upeksha","Vidusha","Warsha","Xaveen","Yoshitha",
    "Amara","Bandula","Chanuka","Deepika","Erandika","Gamini","Heshara",
    "Indunil","Jeewan","Kithmini","Lihiniya","Malith","Navoda","Ovini",
    "Parakrama","Ramya","Sachindra","Thushara","Umesha","Vikum","Wajira",
    "Yenuli","Amitha","Basil","Chinthaka","Dulmini","Erandi","Falitha",
    "Gathika","Himasha","Indika","Janaki","Kalpa","Liyana","Mihiri","Naveen",
    "Omitha","Pavithra","Rithika","Sadeepa","Thilanka","Uditha","Viraj","Waruna"
]

TAMIL_FIRST_NAMES = [
    "Kumar","Rajan","Vel","Shan","Thiru","Muthu","Nirmala","Deepa","Mala",
    "Kamala","Ragavan","Karthik","Vijay","Arun","Shanthi","Lakshmi","Suresh",
    "Durga","Ganesh","Vasanthi","Mohan","Kavitha","Senthil","Meena","Siva",
    "Vani","Priya","Anbu","Bala","Chandra","Devi","Eswari","Gopal","Hema",
    "Indira","Jaya","Krishnan","Lavanya","Mani","Nisha","Oviya","Prem","Raja",
    "Saranya","Thamil","Uma","Vidhya","Yamuna","Aruna","Brinda","Chelvan",
    "Dharani","Elan","Fathima","Gokul","Harini","Inban","Janani","Kayal",
    "Logesh","Madhu","Nandhini","Oviyam","Pooja","Rathna","Selvi","Thenmozhi",
    "Usha","Vanitha","Yadav","Zara","Abinaya","Bavani","Chithra","Devika",
    "Elango","Gayathri","Hari","Ilakiya","Jeeva","Kavin","Latha","Mithun",
    "Nandha","Oviya","Preethi","Ramesh","Sangeetha","Tamizh","Umadevi",
    "Vinothini","Yuvan","Arjun","Bhargavi","Chinnaiah","Dhivya","Ezhil",
    "Guna","Hariharan","Isai","Janarthanan","Kalaivani","Lingam","Malarvizhi"
]

LAST_NAMES = [
    "Fernando","Silva","Perera","Jayasuriya","Wijesinghe","Mendis","Rajapaksa",
    "Wickremasinghe","Gunasekara","Dissanayake","Kumar","Sivarajah","Nadarajah",
    "Thiyagarajah","Selvam","De Silva","Gunawardena","Amarasinghe","Bandara",
    "Jayawardena","Senanayake","Ranasinghe","Herath","Liyanage","Ekanayake",
    "Weerasinghe","Pathirana","Rathnayake","Samaraweera","Kumara","Abeywickrama",
    "Abeysekera","Abeysinghe","Abeyrathne","Abeyrathna","Abeysena","Abeytunga",
    "Alagiyawanna","Alwis","Amaratunge","Amunugama","Ananda","Ariyarathne",
    "Ariyasena","Ariyawansa","Ariyawansha","Athukorala","Attanayake","Attygalle",
    "Baduge","Balasuriya","Basnayake","Bastian","Batuwantudawa","Bogoda",
    "Chandraratne","Chandrasekara","Cooray","Dahanayake","Daluwatte","Danapala",
    "De Alwis","De Chickera","De Mel","De Zoysa","Devasurendra","Dharmaratne",
    "Dharmasena","Dharmawardene","Dheerasinghe","Dias","Dias","Diasz","Edirisinghe",
    "Ediriweera","Ellepola","Ellepolla","Embuldeniya","Fonseka","Galhena",
    "Gamage","Gamagedara","Gamlath","Gammanpila","Gnanapragasam","Gooneratne",
    "Gunaratne","Gunarathna","Gunasinghe","Gunawardana","Hapangama","Hapuarachchi",
    "Hettiarachchi","Hettiarachchy","Hewa","Hewage","Hewawasam","Jayakody",
    "Jayarathne","Jayaratne","Jayasinghe","Jayatilake","Jayatilleke","Karunarathna",
    "Karunaratne","Karunasena","Karunaweera","Kekulawala","Kodagoda","Kotelawala",
    "Kothalawala","Kulatunge","Kumarasinghe","Kumaratunge","Lakshman","Liyanarachchi",
    "Liyanaarachchi","Liyanage","Lokuge","Madanayake","Maddumage","Madduma",
    "Madushan","Mallawarachchi","Mangala","Marasinghe","Medagama","Mendis",
    "Munasinghe","Muthulingam","Nanayakkara","Napagoda","Navaratne","Navaratna",
    "Obeyesekere","Obeyesekera","Padmasiri","Palihawadana","Palliyaguruge",
    "Paranagama","Patabendige","Peiris","Pelpola","Pieris","Piyadasa","Piyaratne",
    "Ponnamperuma","Ponniah","Prematilake","Priyankara","Pushpakumara","Rajapakshe",
    "Rajasingham","Rajakaruna","Rajaratnam","Ranatunga","Ratnasekera","Ratnayake",
    "Ruwanpura","Samarakoon","Samarakkody","Samarathunge","Samarathuna","Samarasekara",
    "Senaratne","Senavirathna","Serasinghe","Siriwardena","Siriwardana","Siriwardene",
    "Somaratne","Somasiri","Subasinghe","Sumathipala","Suriarachchi","Tennakoon",
    "Tilakaratne","Tilakawardane","Udalagama","Uda Malpitage","Udagedara",
    "Waidyaratne","Walpola","Wanigasekara","Weerakoon","Weerakkody","Weeramanthri",
    "Weerasekara","Weerasena","Weerasooria","Wijayaratne","Wijesinghe","Wijewantha",
    "Wijesooriya","Wijayasundara","Yakandawala","Yasaratne","Yatiwella","Yatawara"
]

# Extended artist name pool
ARTIST_NAMES_POOL = [
    # Legendary / Classic
    "W.D. Amaradeva","Nanda Malini","Victor Ratnayake","H.R. Jothipala",
    "Milton Mallawarachchi","Clarence Wijewardena","Annesley Malewana",
    "Gunadasa Kapuge","Malini Bulathsinghala","Amarasiri Peiris",
    "Karunarathna Divulgane","Premasiri Khemadasa","Sunil Ariyaratne",
    "Dharmadasa Walpola","Mohideen Baig","Edward Jayakody","Neela Wickremasinghe",
    "Angeline Gunathilaka","Desmond de Silva","Pandith Amaradeva",
    "Sunil Perera","Deepika Priyadarshani",
    # Contemporary Music
    "Yohani","Bathiya & Santhush","Rookantha Gunathilaka","Sanuka Wickramasinghe",
    "Ashanthi De Alwis","Dinupa Kodagoda","Iraj Weeraratne","Randhir",
    "Umaria","Sashika Nisansala","Chandani Hettiarachchi","Chitral Somapala",
    "Sangeeth Wijesuriya","Uresha Ravihari","Shanika Madumali","Umara Sinhawansa",
    "Dilki Uresha","Thushara Joshep","Damith Asanka","Lahiru Perera",
    "Duleeka Marapana","Haritha Wickramasinghe","Ashan Fernando","Indrachapa Liyanage",
    "Ridma Weerawardena","Nalin Perera","Sunil Edirisinghe","Ravibandu Vidyapathi",
    # Dance
    "Chitrasena","Vajira","Upeka","Navarasa","Rithma","Sandanari","Narthana",
    "Vimukthi Fernando","Sanjaya Navaratne","Dilani Weerakkody","Sampath Kolambage",
    "Raagini Thilakawardane","Pushpa Ranawake","Thilaka Siriwardena",
    # Film / Drama
    "Gamini Fonseka","Joe Abeywickrama","Ranjan Ramanayake","Malini Fonseka",
    "Sabitha Perera","Swarna Mallawarachchi","Jackson Anthony","Nadeesha Hemamali",
    "Sepalika Siriwardena","Anoja Weerasinghe","Jayalath Manoratne",
    "Ravindra Randeniya","Wijeratne Warakagoda","Bimal Jayakody","Sriyantha Mendis",
    "Dharmasena Pathiraja","Vasantha Obeysekera","Sumithra Peiris",
    "Asoka Handagama","Prasanna Vithanage","Vimukthi Jayasundara",
    # Tamil artists
    "M.S. Sethuraman","K. Subramaniam","Sundaramoorthy","Nirmala Devi",
    "Kavitha Rajendran","Ragavan Arts Collective","Tamil Arts Society",
    "Jaffna Cultural Ensemble","Northern Province Dancers",
    # Emerging / Youth
    "Omega X","Tasha Brava","FunkyG","The Quarks","Midnight Crew",
    "Akila Dunuvila","Theekshana Anuradha","Chamika Sirimanne","Shehan Kaushalya",
    "Vidusha Jayarathne","Kasun Kalhara","Pasan Liyanage","Roshani Ranawake",
    "Bhanuka Rajapaksha","Navodi Siriwardana","Tharaka Fernando","Dulani Anuradha",
    "Malintha Perera","Sachini Nimasha","Chathura Bandara","Eshan Liyanage",
    "Gimhana De Silva","Haritha Senanayake","Indika Weerasinghe","Janith Kumara",
    "Keshani Hettiarachchi","Lahiru Seneviratne","Menuka Gamage","Nimali Perera",
    "Osanda Jayawardena","Priyanga Dissanayake","Ruwani Senanayake","Sadun Perera",
    "Thisari Wijewardana","Udara Liyanage","Vimukthi Rathnayake","Waruni Mendis",
]

CITIES = [
    "colombo","kandy","galle","jaffna","negombo","trincomalee","batticaloa",
    "anuradhapura","polonnaruwa","badulla","nuwara_eliya","matara","kurunegala",
    "ratnapura","kegalle","matale","hambantota","ampara","puttalam","mannar",
    "vavuniya","mullaitivu","kilinochchi","monaragala","moneragala","gampaha",
    "kalutara","kalmunai","dambulla","sigiriya","hikkaduwa","unawatuna",
    "mirissa","tangalle","weligama","arugam_bay","ella","haputale","bandarawela",
    "nanu_oya","hatton","nawalapitiya","peradeniya","kadugannawa","mawanella",
    "avissawella","horana","panadura","moratuwa","dehiwala","mount_lavinia",
    "maharagama","nugegoda","boralesgamuwa","piliyandala","ragama","ja_ela",
    "wattala","kandana","nittambuwa","veyangoda","gampola","pussellawa",
    "talawakele","lindula","dickoya","maskeliya","norwood","glen_loch",
    "chilaw","puttalam","wennappuwa","marawila","nattandiya","dankotuwa",
    "kuliyapitiya","nikaweratiya","maho","giriulla","narammala","wariyapola",
    "dambulla","galewela","sigiriya","inamaluwa","habarana","medirigiriya",
    "polonnaruwa","kaduruwela","hingurakgoda","minneriya","giritale",
]

VENUES = [
    "Lionel Wendt Theatre","Nelum Pokuna Theatre","BMICH Grand Hall",
    "Colombo Hilton Grand Ballroom","Gangaramaya Temple Grounds",
    "Galle Face Green","Independence Square","Viharamahadevi Park Open Air Theatre",
    "University of Peradeniya Open Air Theatre","Kandy Lake Club",
    "Jaffna Public Library Hall","Galle Fort Gateway","Anuradhapura Cultural Centre",
    "Elphinstone Theatre","Tower Hall Theatre","Punchi Theatre Borella",
    "Lumbini Theatre Havelock Town","Regal Theatre Colombo","Navarangahala Borella",
    "Royal College Main Hall","Sugathadasa Indoor Stadium","Maharagama Youth Centre",
    "Cinnamon Grand Ballroom","Galadari Hotel Ballroom","National Museum Auditorium",
    "Colombo University Arts Theatre","Dalada Maligawa Kandy","Diyatha Uyana",
    "Waters Edge Battaramulla","Colombo Racecourse","Bishop's College Auditorium",
    "Ladies College Auditorium","Temple Trees Grounds","Sri Lanka Foundation Institute",
    "Kelaniya Raja Maha Vihara Grounds","Mount Lavinia Hotel Ballroom",
    "Cinnamon Lakeside Ballroom","Galle International Cricket Stadium",
    "Bentota Beach Resort Amphitheatre","Sigiriya Open Air Stage",
    "Kandy Esplanade","Nuwara Eliya Grand Hotel Ballroom","Trincomalee Beach Stage",
    "Jaffna Cultural Centre","Batticaloa Dutch Fort Grounds","Ratnapura City Hall",
    "Kurunegala Town Hall","Matara South Rotunda","Hambantota Cultural Centre",
    "Ampara Town Auditorium","Badulla Municipal Grounds","Haputale Tea Plantation Stage",
    "Ella Rock Open Stage","Dambulla Cave Temple Grounds","Polonnaruwa Ancient City Stage",
    "Anuradhapura Ruwanwelisaya Grounds","Independence Arcade Garden",
    "Shangri-La Hotel Colombo","Movenpick Hotel Colombo","Taj Samudra Ballroom",
    "Kingsbury Hotel Grand Ballroom","Colombo Port City Amphitheatre",
    "One Galle Face Mall Atrium","Liberty Plaza Atrium","Majestic City Auditorium",
]

ART_FORMS = ["music", "dance", "film", "drama"]

MUSIC_GENRES = [
    "virindu","vannam","kolam_songs","raban_music","jana_kavi","harvest_songs",
    "sinhala_classical","carnatic_sri_lankan_tamil","light_classical","classical_vocal",
    "traditional_baila","modern_baila","party_baila",
    "sinhala_pop","dance_pop","acoustic_pop","ballads","love_songs","sad_songs",
    "tamil_pop","tamil_melody","tamil_rap","gaana_style",
    "sinhala_rap","trap","drill",
    "sinhala_rock","hard_rock","alternative_rock","metal",
    "buddhist_devotional","bhakti_songs","catholic_hymns","islamic_nasheed",
    "folk_fusion","classical_fusion","ethnic_electronic","traditional_hip_hop_fusion",
    "film_songs","background_scores","orchestral_scores",
]
DANCE_GENRES = [
    "ves_dance","naiyandi","pantheru","uddekki","vannam_dance",
    "kolam_dance","sanni_yakuma","devil_dance","ritual_based","drum_centered",
    "bharatanatyam","wedding_dances","ceremonial_dances",
    "harvest_dances","village_festival_dances","new_year_dances",
    "sri_lankan_contemporary","interpretative","theatrical_dance",
    "hip_hop_dance","breaking","popping","locking","kpop_cover",
    "waltz","tango","salsa","cha_cha",
]
FILM_GENRES = [
    "mass_action","romantic_commercial","comedy_commercial","family_drama",
    "realism","social_realism","symbolic_cinema","festival_cinema",
    "civil_war_themes","social_justice","ethnic_conflict",
    "ancient_sri_lanka","colonial_era","biographical",
    "buddhist_films","hindu_mythology","jataka_tales",
    "civil_war_narratives","military_themes",
    "detective","police_procedural","crime_drama",
    "supernatural_horror","psychological_horror",
    "romantic_drama","romantic_comedy","avant_garde","low_budget_indie",
]
DRAMA_GENRES = [
    "nadagam","kolam_theatre","sokari",
    "social_drama","literary_adaptation","political_theatre",
    "family_drama","romance_series","crime_series","historical_series","political_series",
    "stage_musicals","opera","physical_theatre","immersive_theatre",
]

ART_FORM_GENRES = {
    "music": MUSIC_GENRES,
    "dance": DANCE_GENRES,
    "film": FILM_GENRES,
    "drama": DRAMA_GENRES,
}

MOODS = [
    "celebratory","spiritual","devotional","patriotic","romantic","energetic",
    "reflective","melancholic","sad","joyful","peaceful","intense","dramatic",
    "emotional","hopeful","inspirational","motivational","serious","dark",
    "mysterious","suspenseful","fearful","tense","angry","aggressive","powerful",
    "uplifting","ritualistic","ceremonial","heroic","mythological","traditional",
    "festive","spiritual_trance","cultural_pride","national_pride","rural_village",
    "urban_street","political_awareness","social_awareness","religious_reverence",
    "historical_nostalgia","party","danceable","chill","acoustic_warm","love_longing",
    "heartbreak","empowerment","meditative","dramatic_ballad","fusion_energy","rebel",
    "rhythmic","theatrical","expressive","graceful","fierce","sacred","competitive",
    "suspense","thriller_tension","social_commentary","comedy","tragic","nostalgic",
    "adventurous","humorous","satirical",
]

FESTIVALS = [
    "vesak","esala_perahera","poson","sinhala_new_year","thai_pongal","deepavali",
    "eid","christmas","vel_festival","navam_perahera","kelani_perahera",
    "kandy_perahera","kataragama_festival","poya_days","avurudu",
]

LANGUAGES = ["sinhala", "tamil", "english"]

# ─────────────────────────────────────────────────────────────────────────────
# REAL ARTISTS (15 canonical)
# ─────────────────────────────────────────────────────────────────────────────
REAL_ARTISTS = [
    {"name":"W.D. Amaradeva","art_forms":["music"],"genres":["sinhala_classical","light_classical"],"styles":["classical_semi_classical"],"language":["sinhala"],"city":"colombo","style":["traditional"],"mood_tags":["spiritual","devotional","patriotic"],"festivals":["vesak","poson"],"popularity":"established","follower_count":1500000,"verified":True,"era":"legend","is_real_artist":True},
    {"name":"Nanda Malini","art_forms":["music"],"genres":["sinhala_classical","folk_fusion"],"styles":["classical_semi_classical"],"language":["sinhala"],"city":"colombo","style":["traditional","contemporary"],"mood_tags":["patriotic","emotional","political_awareness"],"festivals":["sinhala_new_year"],"popularity":"established","follower_count":850000,"verified":True,"era":"legend","is_real_artist":True},
    {"name":"Yohani","art_forms":["music"],"genres":["sinhala_pop","dance_pop"],"styles":["sinhala_commercial"],"language":["sinhala","english"],"city":"colombo","style":["contemporary"],"mood_tags":["energetic","joyful","danceable","party"],"festivals":[],"popularity":"established","follower_count":5200000,"verified":True,"era":"contemporary","is_real_artist":True},
    {"name":"Bathiya & Santhush","art_forms":["music"],"genres":["sinhala_pop","acoustic_pop","ballads"],"styles":["sinhala_commercial"],"language":["sinhala","english"],"city":"colombo","style":["contemporary"],"mood_tags":["romantic","emotional","uplifting"],"festivals":[],"popularity":"established","follower_count":1200000,"verified":True,"era":"contemporary","is_real_artist":True},
    {"name":"Rookantha Gunathilaka","art_forms":["music"],"genres":["sinhala_pop","love_songs","ballads"],"styles":["sinhala_commercial"],"language":["sinhala"],"city":"colombo","style":["contemporary"],"mood_tags":["romantic","love_longing","emotional","acoustic_warm"],"festivals":[],"popularity":"established","follower_count":900000,"verified":True,"era":"contemporary","is_real_artist":True},
    {"name":"Victor Ratnayake","art_forms":["music"],"genres":["sinhala_classical","light_classical"],"styles":["classical_semi_classical"],"language":["sinhala"],"city":"kandy","style":["traditional"],"mood_tags":["patriotic","spiritual","emotional"],"festivals":["vesak","sinhala_new_year"],"popularity":"established","follower_count":750000,"verified":True,"era":"legend","is_real_artist":True},
    {"name":"Chitrasena","art_forms":["dance"],"genres":["kandyan_dance","ves_dance","vannam_dance"],"styles":["kandyan_dance"],"language":["sinhala"],"city":"colombo","style":["traditional"],"mood_tags":["ritualistic","sacred","theatrical","graceful"],"festivals":["esala_perahera","poson","vesak"],"popularity":"established","follower_count":400000,"verified":True,"era":"legend","is_real_artist":True},
    {"name":"Sanuka Wickramasinghe","art_forms":["music"],"genres":["sinhala_pop","acoustic_pop","love_songs"],"styles":["sinhala_commercial"],"language":["sinhala","english"],"city":"colombo","style":["contemporary"],"mood_tags":["romantic","emotional","acoustic_warm","chill"],"festivals":[],"popularity":"established","follower_count":620000,"verified":True,"era":"contemporary","is_real_artist":True},
    {"name":"Iraj Weeraratne","art_forms":["music"],"genres":["sinhala_rap","trap","sinhala_pop"],"styles":["hip_hop_rap"],"language":["sinhala","english"],"city":"colombo","style":["contemporary"],"mood_tags":["energetic","rebel","urban_street","party"],"festivals":[],"popularity":"established","follower_count":580000,"verified":True,"era":"contemporary","is_real_artist":True},
    {"name":"Ashanthi De Alwis","art_forms":["music"],"genres":["sinhala_pop","ballads","love_songs"],"styles":["sinhala_commercial"],"language":["sinhala","english"],"city":"colombo","style":["contemporary"],"mood_tags":["emotional","romantic","dramatic_ballad"],"festivals":[],"popularity":"established","follower_count":510000,"verified":True,"era":"contemporary","is_real_artist":True},
    {"name":"Chitral Somapala","art_forms":["music"],"genres":["sinhala_rock","hard_rock","alternative_rock"],"styles":["rock_alternative"],"language":["sinhala","english"],"city":"colombo","style":["contemporary"],"mood_tags":["intense","powerful","aggressive","energetic"],"festivals":[],"popularity":"established","follower_count":380000,"verified":True,"era":"contemporary","is_real_artist":True},
    {"name":"Gunadasa Kapuge","art_forms":["music"],"genres":["sinhala_classical","folk_fusion"],"styles":["classical_semi_classical","fusion"],"language":["sinhala"],"city":"kandy","style":["traditional","fusion"],"mood_tags":["patriotic","traditional","emotional"],"festivals":["sinhala_new_year","esala_perahera"],"popularity":"established","follower_count":450000,"verified":True,"era":"legend","is_real_artist":True},
    {"name":"Ravibandu Vidyapathi","art_forms":["music"],"genres":["classical_fusion","folk_fusion","ethnic_electronic"],"styles":["fusion"],"language":["sinhala","english"],"city":"colombo","style":["fusion"],"mood_tags":["spiritual","meditative","fusion_energy"],"festivals":["vesak"],"popularity":"established","follower_count":280000,"verified":True,"era":"contemporary","is_real_artist":True},
    {"name":"Malini Fonseka","art_forms":["film","drama"],"genres":["romantic_drama","family_drama","social_drama"],"styles":["commercial_cinema","modern_stage_drama"],"language":["sinhala"],"city":"colombo","style":["contemporary","traditional"],"mood_tags":["romantic","emotional","dramatic","theatrical"],"festivals":[],"popularity":"established","follower_count":680000,"verified":True,"era":"legend","is_real_artist":True},
    {"name":"Jackson Anthony","art_forms":["film","drama"],"genres":["historical_series","biographical","social_drama"],"styles":["historical_period","modern_stage_drama"],"language":["sinhala"],"city":"colombo","style":["contemporary"],"mood_tags":["heroic","dramatic","emotional","historical_nostalgia"],"festivals":[],"popularity":"established","follower_count":720000,"verified":True,"era":"contemporary","is_real_artist":True},
]


def _rng_name(ethnicity: str) -> str:
    if ethnicity == "sinhala":
        return f"{random.choice(SINHALA_FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    elif ethnicity == "tamil":
        return f"{random.choice(TAMIL_FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    else:
        return f"{random.choice(SINHALA_FIRST_NAMES + TAMIL_FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def generate_large_users(num_users: int = 10000) -> List[Dict[str, Any]]:
    """Generate large realistic Sri Lankan user pool."""
    print(f"  Generating {num_users:,} users...")
    users = []
    seen_names = set()

    for i in range(num_users):
        ethnicity = random.choices(["sinhala","tamil","other"], weights=[0.75,0.15,0.10])[0]
        # Generate a unique-enough name
        for _ in range(5):
            name = _rng_name(ethnicity)
            if name not in seen_names:
                seen_names.add(name)
                break

        if ethnicity == "sinhala":
            lang_prefs = random.choice([["sinhala"],["sinhala","english"]])
        elif ethnicity == "tamil":
            lang_prefs = random.choice([["tamil"],["tamil","english"]])
        else:
            lang_prefs = ["english"]

        city = random.choice(CITIES)
        num_interests = random.choices([1,2,3], weights=[0.50,0.35,0.15])[0]
        art_interests = random.sample(ART_FORMS, num_interests)
        culture_prefs = random.choices(["traditional","contemporary","fusion"],
                                       weights=[0.30,0.50,0.20],
                                       k=random.randint(1,2))
        mood_prefs = random.sample(MOODS, random.randint(2,5))
        activity_level = random.choices(["high","medium","low"], weights=[0.20,0.50,0.30])[0]

        users.append({
            "user_id": f"U{i:05d}",
            "name": name,
            "ethnicity": ethnicity,
            "language_preferences": lang_prefs,
            "city": city,
            "art_interests": art_interests,
            "culture_preferences": culture_prefs,
            "mood_preferences": mood_prefs,
            "activity_level": activity_level,
            "join_date": (datetime.now() - timedelta(days=random.randint(1,730))).isoformat(),
        })

    print(f"  ✓ {num_users:,} users created")
    return users


def generate_large_artists(num_synthetic: int = 485) -> List[Dict[str, Any]]:
    """Generate 15 real + num_synthetic synthetic artists = 500 total."""
    print(f"  Generating {15 + num_synthetic} artists (15 real + {num_synthetic} synthetic)...")
    artists = []

    # ---- 15 real artists first ----
    for i, ra in enumerate(REAL_ARTISTS):
        a = dict(ra)
        a["artist_id"] = f"A{i:04d}"
        artists.append(a)

    # ---- synthetic artists ----
    name_pool = [n for n in ARTIST_NAMES_POOL if n not in [a["name"] for a in artists]]
    random.shuffle(name_pool)

    for i in range(num_synthetic):
        idx = 15 + i
        art_form = random.choice(ART_FORMS)
        genres_pool = ART_FORM_GENRES[art_form]
        genres = random.sample(genres_pool, random.randint(1,3))
        language = random.choice([["sinhala"],["sinhala","english"],["tamil"],["tamil","english"],["english"]])

        # city bias per art form
        if art_form == "dance" and random.random() < 0.4:
            city = random.choice(["kandy","matale","nuwara_eliya"])
        elif "tamil" in language:
            city = random.choice(["jaffna","trincomalee","batticaloa","kilinochchi"])
        else:
            city = random.choice(CITIES)

        moods = random.sample(MOODS, random.randint(2,4))
        festivals = random.sample(FESTIVALS, random.randint(0,2))
        style = random.choices(["traditional","contemporary","fusion"], weights=[0.25,0.55,0.20])[0]
        popularity = random.choices(["emerging","mid_tier","established"], weights=[0.55,0.30,0.15])[0]
        follower_count = {
            "emerging": random.randint(100, 10000),
            "mid_tier": random.randint(10000, 100000),
            "established": random.randint(100000, 1000000),
        }[popularity]

        # name
        if i < len(name_pool):
            name = name_pool[i]
        else:
            eth = random.choices(["sinhala","tamil"], weights=[0.75,0.25])[0]
            name = _rng_name(eth)

        artists.append({
            "artist_id": f"A{idx:04d}",
            "name": name,
            "art_forms": [art_form],
            "genres": genres,
            "styles": genres[:1],  # first genre as style label
            "language": language,
            "city": city,
            "style": [style],
            "mood_tags": moods,
            "festivals": festivals,
            "popularity": popularity,
            "follower_count": follower_count,
            "verified": random.random() < 0.2,
            "is_real_artist": False,
        })

    print(f"  ✓ {len(artists)} artists created")
    return artists


def generate_large_events(artists: List[Dict], num_events: int = 2000) -> List[Dict[str, Any]]:
    """Generate 2,000 events linked to artists."""
    print(f"  Generating {num_events:,} events...")
    events = []
    base_date = datetime.now()

    for i in range(num_events):
        num_perf = random.choices([1,2,3,4], weights=[0.55,0.30,0.12,0.03])[0]
        perf_artists = random.sample(artists, min(num_perf, len(artists)))
        primary = perf_artists[0]

        # Derive event properties from artists
        art_forms = list({af for a in perf_artists for af in a["art_forms"]})
        genres = list({g for a in perf_artists for g in a["genres"]})
        languages = list({l for a in perf_artists for l in a["language"]})
        moods = list({m for a in perf_artists for m in a["mood_tags"]})

        city = primary.get("city", random.choice(CITIES))
        venue = random.choice(VENUES)

        event_type = random.choices(
            ["concert","performance","exhibition","workshop","festival","competition","ceremony"],
            weights=[0.35,0.25,0.10,0.10,0.10,0.05,0.05]
        )[0]

        days_offset = random.randint(-180, 365)
        event_date = (base_date + timedelta(days=days_offset)).isoformat()

        ticket_price = round(random.choice([0, 500, 1000, 1500, 2000, 2500, 3000, 5000, 10000]), -2)

        name_templates = [
            f"{event_type.title()} of {primary['name']}",
            f"{primary['name']} Live",
            f"An Evening with {primary['name']}",
            f"Night of {primary['name']}",
            f"The {primary['name']} Show",
            f"Celebration of {primary['name']}",
            f"{primary['name']} in Concert",
            f"Live Performance – {primary['name']}",
        ]

        events.append({
            "event_id": f"E{i:05d}",
            "name": random.choice(name_templates),
            "event_type": event_type,
            "art_forms": art_forms,
            "genres": genres[:5],  # cap to 5
            "languages": languages,
            "moods": moods[:4],
            "artist_ids": [a["artist_id"] for a in perf_artists],
            "city": city,
            "venue": venue,
            "date": event_date,
            "ticket_price": ticket_price,
            "capacity": random.choice([100,200,300,500,750,1000,2000,5000,10000]),
            "is_free": ticket_price == 0,
        })

    print(f"  ✓ {len(events)} events created")
    return events


def generate_large_interactions(
    users: List[Dict],
    artists: List[Dict],
    events: List[Dict],
) -> Dict[str, list]:
    """
    Generate follows and attends interactions.

    Strategy:
    - Activity-based follow counts: high=15-30, medium=6-15, low=1-6
    - Affinity-biased: users follow artists matching their interests + moods
    - Attends based on followed artists performing at nearby events
    """
    print("  Generating interactions (follows & attends)...")

    artist_by_id = {a["artist_id"]: a for a in artists}
    event_by_id  = {e["event_id"]:  e for e in events}

    # Pre-index artists by art_form for fast lookup
    af_index: Dict[str, List[str]] = {af: [] for af in ART_FORMS}
    for a in artists:
        for af in a["art_forms"]:
            af_index[af].append(a["artist_id"])

    follows = []
    attends = []
    base_date = datetime.now()

    follow_count_range = {"high": (15,30), "medium": (6,15), "low": (1,6)}

    for user in users:
        uid = user["user_id"]
        activity = user["activity_level"]
        lo, hi = follow_count_range[activity]
        n_follows = random.randint(lo, hi)

        # Candidate artists: prefer matching art_form interests
        candidates = []
        for interest in user["art_interests"]:
            candidates.extend(af_index.get(interest, []))
        # Deduplicate, pad with random if needed
        candidates = list(set(candidates))
        if len(candidates) < n_follows:
            extra = [a["artist_id"] for a in artists if a["artist_id"] not in candidates]
            random.shuffle(extra)
            candidates.extend(extra[:n_follows - len(candidates)])

        chosen_artist_ids = random.sample(candidates, min(n_follows, len(candidates)))

        for aid in chosen_artist_ids:
            ts = base_date - timedelta(days=random.randint(0, 365), hours=random.randint(0,23))
            follows.append({"user_id": uid, "artist_id": aid, "timestamp": ts.isoformat()})

        # Attends: events where followed artists perform, prefer same city
        user_city = user.get("city", "")
        eligible_events = [
            e for e in events
            if any(aid in chosen_artist_ids for aid in e.get("artist_ids", []))
        ]
        # Also include same-city events
        city_events = [e for e in events if e.get("city") == user_city]
        eligible_events = list({e["event_id"]: e for e in eligible_events + city_events[:10]}.values())

        n_attends = random.randint(0, min(8, len(eligible_events)))
        attended = random.sample(eligible_events, n_attends)
        for ev in attended:
            ts = base_date - timedelta(days=random.randint(0, 180), hours=random.randint(0,23))
            attends.append({"user_id": uid, "event_id": ev["event_id"], "timestamp": ts.isoformat()})

    print(f"  ✓ {len(follows):,} follows, {len(attends):,} attends")
    return {"follows": follows, "attends": attends, "likes_genre": [], "rates": []}


def generate_large_dataset(
    num_users: int = 10000,
    num_artists: int = 500,
    num_events: int = 2000,
    output_dir: str = None,
) -> Dict:
    """
    Master generator. Returns dataset dict and saves to disk.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "sample_dataset")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(" RASASWADAYA — LARGE-SCALE DATASET GENERATOR")
    print("=" * 60)
    print(f" Target: {num_users:,} users | {num_artists} artists | {num_events:,} events")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    users   = generate_large_users(num_users)
    artists = generate_large_artists(num_artists - 15)  # 15 real + rest synthetic
    events  = generate_large_events(artists, num_events)
    interactions = generate_large_interactions(users, artists, events)

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "num_users": len(users),
        "num_artists": len(artists),
        "num_events": len(events),
        "num_follows": len(interactions["follows"]),
        "num_attends": len(interactions["attends"]),
        "real_artists": 15,
        "synthetic_artists": len(artists) - 15,
        "generator_version": "2.0_large_scale",
    }

    dataset = {
        "users": users,
        "artists": artists,
        "events": events,
        "interactions": interactions,
        "metadata": metadata,
    }

    # Save
    json_path = os.path.join(output_dir, "rasaswadaya_large_dataset.json")
    pkl_path  = os.path.join(output_dir, "rasaswadaya_large_dataset.pkl")

    print(f"\n  Saving JSON → {json_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"  Saving Pickle → {pkl_path}")
    with open(pkl_path, "wb") as f:
        pickle.dump(dataset, f)

    print()
    print("=" * 60)
    print(" ✅ LARGE DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Users:    {len(users):,}")
    print(f"  Artists:  {len(artists):,}  (15 real + {len(artists)-15} synthetic)")
    print(f"  Events:   {len(events):,}")
    print(f"  Follows:  {len(interactions['follows']):,}")
    print(f"  Attends:  {len(interactions['attends']):,}")
    print(f"  Total interactions: {len(interactions['follows'])+len(interactions['attends']):,}")
    print("=" * 60)

    return dataset


if __name__ == "__main__":
    generate_large_dataset()
