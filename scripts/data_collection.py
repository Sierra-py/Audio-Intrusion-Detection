""" Documentation
This script downloads data from Freesound api and store it in the data/raw folder category wise.
"""

import os
import requests
import time
from config.config import FREESOUND_API_KEY

""" Some Comments:
Normal sounds to consider - wind, rain, birds, cricket

Abnormal Sounds to consider - glass breaking, gunshots, chainsaw, metal scraping


# Custom url to  find actual sounds of the forests

# https://freesound.org/apiv2/search/text/?token=API_KEY&page_size=150&fields=id,name,duration,previews,download,tags&filter=tag:(forest%20AND%20field-recording)-tag:(music%20OR%20voice%20OR%20talking%20OR%20glitch%20OR%20kids%20OR%20car%20OR%20airplane)%20duration:[1%20TO%2015]&similar_to=611605

# A url for finding sounds similar to any sound id with duration filter
# 
# https://freesound.org/apiv2/search/text/?token=5Y9lwpGiNPQYYeGM1vhCp3x9IFOt0yDDQQVNuIHa&page_size=150&fields=id,name,duration,previews,tags,download&filter=tag:(*)%20duration:[1%20TO%2015]&similar_to=51034 

Sound IDs:
    Normal:
        Wind - 51034
        Rain - 50059
        Birds - 20783
        crickets - 403294
        thunder - 21887
    Abnormal:
        Gunshots: 180961
        Glass Breaking: 491051
        ChainSaw: 505192

"""

URL = "https://freesound.org/apiv2"

CATEGORIES = {
    "wind": 51034,
    "rain": 50059,
    "birds": 20783,
    "crickets": 403294,
    "gunshots": 180961,
    "glass_breaking": 491051,
    "chainsaw": 505192,
    "thunder": 21887,
    
}

def search_sounds(similar_id, num_results=150):
    url = f"{URL}/search/text"

    params = {
        "token": FREESOUND_API_KEY,
        "fields":"id,name,duration,previews,download",
        "filter": "tag:(*) duration:[1 TO 10]",
        "page_size": num_results,
        "similar_to": similar_id,
        "format": "json"        
    }

    response = requests.get(url, params)
    response.raise_for_status()
    return response.json()["results"]

def download_sounds(sound, category):

    save_dir = f"data/raw/{category}"
    os.makedirs(save_dir, exist_ok=True)

    preview_url = sound["previews"]["preview-hq-mp3"]
    filepath = f"{save_dir}/{sound["id"]}.mp3"

    if os.path.exists(filepath):
        return 

    try:
        audio = requests.get(preview_url, headers={"Authorization": f"Token {FREESOUND_API_KEY}"})
        audio.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(audio.content)
        time.sleep(0.5)
    except Exception as e:
        print(f"An error occured **{type(e).__name__}** - {e}")
        time.sleep(5) # giving it time to cool down

def collect_all():
    for category, id in CATEGORIES.items():
        print(f"Fetching {category}...")
        sounds = search_sounds(id)
        print(f"{len(sounds)} sounds of {category}")
        for sound in sounds:
            download_sounds(sound, category)
        print("Done.")

if __name__ == "__main__":
    collect_all()
