import time

from google_play_scraper import Sort, reviews

def get_reviews(app_id, max_reviews):
    
    rows, token = [], None
    while len(rows) < max_reviews:
        batch, token = reviews(
            app_id, lang="en", country="au", sort=Sort.NEWEST,
            count=min(200, max_reviews - len(rows)), continuation_token=token,
        )
        if not batch:
            break
        rows.extend(batch)
        if token is None or token.token is None:
            break
        time.sleep(1)  # need time between requests


    return rows