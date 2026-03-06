"""Download tag database from Danbooru or e621 API for autocomplete."""
import csv
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json

SCRIPT_INFO = {
    'name': 'Populate Caption Tags Autocomplete',
    'description': (
        'Download tags from Danbooru or e621 API to tags.csv '
        'for autocomplete in the Caption tab. '
        'Downloads all tags with post_count >= min_posts.'
    ),
    'parameters': [
        {
            'name': 'source',
            'type': 'combo',
            'label': 'Tag source',
            'default': 'danbooru',
            'options': ['danbooru', 'e621'],
        },
        {
            'name': 'min_posts',
            'type': 'str',
            'label': 'Min post count',
            'default': '100',
            'placeholder': '100',
        },
        {
            'name': 'output',
            'type': 'str',
            'label': 'Output file',
            'default': 'tags.csv',
            'placeholder': 'tags.csv',
        },
    ],
}

SOURCES = {
    'danbooru': {
        'url': 'https://danbooru.donmai.us/tags.json',
        'limit': 1000,
        'delay': 0.12,      # 10 req/s max
        'user_agent': 'LabelToDataset/1.0',
    },
    'e621': {
        'url': 'https://e621.net/tags.json',
        'limit': 1000,
        'delay': 0.6,       # 2 req/s max
        'user_agent': 'LabelToDataset/1.0 (tag-autocomplete-download)',
    },
}


def check_available() -> tuple[bool, str]:
    return True, ''


def _fetch_json(url: str, user_agent: str) -> list[dict]:
    req = Request(url)
    req.add_header('User-Agent', user_agent)
    req.add_header('Accept', 'application/json')
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8'))


def run(params: dict, progress_callback) -> None:
    source_name = params.get('source', 'danbooru').strip().lower()
    source = SOURCES.get(source_name)
    if source is None:
        raise ValueError(
            f'Unknown source "{source_name}". '
            f'Available: {", ".join(SOURCES.keys())}')

    try:
        min_posts = int(params.get('min_posts', '100'))
    except ValueError:
        min_posts = 100

    output = Path(params.get('output', 'tags.csv').strip() or 'tags.csv')

    base_url = source['url']
    limit = source['limit']
    delay = source['delay']
    ua = source['user_agent']

    # Keyset pagination: order by count desc, then page through by id
    # Strategy: fetch pages ordered by post_count desc until we hit min_posts
    all_tags = []
    page = 1
    total_fetched = 0

    progress_callback(0, 0, f'Downloading tags from {source_name}...')

    while True:
        url = (f'{base_url}?limit={limit}&page={page}'
               f'&search[order]=count'
               f'&search[hide_empty]=true')

        try:
            tags = _fetch_json(url, ua)
        except HTTPError as e:
            if e.code == 410:
                # Past page limit — switch to done
                break
            if e.code == 429:
                progress_callback(total_fetched, 0,
                                  'Rate limited, waiting 5s...')
                time.sleep(5)
                continue
            raise ValueError(f'API error {e.code}: {e.reason}')
        except (URLError, OSError) as e:
            raise ValueError(f'Network error: {e}')

        if not tags:
            break

        # Check if we've gone below min_posts threshold
        below_threshold = False
        for tag in tags:
            pc = tag.get('post_count', 0)
            if pc < min_posts:
                below_threshold = True
                break
            all_tags.append(tag)

        total_fetched = len(all_tags)
        progress_callback(total_fetched, 0,
                          f'Downloaded {total_fetched} tags '
                          f'(page {page})...')

        if below_threshold:
            break

        page += 1
        time.sleep(delay)

    if not all_tags:
        progress_callback(0, 0, 'No tags found above threshold')
        return

    # Write CSV in danbooru-compatible format: name,category,count,"aliases"
    progress_callback(total_fetched, total_fetched,
                      f'Writing {len(all_tags)} tags to {output}...')

    with open(output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for tag in all_tags:
            name = tag.get('name', '')
            category = tag.get('category', 0)
            post_count = tag.get('post_count', 0)
            # API doesn't return aliases in bulk; leave empty
            writer.writerow([name, category, post_count, ''])

    size_mb = os.path.getsize(output) / (1024 * 1024)
    progress_callback(
        total_fetched, total_fetched,
        f'Done — {len(all_tags)} tags from {source_name} '
        f'(>= {min_posts} posts, {size_mb:.1f} MB)')
