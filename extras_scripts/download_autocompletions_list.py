"""Download tag database from HuggingFace for autocomplete."""
import os
from pathlib import Path

REPO_ID = 'deepghs/site_tags'

SOURCES = [
    'anime-pictures.net',
    'danbooru.donmai.us',
    'e621.net',
    'en.pixiv.net',
    'gelbooru.com',
    'hypnohub.net',
    'konachan.com',
    'konachan.net',
    'pixiv.net',
    'rule34.xxx',
    'safebooru.donmai.us',
    'wallhaven.cc',
    'xbooru.com',
    'yande.re',
]

SCRIPT_INFO = {
    'name': 'Download Autocompletions List',
    'description': (
        'Download a tag database from HuggingFace (deepghs/site_tags) '
        'for autocomplete in the Caption tab. '
        'Filters by min post count and saves as parquet to autocompletions/.'
    ),
    'project_url': 'https://huggingface.co/datasets/deepghs/site_tags',
    'parameters': [
        {
            'name': 'source',
            'type': 'combo',
            'label': 'Tag source',
            'default': 'danbooru.donmai.us',
            'options': SOURCES,
        },
        {
            'name': 'min_posts',
            'type': 'str',
            'label': 'Min post count',
            'default': '100',
            'placeholder': '100',
        },
    ],
}


def check_available() -> tuple[bool, str]:
    try:
        import huggingface_hub  # noqa: F401
        import pyarrow  # noqa: F401
        return True, ''
    except ImportError as e:
        return False, f'Missing dependency: {e.name}'


def run(params: dict, progress_callback) -> None:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    import pyarrow as pa

    source = params.get('source', 'danbooru.donmai.us').strip()
    if source not in SOURCES:
        raise ValueError(
            f'Unknown source "{source}". '
            f'Available: {", ".join(SOURCES)}')

    try:
        min_posts = int(params.get('min_posts', '100'))
    except ValueError:
        min_posts = 100

    # Download from HuggingFace
    remote_path = f'{source}/tags.parquet'
    progress_callback(0, 0, f'Downloading {remote_path} from HuggingFace...')

    local_path = hf_hub_download(
        REPO_ID, remote_path, repo_type='dataset')

    # Read and filter
    progress_callback(0, 0, 'Reading and filtering tags...')
    table = pq.read_table(local_path, columns=['name', 'category', 'post_count'])

    # Filter by min post count
    mask = pa.compute.greater_equal(table.column('post_count'), min_posts)
    table = table.filter(mask)

    # Sort by post_count descending for faster search later
    indices = pa.compute.sort_indices(
        table, sort_keys=[('post_count', 'descending')])
    table = table.take(indices)

    total_tags = table.num_rows
    if total_tags == 0:
        progress_callback(0, 0, 'No tags found above threshold')
        return

    # Save to autocompletions/
    output_dir = Path('autocompletions')
    output_dir.mkdir(exist_ok=True)

    # Use source name without TLD as filename (e.g. "danbooru" from "danbooru.donmai.us")
    filename = source.split('.')[0] + '.parquet'
    output = output_dir / filename

    progress_callback(total_tags, total_tags,
                      f'Writing {total_tags} tags to {output}...')
    pq.write_table(table, output)

    size_mb = os.path.getsize(output) / (1024 * 1024)
    progress_callback(
        total_tags, total_tags,
        f'Done — {total_tags} tags from {source} '
        f'(>= {min_posts} posts, {size_mb:.1f} MB)')
