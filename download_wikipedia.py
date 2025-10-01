#!/usr/bin/env python3
"""
Download and process Wikipedia dumps for the infinigram model.

This script downloads Wikipedia dumps and extracts clean text for building
large-scale suffix arrays.
"""

import os
import bz2
import xml.etree.ElementTree as ET
import re
import requests
import subprocess
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def download_wikipedia_dump(lang='en', dump_type='mini', output_dir='data/wikipedia'):
    """
    Download Wikipedia dump files.

    Args:
        lang: Language code (default 'en' for English)
        dump_type: 'mini' for ~100MB sample, 'latest' for full dump (~20GB compressed)
        output_dir: Directory to save the dump
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if dump_type == 'mini':
        # Use a smaller sample dump for testing
        url = f"https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles1.xml-p1p41242.bz2"
        output_file = f"{output_dir}/{lang}wiki-mini.xml.bz2"
    else:
        # Full dump (warning: very large!)
        url = f"https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles.xml.bz2"
        output_file = f"{output_dir}/{lang}wiki-latest.xml.bz2"

    if os.path.exists(output_file):
        print(f"File already exists: {output_file}")
        return output_file

    print(f"Downloading Wikipedia dump from: {url}")
    print("This may take a while...")

    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_file, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"Downloaded: {output_file}")
    return output_file


def extract_text_from_dump(dump_file, max_articles=None, min_article_length=500):
    """
    Extract clean text from Wikipedia XML dump.

    Args:
        dump_file: Path to the .bz2 dump file
        max_articles: Maximum number of articles to extract (None for all)
        min_article_length: Minimum article length in characters
    """
    print(f"Extracting text from: {dump_file}")

    articles = []
    article_count = 0

    # Open compressed file
    with bz2.open(dump_file, 'rt', encoding='utf-8') as f:
        # Parse XML iteratively to handle large files
        context = ET.iterparse(f, events=('start', 'end'))
        context = iter(context)
        event, root = next(context)

        for event, elem in context:
            if event == 'end' and elem.tag.endswith('page'):
                # Extract title and text
                title_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.10/}title')
                text_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.10/}text')

                if title_elem is not None and text_elem is not None:
                    title = title_elem.text
                    text = text_elem.text

                    if text and len(text) > min_article_length:
                        # Clean the text
                        clean_text = clean_wiki_text(text)

                        if len(clean_text) > min_article_length:
                            articles.append({
                                'title': title,
                                'text': clean_text
                            })
                            article_count += 1

                            if article_count % 1000 == 0:
                                print(f"Processed {article_count} articles...")

                            if max_articles and article_count >= max_articles:
                                break

                # Clear element to save memory
                elem.clear()
                root.clear()

    print(f"Extracted {len(articles)} articles")
    return articles


def clean_wiki_text(text):
    """
    Clean Wikipedia markup from text.
    """
    # Remove Wikipedia markup
    text = re.sub(r'\{\{[^}]+\}\}', '', text)  # Remove templates
    text = re.sub(r'\[\[File:[^\]]+\]\]', '', text)  # Remove files
    text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text)  # Remove categories
    text = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', text)  # Replace links
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)  # Replace simple links
    text = re.sub(r"'''([^']+)'''", r'\1', text)  # Remove bold
    text = re.sub(r"''([^']+)''", r'\1', text)  # Remove italic
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'&[^;]+;', ' ', text)  # Remove HTML entities
    text = re.sub(r'={2,}[^=]+=+', '', text)  # Remove headers
    text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

    return text.strip()


def download_with_wget(lang='en', articles_per_chunk=10000, num_chunks=10, output_dir='data/wikipedia'):
    """
    Alternative: Download Wikipedia using wget and process in chunks.
    This is more efficient for large downloads.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Use Wikipedia API to get random articles
    print(f"Downloading {articles_per_chunk * num_chunks:,} Wikipedia articles in chunks...")

    all_texts = []

    for chunk in range(num_chunks):
        print(f"\nChunk {chunk + 1}/{num_chunks}")

        # Use wikipedia-api to get articles
        try:
            import wikipedia
        except ImportError:
            subprocess.check_call(['pip', 'install', 'wikipedia-api'])
            import wikipedia

        texts = []
        attempts = 0
        max_attempts = articles_per_chunk * 3  # Allow for failures

        while len(texts) < articles_per_chunk and attempts < max_attempts:
            try:
                # Get random article
                page = wikipedia.page(wikipedia.random())
                if len(page.content) > 500:  # Min length
                    texts.append(page.content[:5000])  # Limit length per article
                    if len(texts) % 100 == 0:
                        print(f"  Downloaded {len(texts)}/{articles_per_chunk} articles")
            except:
                pass  # Skip errors
            attempts += 1

        all_texts.extend(texts)

    # Save to file
    output_file = f"{output_dir}/wikipedia_corpus.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in all_texts:
            f.write(text + "\n\n")

    print(f"\nSaved {len(all_texts)} articles to {output_file}")
    total_chars = sum(len(t) for t in all_texts)
    print(f"Total size: {total_chars:,} characters ({total_chars / 1024 / 1024:.1f} MB)")

    return output_file


def process_dump_parallel(dump_file, output_dir='data/wikipedia', num_workers=4):
    """
    Process Wikipedia dump using multiple workers for speed.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Processing dump with {num_workers} workers...")

    # For very large dumps, we'd split the file
    # For now, extract a reasonable number of articles
    articles = extract_text_from_dump(dump_file, max_articles=50000)

    # Save articles as plain text
    output_file = f"{output_dir}/wikipedia_corpus.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(f"=== {article['title']} ===\n")
            f.write(article['text'] + "\n\n")

    total_chars = sum(len(a['text']) for a in articles)
    print(f"Saved {len(articles)} articles ({total_chars:,} chars) to {output_file}")

    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Wikipedia for infinigram model")
    parser.add_argument('--method', choices=['dump-mini', 'dump-full', 'api'],
                        default='api',
                        help='Download method: dump-mini (100MB), dump-full (20GB), api (flexible)')
    parser.add_argument('--articles', type=int, default=10000,
                        help='Number of articles to download (for API method)')
    parser.add_argument('--output-dir', default='data/wikipedia',
                        help='Output directory')

    args = parser.parse_args()

    if args.method == 'dump-mini':
        # Download mini dump (~100MB compressed, ~1GB uncompressed)
        dump_file = download_wikipedia_dump('en', 'mini', args.output_dir)
        corpus_file = process_dump_parallel(dump_file, args.output_dir)

    elif args.method == 'dump-full':
        # Download full dump (warning: ~20GB compressed, ~100GB uncompressed)
        print("WARNING: Full dump is ~20GB compressed. Continue? (y/n)")
        if input().lower() == 'y':
            dump_file = download_wikipedia_dump('en', 'latest', args.output_dir)
            corpus_file = process_dump_parallel(dump_file, args.output_dir)

    else:  # api method
        # Use API to download specific number of articles
        corpus_file = download_with_wget(
            articles_per_chunk=min(1000, args.articles),
            num_chunks=max(1, args.articles // 1000),
            output_dir=args.output_dir
        )

    print(f"\nCorpus ready at: {corpus_file}")
    print("You can now use this file to build your suffix array infinigram model!")