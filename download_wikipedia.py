#!/usr/bin/env python3
"""
Download and process English Wikipedia for n-gram model training.
Uses Wikipedia dumps and processes them efficiently.
"""

import os
import bz2
import re
import json
import time
import requests
from xml.etree import ElementTree as ET
from collections import defaultdict
from typing import Iterator, Tuple
import pickle


class WikipediaDownloader:
    """Download and process Wikipedia dumps."""

    def __init__(self, data_dir: str = "wikipedia_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Use smaller Wikipedia extract for faster processing
        # Full dump is ~20GB compressed, ~100GB uncompressed
        self.dump_url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream1.xml-p1p41242.bz2"
        # This is the first chunk (~200MB compressed) containing ~40K articles

        self.dump_file = os.path.join(data_dir, "enwiki-chunk1.xml.bz2")
        self.processed_file = os.path.join(data_dir, "wikipedia_text.txt")
        self.ngram_file = os.path.join(data_dir, "wikipedia_ngrams.pkl")

    def download_dump(self, force: bool = False):
        """Download Wikipedia dump if not already present."""
        if os.path.exists(self.dump_file) and not force:
            print(f"✓ Wikipedia dump already exists: {self.dump_file}")
            return

        print(f"Downloading Wikipedia dump from {self.dump_url}")
        print("This is a ~200MB file containing ~40,000 articles...")

        response = requests.get(self.dump_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(self.dump_file, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"Progress: {percent:.1f}% ({downloaded/1024/1024:.1f}MB)", end='\r')

        print(f"\n✓ Downloaded to {self.dump_file}")

    def extract_text_from_dump(self) -> Iterator[str]:
        """Extract article text from Wikipedia XML dump."""
        print(f"Extracting text from {self.dump_file}...")

        # Parse compressed XML
        with bz2.open(self.dump_file, 'rt', encoding='utf-8') as f:
            # Use iterative parsing to handle large files
            context = ET.iterparse(f, events=('start', 'end'))
            context = iter(context)
            event, root = next(context)

            article_count = 0
            for event, elem in context:
                if event == 'end' and elem.tag.endswith('page'):
                    # Extract title and text
                    title_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.10/}title')
                    text_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.10/}text')

                    if title_elem is not None and text_elem is not None:
                        title = title_elem.text
                        text = text_elem.text or ""

                        # Skip redirects and special pages
                        if not text.startswith('#REDIRECT') and ':' not in title:
                            # Clean wiki markup (simplified)
                            text = self.clean_wiki_text(text)
                            if text:
                                article_count += 1
                                if article_count % 1000 == 0:
                                    print(f"Processed {article_count} articles...", end='\r')
                                yield text

                    # Clear the element to save memory
                    elem.clear()
                    root.clear()

            print(f"\n✓ Extracted text from {article_count} articles")

    def clean_wiki_text(self, text: str) -> str:
        """Clean Wikipedia markup from text."""
        # Remove wiki markup elements
        text = re.sub(r'\{\{[^}]+\}\}', '', text)  # Remove templates
        text = re.sub(r'\[\[(?:[^|\]]+\|)?([^\]]+)\]\]', r'\1', text)  # Extract link text
        text = re.sub(r"'''?", '', text)  # Remove bold/italic
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

        # Extract first meaningful paragraph (skip infoboxes)
        lines = text.split('.')
        for line in lines[:5]:  # Check first 5 sentences
            if len(line) > 50:  # Meaningful content
                return line + '.'

        return text[:500] if len(text) > 500 else text

    def process_wikipedia(self):
        """Process Wikipedia dump into text file."""
        if os.path.exists(self.processed_file):
            print(f"✓ Processed text already exists: {self.processed_file}")
            return

        print("Processing Wikipedia articles...")

        with open(self.processed_file, 'w', encoding='utf-8') as out_file:
            article_count = 0
            for text in self.extract_text_from_dump():
                out_file.write(text + '\n')
                article_count += 1

        print(f"✓ Processed {article_count} articles to {self.processed_file}")

        # Print file size
        size_mb = os.path.getsize(self.processed_file) / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")


class WikipediaNGramBuilder:
    """Build n-gram model from Wikipedia text."""

    def __init__(self, text_file: str, n: int = 3):
        self.text_file = text_file
        self.n = n
        self.ngrams = defaultdict(lambda: defaultdict(int))
        self.vocab = set()

    def build_ngrams(self, max_articles: int = None):
        """Build n-gram model from text file."""
        print(f"Building {self.n}-gram model from {self.text_file}...")

        article_count = 0
        total_ngrams = 0

        with open(self.text_file, 'r', encoding='utf-8') as f:
            for line in f:
                if max_articles and article_count >= max_articles:
                    break

                # Tokenize (simple word splitting)
                tokens = line.lower().split()

                # Add to vocabulary
                self.vocab.update(tokens)

                # Extract n-grams
                for i in range(len(tokens) - self.n + 1):
                    context = tuple(tokens[i:i + self.n - 1])
                    next_token = tokens[i + self.n - 1]
                    self.ngrams[context][next_token] += 1
                    total_ngrams += 1

                article_count += 1
                if article_count % 1000 == 0:
                    print(f"Processed {article_count} articles, {total_ngrams:,} n-grams...", end='\r')

        print(f"\n✓ Built n-gram model:")
        print(f"  Articles: {article_count:,}")
        print(f"  Unique n-grams: {len(self.ngrams):,}")
        print(f"  Vocabulary size: {len(self.vocab):,}")
        print(f"  Total n-grams: {total_ngrams:,}")

    def save_model(self, output_file: str):
        """Save n-gram model to file."""
        print(f"Saving n-gram model to {output_file}...")

        model_data = {
            'n': self.n,
            'ngrams': dict(self.ngrams),
            'vocab': list(self.vocab),
            'metadata': {
                'source': self.text_file,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'unique_contexts': len(self.ngrams),
                'vocab_size': len(self.vocab)
            }
        }

        with open(output_file, 'wb') as f:
            pickle.dump(model_data, f)

        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"✓ Saved model ({size_mb:.1f} MB)")

    def get_top_ngrams(self, limit: int = 20):
        """Get most frequent n-grams."""
        all_ngrams = []
        for context, next_tokens in self.ngrams.items():
            for token, count in next_tokens.items():
                all_ngrams.append((' '.join(context) + ' ' + token, count))

        all_ngrams.sort(key=lambda x: x[1], reverse=True)
        return all_ngrams[:limit]


def main():
    """Download Wikipedia and build n-gram model."""
    print("="*70)
    print("WIKIPEDIA N-GRAM MODEL BUILDER")
    print("="*70)

    # Step 1: Download Wikipedia
    downloader = WikipediaDownloader()

    print("\n1. DOWNLOADING WIKIPEDIA")
    print("-"*40)
    downloader.download_dump()

    # Step 2: Process Wikipedia
    print("\n2. PROCESSING WIKIPEDIA")
    print("-"*40)
    downloader.process_wikipedia()

    # Step 3: Build n-gram models
    print("\n3. BUILDING N-GRAM MODELS")
    print("-"*40)

    # Build different n-gram models
    for n in [2, 3, 4]:
        print(f"\nBuilding {n}-gram model...")

        builder = WikipediaNGramBuilder(
            downloader.processed_file,
            n=n
        )

        # Build from first 10,000 articles (for speed)
        builder.build_ngrams(max_articles=10000)

        # Save model
        output_file = os.path.join(
            downloader.data_dir,
            f"wikipedia_{n}grams.pkl"
        )
        builder.save_model(output_file)

        # Show sample n-grams
        print(f"\nTop {n}-grams:")
        for ngram, count in builder.get_top_ngrams(10):
            print(f"  '{ngram}': {count:,}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("✓ Wikipedia data downloaded and processed")
    print("✓ N-gram models built (2-gram, 3-gram, 4-gram)")
    print(f"✓ Models saved to {downloader.data_dir}/")
    print("\nNext steps:")
    print("1. Load models in notebook for experimentation")
    print("2. Test lightweight grounding with real Wikipedia n-grams")
    print("3. Compare with synthetic n-gram models")


if __name__ == "__main__":
    main()