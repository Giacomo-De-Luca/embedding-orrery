"""
Export WordNet to TSV file for embedding.

This script extracts all word senses from WordNet and exports them to a
tab-separated values (TSV) file with one row per word-sense combination.

Output format:
    word    pos    synset_id    definition    examples
"""

import csv
import argparse
import os
from tqdm import tqdm

try:
    from .wordnet_parser import WordNetParser
except ImportError:
    from wordnet_parser import WordNetParser


# Default paths
WORDNET_XML_PATH = "interpretability/resources/english-wordnet-2024.xml"
DEFAULT_OUTPUT_PATH = "interpretability/resources/wordnet_senses.tsv"


def export_wordnet_to_tsv(output_path: str, xml_path: str = WORDNET_XML_PATH):
    """
    Export all WordNet senses to a TSV file.

    Args:
        output_path: Path to write the TSV file
        xml_path: Path to the WordNet XML file
    """
    print("=" * 70)
    print("WordNet TSV Export")
    print("=" * 70)

    # Check WordNet file
    if not os.path.exists(xml_path):
        print(f"Error: WordNet XML file not found at: {xml_path}")
        return

    # Initialize and parse WordNet
    print(f"\nLoading WordNet from: {xml_path}")
    wn = WordNetParser(xml_path)
    wn.parse()

    stats = wn.get_stats()
    print(f"  Words: {stats['total_words']:,}")
    print(f"  Synsets: {stats['total_synsets']:,}")
    print(f"  Senses: {stats['total_senses']:,}")

    # Export to TSV
    print(f"\nExporting to: {output_path}")

    row_count = 0
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['word', 'pos', 'synset_id', 'definition', 'examples'])

        all_words = wn.get_all_words()
        for word in tqdm(all_words, desc="Exporting", unit="word"):
            # Access Word objects directly to get synset_id
            word_entries = wn.words.get(word, [])
            for word_entry in word_entries:
                for sense in word_entry.senses:
                    # Join multiple examples with pipe separator
                    examples_str = ' | '.join(sense.examples) if sense.examples else ''

                    writer.writerow([
                        word,
                        sense.part_of_speech,
                        sense.synset_id,
                        sense.definition,
                        examples_str
                    ])
                    row_count += 1

    print(f"\nExport complete!")
    print(f"  Total rows: {row_count:,}")
    print(f"  Output file: {output_path}")

    # Show file size
    file_size = os.path.getsize(output_path)
    if file_size > 1024 * 1024:
        print(f"  File size: {file_size / (1024 * 1024):.1f} MB")
    else:
        print(f"  File size: {file_size / 1024:.1f} KB")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Export WordNet to TSV file for embedding"
    )
    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output TSV file path (default: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "-x", "--xml",
        default=WORDNET_XML_PATH,
        help=f"WordNet XML file path (default: {WORDNET_XML_PATH})"
    )

    args = parser.parse_args()

    export_wordnet_to_tsv(args.output, args.xml)


if __name__ == "__main__":
    main()
