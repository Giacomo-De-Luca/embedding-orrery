"""
English WordNet XML Parser

This module parses the english-wordnet-2024.xml file into a structured format
that allows easy access to words and their definitions.

Available Relation Types in WordNet XML:
=========================================
| Relation Type      | Count   | Method                      |
|--------------------|---------|-----------------------------|
| **Taxonomic**      |         |                             |
| hypernym           | 93,446  | get_hypernyms()             |
| hyponym            | 93,446  | get_hyponyms()              |
| instance_hypernym  | 8,614   | get_instance_hypernyms()    |
| instance_hyponym   | 8,614   | get_instance_hyponyms()     |
| **Meronyms (parts)**|        |                             |
| mero_member        | 12,296  | get_member_meronyms()       |
| mero_part          | 9,202   | get_part_meronyms()         |
| mero_substance     | 830     | get_substance_meronyms()    |
| **Holonyms (wholes)**|       |                             |
| holo_member        | 12,296  | get_member_holonyms()       |
| holo_part          | 9,202   | get_part_holonyms()         |
| holo_substance     | 830     | get_substance_holonyms()    |
| **Other semantic** |         |                             |
| similar            | 23,190  | get_similar()               |
| antonym            | 7,996   | get_antonyms()              |
| derivation         | 74,646  | get_derivations()           |
| also               | 3,876   | get_also_sees()             |
| attribute          | 1,278   | get_attributes()            |
| entails            | 407     | get_entailments()           |
| causes             | 221     | get_causes()                |
| pertainym          | 8,072   | get_pertainyms()            |
| other              | 16,887  | get_related_synsets()       |
| **Domain**         |         |                             |
| domain_topic       | 6,946   | get_domain_topics()         |
| has_domain_topic   | 6,946   | get_synsets_in_domain()     |
| domain_region      | 1,349   | get_domain_regions()        |
| has_domain_region  | 1,349   | get_synsets_in_region()     |
| exemplifies        | 8,264   | get_exemplifies()           |
| is_exemplified_by  | 8,264   | get_examples_of()           |
| participle         | 73      | get_related_synsets()       |
"""

import gzip
import os
import pickle
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from tqdm import tqdm


@dataclass
class SynsetRelation:
    """Represents a relationship between synsets."""
    relation_type: str
    target_synset_id: str


@dataclass
class Synset:
    """Represents a synset (synonym set) with its definition and examples."""
    id: str
    definition: str
    part_of_speech: str
    examples: list[str] = field(default_factory=list)
    members: list[str] = field(default_factory=list)
    relations: list[SynsetRelation] = field(default_factory=list)


@dataclass
class Sense:
    """Represents a sense (meaning) of a word."""
    synset_id: str
    definition: str
    part_of_speech: str
    examples: list[str] = field(default_factory=list)


@dataclass
class Word:
    """Represents a word with all its senses/meanings."""
    word: str
    part_of_speech: str
    senses: list[Sense] = field(default_factory=list)


class WordNetParser:
    """Parser for English WordNet XML files."""

    def __init__(self, xml_file_path: str = "resources/english-wordnet-2024.xml", cache_dir: str | None = None):
        """
        Initialize the parser with the path to the WordNet XML file.

        Args:
            xml_file_path: Path to the english-wordnet-2024.xml file.
                          Relative paths are resolved from this script's directory.
            cache_dir: Directory to store the pickle cache. If None, uses the same directory as the XML file.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        resolved_xml_path = (
            xml_file_path
            if os.path.isabs(xml_file_path)
            else os.path.join(script_dir, xml_file_path)
        )
        resolved_xml_path = os.path.abspath(resolved_xml_path)

        if not os.path.exists(resolved_xml_path):
            os.makedirs(os.path.dirname(resolved_xml_path), exist_ok=True)
            url = "https://en-word.net/static/english-wordnet-2024.xml.gz"
            print(f"Downloading WordNet XML to: {resolved_xml_path}")
            self._download_xml(url, resolved_xml_path)
            print("Downloading wordnet XML complete.")

        self.xml_file_path = resolved_xml_path
        self.synsets: dict[str, Synset] = {}
        self.words: dict[str, list[Word]] = {}  # word -> list of Word objects (different POS)
        self.synset_to_words: dict[str, list[str]] = {}  # synset_id -> list of words
        self._parsed = False

        # Set up pickle cache path
        if cache_dir is None:
            cache_dir = os.path.dirname(self.xml_file_path) or "."
        elif not os.path.isabs(cache_dir):
            cache_dir = os.path.join(script_dir, cache_dir)
        cache_dir = os.path.abspath(cache_dir)

        xml_basename = os.path.basename(self.xml_file_path)
        pickle_name = os.path.splitext(xml_basename)[0] + ".pkl"
        self._pickle_path = os.path.join(cache_dir, pickle_name)

        # Try to load from pickle cache
        self._load_from_pickle()

    def _download_xml(self, url: str, output_path: str, retries: int = 4):
        """Download XML with retries and tqdm progress."""
        last_error: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36",
                        "Accept": "application/xml,text/xml,*/*",
                    },
                )
                with urllib.request.urlopen(request, timeout=60) as response, open(output_path, 'wb') as output_file:
                    total_size = int(response.headers.get('Content-Length', 0))
                    chunk_size = 1024 * 1024
                    is_gzip = url.endswith('.gz')

                    with tqdm(
                        total=total_size if total_size > 0 else None,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        desc='english-wordnet-2024.xml',
                    ) as progress_bar:
                        if is_gzip:
                            with gzip.GzipFile(fileobj=response) as gz:
                                while True:
                                    chunk = gz.read(chunk_size)
                                    if not chunk:
                                        break
                                    output_file.write(chunk)
                                    progress_bar.update(len(chunk))
                        else:
                            while True:
                                chunk = response.read(chunk_size)
                                if not chunk:
                                    break
                                output_file.write(chunk)
                                progress_bar.update(len(chunk))
                return
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
                last_error = error
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass

                if attempt < retries:
                    wait_seconds = 2 ** (attempt - 1)
                    print(f"Download failed (attempt {attempt}/{retries}): {error}. Retrying in {wait_seconds}s...")
                    time.sleep(wait_seconds)
                else:
                    break

        raise RuntimeError(f"Failed to download WordNet XML after {retries} attempts: {last_error}")

    def _load_from_pickle(self) -> bool:
        """
        Try to load parsed data from pickle cache.
        
        Returns:
            True if successfully loaded from cache, False otherwise.
        """
        if os.path.exists(self._pickle_path):
            # Check if pickle is newer than XML file
            xml_mtime = os.path.getmtime(self.xml_file_path) if os.path.exists(self.xml_file_path) else 0
            pickle_mtime = os.path.getmtime(self._pickle_path)

            if pickle_mtime >= xml_mtime:
                try:
                    print(f"Loading WordNet from cache: {self._pickle_path}")
                    with open(self._pickle_path, 'rb') as f:
                        data = pickle.load(f)
                    self.synsets = data['synsets']
                    self.words = data['words']
                    self.synset_to_words = data['synset_to_words']
                    self._parsed = True
                    print(f"Loaded {len(self.words):,} words and {len(self.synsets):,} synsets from cache.")
                    return True
                except Exception as e:
                    print(f"Failed to load pickle cache: {e}")
                    return False
            else:
                print("XML file is newer than cache, will re-parse.")
        return False

    def _save_to_pickle(self):
        """Save parsed data to pickle cache."""
        try:
            print(f"Saving WordNet to cache: {self._pickle_path}")
            data = {
                'synsets': self.synsets,
                'words': self.words,
                'synset_to_words': self.synset_to_words
            }
            with open(self._pickle_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Cache saved successfully.")
        except Exception as e:
            print(f"Failed to save pickle cache: {e}")

    def parse(self):
        """Parse the XML file and build the internal data structures."""
        if self._parsed:
            return

        print("Parsing WordNet XML file...")
        print("Step 1: Parsing synsets...")

        # First pass: collect all synsets
        context = ET.iterparse(self.xml_file_path, events=('start', 'end'))
        context = iter(context)

        synset_count = 0
        for event, elem in context:
            if event == 'end' and elem.tag == 'Synset':
                synset_id = elem.get('id')
                pos = elem.get('partOfSpeech', '')

                # Get definition
                definition_elem = elem.find('Definition')
                definition = definition_elem.text if definition_elem is not None else ''

                # Get examples
                examples = [ex.text for ex in elem.findall('Example') if ex.text]

                # Get members
                members_attr = elem.get('members', '')
                members = members_attr.split() if members_attr else []

                # Get synset relations
                relations = []
                for rel_elem in elem.findall('SynsetRelation'):
                    rel_type = rel_elem.get('relType', '')
                    target = rel_elem.get('target', '')
                    if rel_type and target:
                        relations.append(SynsetRelation(
                            relation_type=rel_type,
                            target_synset_id=target
                        ))
                ## TODO check types, at the moment is on ignore as it's working and it's not critica
                self.synsets[synset_id] = Synset( # type: ignore
                    id=synset_id, # type: ignore
                    definition=definition, # type: ignore
                    part_of_speech=pos,
                    examples=examples,
                    members=members,
                    relations=relations
                )

                synset_count += 1
                if synset_count % 10000 == 0:
                    print(f"  Parsed {synset_count} synsets...")

                # Clear element to free memory
                elem.clear()

        print(f"  Total synsets parsed: {synset_count}")
        print("\nStep 2: Parsing lexical entries...")

        # Second pass: collect all lexical entries
        context = ET.iterparse(self.xml_file_path, events=('start', 'end'))
        context = iter(context)

        entry_count = 0
        for event, elem in context:
            if event == 'end' and elem.tag == 'LexicalEntry':
                # Get the lemma (word form)
                lemma_elem = elem.find('Lemma')
                if lemma_elem is not None:
                    word_form = lemma_elem.get('writtenForm', '')
                    pos = lemma_elem.get('partOfSpeech', '')

                    # Get all senses
                    senses = []
                    for sense_elem in elem.findall('Sense'):
                        synset_id = sense_elem.get('synset')
                        if synset_id and synset_id in self.synsets:
                            synset = self.synsets[synset_id]
                            senses.append(Sense(
                                synset_id=synset_id,
                                definition=synset.definition,
                                part_of_speech=synset.part_of_speech,
                                examples=synset.examples
                            ))

                            # Map synset to word for synonym lookups
                            if synset_id not in self.synset_to_words:
                                self.synset_to_words[synset_id] = []
                            if word_form not in self.synset_to_words[synset_id]:
                                self.synset_to_words[synset_id].append(word_form)

                    # Create Word object
                    word_obj = Word(
                        word=word_form,
                        part_of_speech=pos,
                        senses=senses
                    )

                    # Store in dictionary (a word can have multiple entries for different POS)
                    if word_form not in self.words:
                        self.words[word_form] = []
                    self.words[word_form].append(word_obj)

                    entry_count += 1
                    if entry_count % 10000 == 0:
                        print(f"  Parsed {entry_count} lexical entries...")

                # Clear element to free memory
                elem.clear()

        print(f"  Total lexical entries parsed: {entry_count}")
        print(f"\nParsing complete! Total unique words: {len(self.words)}")
        self._parsed = True

        # Save to pickle cache
        self._save_to_pickle()

    def get_all_words(self) -> list[str]:
        """
        Get a list of all words in the WordNet.

        Returns:
            Sorted list of all unique words
        """
        if not self._parsed:
            self.parse()
        return sorted(self.words.keys())

    def get_word(self, word: str) -> list[Word] | None:
        """
        Get all entries for a specific word.

        Args:
            word: The word to look up

        Returns:
            List of Word objects (one for each part of speech), or None if not found
        """
        if not self._parsed:
            self.parse()
        return self.words.get(word)

    def get_definitions(self, word: str) -> list[dict[str, any]]: #type: ignore
        """
        Get all definitions for a word in a simple format.

        Args:
            word: The word to look up

        Returns:
            List of dictionaries with 'pos', 'definition', and 'examples' keys
        """
        if not self._parsed:
            self.parse()

        word_entries = self.words.get(word)
        if not word_entries:
            return []

        definitions = []
        for word_entry in word_entries:
            for i, sense in enumerate(word_entry.senses, 1):
                definitions.append({
                    'sense_number': i,
                    'part_of_speech': sense.part_of_speech,
                    'definition': sense.definition,
                    'examples': sense.examples
                })

        return definitions

    def search_words(self, prefix: str) -> list[str]:
        """
        Search for words starting with a given prefix.

        Args:
            prefix: The prefix to search for

        Returns:
            List of words starting with the prefix
        """
        if not self._parsed:
            self.parse()

        return [word for word in sorted(self.words.keys()) if word.startswith(prefix)]

    def get_stats(self) -> dict[str, int]:
        """
        Get statistics about the WordNet data.

        Returns:
            Dictionary with statistics
        """
        if not self._parsed:
            self.parse()

        total_senses = sum(len(word.senses) for words in self.words.values() for word in words)

        return {
            'total_words': len(self.words),
            'total_synsets': len(self.synsets),
            'total_senses': total_senses
        }

    # ========== Synset-related methods ==========

    def get_synset(self, synset_id: str) -> Synset | None:
        """
        Get a synset by its ID.

        Args:
            synset_id: The synset ID to look up

        Returns:
            Synset object or None if not found
        """
        if not self._parsed:
            self.parse()
        return self.synsets.get(synset_id)

    def get_synsets_for_word(self, word: str) -> list[Synset]:
        """
        Get all synsets that contain a given word.

        Args:
            word: The word to look up

        Returns:
            List of Synset objects
        """
        if not self._parsed:
            self.parse()

        word_entries = self.words.get(word)
        if not word_entries:
            return []

        synsets = []
        seen_synset_ids = set()

        for word_entry in word_entries:
            for sense in word_entry.senses:
                if sense.synset_id not in seen_synset_ids:
                    synset = self.synsets.get(sense.synset_id)
                    if synset:
                        synsets.append(synset)
                        seen_synset_ids.add(sense.synset_id)

        return synsets

    def get_synonyms(self, word: str, sense_number: int | None = None) -> list[str]:
        """
        Get synonyms for a word. If sense_number is provided, only get synonyms
        for that specific sense. Otherwise, get all synonyms across all senses.

        Args:
            word: The word to find synonyms for
            sense_number: Optional sense number (1-indexed). If None, return synonyms for all senses.

        Returns:
            List of synonym words
        """
        if not self._parsed:
            self.parse()

        word_entries = self.words.get(word)
        if not word_entries:
            return []

        synonyms = set()

        for word_entry in word_entries:
            for i, sense in enumerate(word_entry.senses, 1):
                # If sense_number is specified, only process that sense
                if sense_number is not None and i != sense_number:
                    continue

                # Get all words in this synset
                synset_words = self.synset_to_words.get(sense.synset_id, [])
                for syn_word in synset_words:
                    if syn_word != word:  # Don't include the word itself
                        synonyms.add(syn_word)

        return sorted(list(synonyms))

    def get_words_in_synset(self, synset_id: str) -> list[str]:
        """
        Get all words that belong to a given synset (i.e., all synonyms in the synset).

        Args:
            synset_id: The synset ID

        Returns:
            List of words in the synset
        """
        if not self._parsed:
            self.parse()
        return self.synset_to_words.get(synset_id, [])

    def get_related_synsets(self, synset_id: str, relation_type: str | None = None) -> list[Synset]:
        """
        Get synsets related to the given synset by specific relationship type(s).

        Args:
            synset_id: The synset ID to find relations for
            relation_type: Optional relation type to filter by (e.g., 'hypernym', 'hyponym').
                          If None, returns all related synsets regardless of type.

        Returns:
            List of related Synset objects
        """
        if not self._parsed:
            self.parse()

        synset = self.synsets.get(synset_id)
        if not synset:
            return []

        related = []
        for relation in synset.relations:
            if relation_type is None or relation.relation_type == relation_type:
                target_synset = self.synsets.get(relation.target_synset_id)
                if target_synset:
                    related.append(target_synset)

        return related

    def get_hypernyms(self, synset_id: str) -> list[Synset]:
        """
        Get hypernyms (more general concepts) for a synset.
        E.g., 'animal' is a hypernym of 'dog'.

        Args:
            synset_id: The synset ID

        Returns:
            List of hypernym Synset objects
        """
        return self.get_related_synsets(synset_id, 'hypernym')

    def get_hyponyms(self, synset_id: str) -> list[Synset]:
        """
        Get hyponyms (more specific concepts) for a synset.
        E.g., 'dog' is a hyponym of 'animal'.

        Args:
            synset_id: The synset ID

        Returns:
            List of hyponym Synset objects
        """
        return self.get_related_synsets(synset_id, 'hyponym')

    def get_instance_hypernyms(self, synset_id: str) -> list[Synset]:
        """
        Get instance hypernyms (classes that this instance belongs to).
        E.g., 'city' is an instance hypernym of 'Paris'.

        Args:
            synset_id: The synset ID

        Returns:
            List of instance hypernym Synset objects
        """
        return self.get_related_synsets(synset_id, 'instance_hypernym')

    def get_instance_hyponyms(self, synset_id: str) -> list[Synset]:
        """
        Get instance hyponyms (instances of this class).
        E.g., 'Paris' is an instance hyponym of 'city'.

        Args:
            synset_id: The synset ID

        Returns:
            List of instance hyponym Synset objects
        """
        return self.get_related_synsets(synset_id, 'instance_hyponym')

    # ========== Meronyms (part-of relationships) ==========

    def get_member_meronyms(self, synset_id: str) -> list[Synset]:
        """
        Get member meronyms (members of this group).
        E.g., 'tree' is a member meronym of 'forest'.

        Args:
            synset_id: The synset ID

        Returns:
            List of member meronym Synset objects
        """
        return self.get_related_synsets(synset_id, 'mero_member')

    def get_part_meronyms(self, synset_id: str) -> list[Synset]:
        """
        Get part meronyms (parts of this whole).
        E.g., 'wheel' is a part meronym of 'car'.

        Args:
            synset_id: The synset ID

        Returns:
            List of part meronym Synset objects
        """
        return self.get_related_synsets(synset_id, 'mero_part')

    def get_substance_meronyms(self, synset_id: str) -> list[Synset]:
        """
        Get substance meronyms (substances this is made of).
        E.g., 'wood' is a substance meronym of 'tree'.

        Args:
            synset_id: The synset ID

        Returns:
            List of substance meronym Synset objects
        """
        return self.get_related_synsets(synset_id, 'mero_substance')

    # ========== Holonyms (whole-of relationships) ==========

    def get_member_holonyms(self, synset_id: str) -> list[Synset]:
        """
        Get member holonyms (groups this is a member of).
        E.g., 'forest' is a member holonym of 'tree'.

        Args:
            synset_id: The synset ID

        Returns:
            List of member holonym Synset objects
        """
        return self.get_related_synsets(synset_id, 'holo_member')

    def get_part_holonyms(self, synset_id: str) -> list[Synset]:
        """
        Get part holonyms (wholes this is a part of).
        E.g., 'car' is a part holonym of 'wheel'.

        Args:
            synset_id: The synset ID

        Returns:
            List of part holonym Synset objects
        """
        return self.get_related_synsets(synset_id, 'holo_part')

    def get_substance_holonyms(self, synset_id: str) -> list[Synset]:
        """
        Get substance holonyms (things made of this substance).
        E.g., 'tree' is a substance holonym of 'wood'.

        Args:
            synset_id: The synset ID

        Returns:
            List of substance holonym Synset objects
        """
        return self.get_related_synsets(synset_id, 'holo_substance')

    # ========== Other semantic relationships ==========

    def get_antonyms(self, synset_id: str) -> list[Synset]:
        """
        Get antonyms (opposites).
        E.g., 'hot' is an antonym of 'cold'.

        Args:
            synset_id: The synset ID

        Returns:
            List of antonym Synset objects
        """
        return self.get_related_synsets(synset_id, 'antonym')

    def get_similar(self, synset_id: str) -> list[Synset]:
        """
        Get similar synsets (especially for adjectives).

        Args:
            synset_id: The synset ID

        Returns:
            List of similar Synset objects
        """
        return self.get_related_synsets(synset_id, 'similar')

    def get_also_sees(self, synset_id: str) -> list[Synset]:
        """
        Get 'also see' related synsets.

        Args:
            synset_id: The synset ID

        Returns:
            List of related Synset objects
        """
        return self.get_related_synsets(synset_id, 'also')

    def get_attributes(self, synset_id: str) -> list[Synset]:
        """
        Get attribute synsets (for adjective-noun relationships).
        E.g., 'heavy' has attribute 'weight'.

        Args:
            synset_id: The synset ID

        Returns:
            List of attribute Synset objects
        """
        return self.get_related_synsets(synset_id, 'attribute')

    def get_entailments(self, synset_id: str) -> list[Synset]:
        """
        Get entailments (for verbs - what this verb entails).
        E.g., 'snore' entails 'sleep'.

        Args:
            synset_id: The synset ID

        Returns:
            List of entailed Synset objects
        """
        return self.get_related_synsets(synset_id, 'entails')

    def get_causes(self, synset_id: str) -> list[Synset]:
        """
        Get causes (what this verb causes).
        E.g., 'kill' causes 'die'.

        Args:
            synset_id: The synset ID

        Returns:
            List of caused Synset objects
        """
        return self.get_related_synsets(synset_id, 'causes')

    def get_derivations(self, synset_id: str) -> list[Synset]:
        """
        Get derivationally related forms.
        E.g., 'runner' is derivationally related to 'run'.

        Args:
            synset_id: The synset ID

        Returns:
            List of derivationally related Synset objects
        """
        return self.get_related_synsets(synset_id, 'derivation')

    def get_pertainyms(self, synset_id: str) -> list[Synset]:
        """
        Get pertainyms (for adjectives - what noun this pertains to).
        E.g., 'musical' pertains to 'music'.

        Args:
            synset_id: The synset ID

        Returns:
            List of pertainym Synset objects
        """
        return self.get_related_synsets(synset_id, 'pertainym')

    # ========== Domain relationships ==========

    def get_domain_topics(self, synset_id: str) -> list[Synset]:
        """
        Get domain topics (what topic/field this belongs to).
        E.g., 'scalpel' has domain topic 'medicine'.

        Args:
            synset_id: The synset ID

        Returns:
            List of domain topic Synset objects
        """
        return self.get_related_synsets(synset_id, 'domain_topic')

    def get_domain_regions(self, synset_id: str) -> list[Synset]:
        """
        Get domain regions (what region this is used in).

        Args:
            synset_id: The synset ID

        Returns:
            List of domain region Synset objects
        """
        return self.get_related_synsets(synset_id, 'domain_region')

    def get_exemplifies(self, synset_id: str) -> list[Synset]:
        """
        Get what this synset exemplifies.

        Args:
            synset_id: The synset ID

        Returns:
            List of exemplified Synset objects
        """
        return self.get_related_synsets(synset_id, 'exemplifies')

    def get_examples_of(self, synset_id: str) -> list[Synset]:
        """
        Get synsets that are examples of this synset.

        Args:
            synset_id: The synset ID

        Returns:
            List of example Synset objects
        """
        return self.get_related_synsets(synset_id, 'is_exemplified_by')

    def get_synsets_in_domain(self, synset_id: str) -> list[Synset]:
        """
        Get synsets that belong to this domain topic.
        E.g., for 'medicine' domain, get all medical terms.

        Args:
            synset_id: The synset ID of the domain topic

        Returns:
            List of Synset objects in this domain
        """
        return self.get_related_synsets(synset_id, 'has_domain_topic')

    def get_synsets_in_region(self, synset_id: str) -> list[Synset]:
        """
        Get synsets that belong to this region.

        Args:
            synset_id: The synset ID of the region

        Returns:
            List of Synset objects in this region
        """
        return self.get_related_synsets(synset_id, 'has_domain_region')

    def get_all_domain_topics(self) -> list[Synset]:
        """
        Get all synsets that serve as domain topics.
        These are synsets that other synsets reference via 'domain_topic'.

        Returns:
            List of unique domain topic Synset objects
        """
        if not self._parsed:
            self.parse()

        domain_topics = set()
        for synset in self.synsets.values():
            for relation in synset.relations:
                if relation.relation_type == 'domain_topic':
                    domain_topics.add(relation.target_synset_id)

        return [self.synsets[sid] for sid in sorted(domain_topics) if sid in self.synsets]

    def get_all_domain_regions(self) -> list[Synset]:
        """
        Get all synsets that serve as domain regions.
        These are synsets that other synsets reference via 'domain_region'.

        Returns:
            List of unique domain topic Synset objects
        """
        if not self._parsed:
            self.parse()

        domain_regions = set()
        for synset in self.synsets.values():
            for relation in synset.relations:
                if relation.relation_type == 'domain_region':
                    domain_regions.add(relation.target_synset_id)
        return [self.synsets[sid] for sid in sorted(domain_regions) if sid in self.synsets]

    def search_domain_topics(self, query: str) -> list[Synset]:
        """
        Search for domain topics by name.

        Args:
            query: Search string (case-insensitive)

        Returns:
            List of matching domain topic Synset objects
        """
        domain_topics = self.get_all_domain_topics()
        query_lower = query.lower()

        matching = []
        for synset in domain_topics:
            # Check definition
            if query_lower in synset.definition.lower():
                matching.append(synset)
                continue
            # Check words in synset
            words = self.get_words_in_synset(synset.id)
            if any(query_lower in w.lower() for w in words):
                matching.append(synset)

        return matching

    def get_relation_types(self, synset_id: str) -> list[str]:
        """
        Get all relation types available for a given synset.

        Args:
            synset_id: The synset ID

        Returns:
            List of relation type strings
        """
        if not self._parsed:
            self.parse()

        synset = self.synsets.get(synset_id)
        if not synset:
            return []

        return sorted(list(set(rel.relation_type for rel in synset.relations)))

    def explore_synset_chain(self, synset_id: str, relation_type: str, max_depth: int = 3) -> list[list[Synset]]:
        """
        Explore a chain of synset relationships up to a certain depth.
        E.g., explore all hypernyms up to 3 levels.

        Args:
            synset_id: Starting synset ID
            relation_type: Type of relation to follow (e.g., 'hypernym', 'hyponym')
            max_depth: Maximum depth to explore

        Returns:
            List of paths, where each path is a list of Synset objects
        """
        if not self._parsed:
            self.parse()

        start_synset = self.synsets.get(synset_id)
        if not start_synset:
            return []

        def explore_recursive(current_synset: Synset, depth: int, path: list[Synset]) -> list[list[Synset]]:
            if depth >= max_depth:
                return [path]

            related = self.get_related_synsets(current_synset.id, relation_type)
            if not related:
                return [path]

            all_paths = []
            for rel_synset in related:
                new_path = path + [rel_synset]
                all_paths.extend(explore_recursive(rel_synset, depth + 1, new_path))

            return all_paths

        return explore_recursive(start_synset, 0, [start_synset])


if __name__ == '__main__':
    # Example usage
    parser = WordNetParser('english-wordnet-2024.xml')
    parser.parse()

    # Show statistics
    print("\nWordNet Statistics:")
    stats = parser.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:,}")

    # Example: Look up a word
    print("\n" + "="*60)
    print("Example: Looking up the word 'run'")
    print("="*60)

    definitions = parser.get_definitions('run')
    for i, defn in enumerate(definitions, 1):
        print(f"\n{i}. [{defn['part_of_speech']}] {defn['definition']}")
        if defn['examples']:
            print("   Examples:")
            for ex in defn['examples'][:2]:  # Show max 2 examples
                print(f"     - {ex}")

    # Example: Search for words
    print("\n" + "="*60)
    print("Example: Words starting with 'aard'")
    print("="*60)
    words = parser.search_words('aard')
    for word in words[:10]:  # Show first 10
        print(f"  - {word}")
