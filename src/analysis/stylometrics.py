"""
Stylometric feature extraction for LLM-generated texts.

Extracts 19 features across lexical diversity, readability,
sentence structure, POS ratios, punctuation, and discourse.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import spacy
import yaml

from .constants import DISCOURSE_MARKERS


@dataclass
class StylometricFeatures:
    """Container for extracted stylometric features."""

    lexical: Dict[str, float] = field(default_factory=dict)
    readability: Dict[str, float] = field(default_factory=dict)
    syntactic: Dict[str, float] = field(default_factory=dict)
    punctuation: Dict[str, float] = field(default_factory=dict)
    discourse: Dict[str, float] = field(default_factory=dict)
    raw_text: Optional[str] = None

    def to_dict(self) -> Dict[str, float]:
        output = {}
        for cat in [self.lexical, self.readability, self.syntactic,
                     self.punctuation, self.discourse]:
            output.update(cat)
        return output

class StylometricAnalyzer:
    """Extracts stylometric features from text using spaCy."""

    def __init__(self, spacy_model: Optional[str] = None):
        if spacy_model is not None:
            self._spacy_model = spacy_model
        else:
            try:
                project_root = Path(__file__).resolve().parent.parent.parent
                with open(project_root / "config.yaml") as f:
                    config = yaml.safe_load(f)
                self._spacy_model = config.get("analysis", {}).get("spacy_model")
            except FileNotFoundError:
                self._spacy_model = None
        if not self._spacy_model:
            raise ValueError("spaCy model not specified — pass spacy_model or set analysis.spacy_model in config.yaml")
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self._spacy_model)
            except Exception as e:
                raise Exception(f"Error loading spaCy model {self._spacy_model}: {e}")
        return self._nlp

    def extract_features(self, text: str) -> StylometricFeatures:
        """Extract all 19 stylometric features from a single text.

        Parses the spaCy doc once and reuses it across all extraction methods.
        """
        doc = self.nlp(text)
        tokens = [t.text.lower() for t in doc if not t.is_punct and not t.is_space]
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

        features = StylometricFeatures(raw_text=text)
        features.lexical = self._lexical_features(doc, tokens)
        features.readability = self._readability_features(tokens, sentences)
        features.syntactic = self._syntactic_features(doc, tokens, sentences)
        features.punctuation = self._punctuation(text, sentences)
        features.discourse = self._discourse(text, tokens)

        return features
    
    def _lexical_features(self, doc, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            raise ValueError("Cannot compute lexical features: text produced no tokens")

        freq = Counter(tokens)
        n = len(tokens)
        hapax = sum(1 for c in freq.values() if c == 1)

        stop_count = sum(
            1 for t in doc
            if t.is_stop and not t.is_punct and not t.is_space
        )

        bigrams = list(zip(tokens[:-1], tokens[1:])) if n >= 2 else []
        trigrams = list(zip(tokens[:-2], tokens[1:-1], tokens[2:])) if n >= 3 else []

        return {
            "type_token_ratio": len(freq) / n,
            "hapax_legomena_ratio": hapax / n,
            "avg_word_length": np.mean([len(w) for w in tokens]),
            "stop_word_ratio": stop_count / n,
            "bigram_uniqueness_ratio": len(set(bigrams)) / len(bigrams) if bigrams else 0.0,
            "trigram_uniqueness_ratio": len(set(trigrams)) / len(trigrams) if trigrams else 0.0,
        }

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Vowel-group heuristic, strip trailing silent 'e'."""
        word = word.lower().rstrip("e")
        if not word:
            return 1
        return max(len(re.findall(r'[aeiouy]+', word)), 1)

    def _readability_features(self, tokens: List[str], sents: List[str]) -> Dict[str, float]:
        n_words = len(tokens)
        n_sents = max(len(sents), 1)

        if n_words == 0:
            raise ValueError("Cannot compute readability features: text produced no tokens")

        complex_words = sum(1 for w in tokens if self._count_syllables(w) >= 3)
        fog = 0.4 * ((n_words / n_sents) + 100.0 * (complex_words / n_words))

        return {"gunning_fog_index": fog}

    def _syntactic_features(self, doc, tokens: List[str], sents: List[str]) -> Dict[str, float]:
        if not sents:
            raise ValueError("Cannot compute syntactic features: text produced no sentences")
        else:
            sent_lens = [
                len([t for t in s if not t.is_punct and not t.is_space])
                for s in doc.sents
            ]
            feats = {
                "avg_sentence_length": np.mean(sent_lens),
                "sentence_length_std": np.std(sent_lens) if len(sent_lens) > 1 else 0.0,
            }

        # POS ratios
        pos_counts = Counter(t.pos_ for t in doc)
        total = max(len(doc), 1)
        for tag in ["NOUN", "VERB", "ADJ", "ADV"]:
            feats[f"pos_{tag.lower()}_ratio"] = pos_counts.get(tag, 0) / total

        # avg dependency-tree depth
        depths = []
        for sent in doc.sents:
            for tok in sent:
                d = 0
                cur = tok
                while cur.head != cur:
                    d += 1
                    cur = cur.head
                depths.append(d)
        feats["avg_syntactic_depth"] = np.mean(depths) if depths else 0.0

        return feats

    def _punctuation(self, text: str, sents: List[str]) -> Dict[str, float]:
        """Per-sentence punctuation counts."""
        n = max(len(sents), 1)
        return {
            "comma_ratio": text.count(",") / n,
            "question_mark_ratio": text.count("?") / n,
        }
    
    def _discourse(self, text: str, tokens: List[str]) -> Dict[str, float]:
        n = max(len(tokens), 1)

        total_markers = 0
        for markers in DISCOURSE_MARKERS.values():
            for marker in markers:
                words = marker.split()
                if len(words) == 1:
                    total_markers += tokens.count(marker)
                else:
                    for i in range(len(tokens) - len(words) + 1):
                        if tokens[i:i + len(words)] == words:
                            total_markers += 1

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        para_count = len(paragraphs)

        if paragraphs:
            avg_para_len = np.mean([len(re.findall(r'\b\w+\b', p)) for p in paragraphs])
        else:
            avg_para_len = float(n)

        return {
            "discourse_total_ratio": total_markers / n,
            "paragraph_count": para_count,
            "avg_paragraph_length": avg_para_len,
        }

    def analyze_corpus(
        self,
        texts: List[str],
        labels:List[str],
        genres: List[str],
        prompt_ids: List[str],
        runs: List[int],
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with 19 feature columns plus any metadata
        columns (label, genre, prompt_id, run) when provided.
        """
        rows = [self.extract_features(t).to_dict() for t in texts]
        df = pd.DataFrame(rows)
        
        df["label"] = labels
        df["genre"] = genres
        df["prompt_id"] = prompt_ids
        df["run"] = runs

        return df
