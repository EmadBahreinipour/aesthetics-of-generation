
# 19-feature set based on Opara (2024) and Kumarage et al. (2023).
CORE_FEATURES: list[str] = [
    # Lexical diversity
    "type_token_ratio",
    "hapax_legomena_ratio",
    "avg_word_length",
    "stop_word_ratio",
    "bigram_uniqueness_ratio",
    "trigram_uniqueness_ratio",
    # Readability
    "gunning_fog_index",
    # Sentence / phraseology
    "avg_sentence_length",
    "sentence_length_std",
    "paragraph_count",
    "avg_paragraph_length",
    # POS ratios
    "pos_noun_ratio",
    "pos_verb_ratio",
    "pos_adj_ratio",
    "pos_adv_ratio",
    # Punctuation
    "comma_ratio",
    "question_mark_ratio",
    # Higher-level style
    "avg_syntactic_depth",
    "discourse_total_ratio",
]

DISCOURSE_MARKERS = {
    "additive": ["also", "moreover", "furthermore", "additionally", "besides"],
    "contrastive": ["however", "but", "although", "nevertheless", "yet", "though"],
    "causal": ["because", "therefore", "thus", "hence", "consequently", "so"],
    "temporal": ["then", "first", "finally", "meanwhile", "subsequently", "next"],
    "elaborative": ["specifically", "particularly", "especially", "indeed", "in fact"],
}
