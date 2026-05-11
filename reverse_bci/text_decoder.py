"""
Text Decoder: Recovered Text Features -> Words.

Converts the text features recovered by the ReverseDecoder into actual
words that the user is thinking. Two complementary approaches:

1. Nearest-neighbor retrieval: Find the closest word/phrase in a pre-built
   vocabulary embedding space. Fast, interpretable, constrained.
2. Neural decoder: Learned mapping from text features to vocabulary logits.
   More flexible, can handle novel combinations.

For a BCI system, we need both:
- Retrieval gives high-confidence single words
- Neural decoder handles phrases and novel combinations
- Language model re-ranking fixes errors using context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# Core BCI vocabulary: words most useful for someone who is locked-in
BCI_VOCABULARY = {
    "essential": [
        "yes", "no", "help", "pain", "water", "food", "tired", "cold", "hot",
        "stop", "go", "more", "less", "please", "thank you", "sorry",
        "hello", "goodbye", "love", "family",
    ],
    "needs": [
        "hungry", "thirsty", "bathroom", "medicine", "doctor", "nurse",
        "sleep", "rest", "sit", "stand", "move", "turn", "pillow", "blanket",
        "light", "dark", "quiet", "music", "tv", "phone",
    ],
    "emotions": [
        "happy", "sad", "angry", "scared", "frustrated", "confused",
        "comfortable", "uncomfortable", "bored", "excited", "grateful",
        "lonely", "hopeful", "worried", "calm", "anxious",
    ],
    "communication": [
        "i want", "i need", "i feel", "i think", "i know", "i don't know",
        "tell me", "show me", "wait", "repeat", "understand", "don't understand",
        "correct", "wrong", "maybe", "later", "now", "important",
    ],
    "people": [
        "mom", "dad", "wife", "husband", "son", "daughter", "brother", "sister",
        "friend", "doctor", "nurse", "everyone", "someone", "nobody",
    ],
    "body": [
        "head", "eyes", "mouth", "hand", "arm", "leg", "back", "chest",
        "stomach", "left", "right", "up", "down",
    ],
    "time": [
        "today", "tomorrow", "yesterday", "morning", "afternoon", "evening",
        "night", "soon", "later", "always", "never", "sometimes",
    ],
    "questions": [
        "what", "where", "when", "who", "why", "how", "which",
        "how much", "how long", "how many",
    ],
    "letters": list("abcdefghijklmnopqrstuvwxyz"),
    "numbers": [str(i) for i in range(10)],
}


@dataclass
class TextDecoderConfig:
    text_feature_dim: int = 4096 * 6
    hidden_dim: int = 2048
    vocab_size: int = 0  # Auto-computed from vocabulary
    n_decoder_layers: int = 3
    dropout: float = 0.1
    use_retrieval: bool = True
    use_neural: bool = True
    use_language_model: bool = True
    beam_size: int = 5
    confidence_threshold: float = 0.3
    max_sequence_length: int = 10
    vocabulary_categories: list = field(
        default_factory=lambda: [
            "essential", "needs", "emotions", "communication",
            "people", "body", "time", "questions",
        ]
    )


class VocabularyEmbedding(nn.Module):
    """Manages the vocabulary and its embedding space.

    Each word/phrase in the BCI vocabulary gets an embedding vector
    that lives in the same space as the recovered text features.
    During retrieval, we find the nearest vocabulary embedding.
    """

    def __init__(self, vocab: list[str], feature_dim: int):
        super().__init__()
        self.vocab = vocab
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(vocab)}

        # Learnable embeddings (initialized from text encoder during training)
        self.embeddings = nn.Embedding(len(vocab), feature_dim)

        # Optional: projection for better retrieval
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.LayerNorm(feature_dim // 4),
        )

    def forward(self) -> torch.Tensor:
        """Get all vocabulary embeddings."""
        indices = torch.arange(len(self.vocab), device=self.embeddings.weight.device)
        return self.embeddings(indices)

    def retrieve(
        self, query: torch.Tensor, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Find nearest vocabulary words to a query feature vector.

        Args:
            query: (feature_dim,) or (B, feature_dim) query vectors
            top_k: number of results to return

        Returns:
            List of (word, similarity_score) tuples
        """
        if query.ndim == 1:
            query = query.unsqueeze(0)

        # Project both to retrieval space
        q = self.proj(query)
        v = self.proj(self.forward())

        # Normalize for cosine similarity
        q = F.normalize(q, dim=-1)
        v = F.normalize(v, dim=-1)

        # Compute similarities
        sims = torch.matmul(q, v.T)  # (B, vocab_size)

        results = []
        for b in range(sims.shape[0]):
            scores, indices = sims[b].topk(top_k)
            batch_results = [
                (self.idx_to_word[idx.item()], score.item())
                for idx, score in zip(indices, scores)
            ]
            results.append(batch_results)

        return results if len(results) > 1 else results[0]


class NeuralTextDecoder(nn.Module):
    """Neural network that maps text features to vocabulary logits."""

    def __init__(self, feature_dim: int, vocab_size: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()

        layers = []
        in_dim = feature_dim
        for i in range(n_layers - 1):
            out_dim = hidden_dim if i < n_layers - 2 else hidden_dim // 2
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, vocab_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LanguageModelReranker(nn.Module):
    """Simple n-gram language model for re-ranking decoded sequences.

    In a BCI context, we care about:
    1. Common phrases ("I want water", not "water want I")
    2. Contextual coherence (if they said "I feel" -> emotion word likely next)
    3. Reducing errors by using linguistic constraints
    """

    def __init__(self, vocab: list[str], context_size: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.vocab_size = len(vocab)
        self.context_size = context_size

        self.embed = nn.Embedding(len(vocab), hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, len(vocab))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Predict next token probabilities given context.

        Args:
            token_ids: (B, T) sequence of vocabulary indices

        Returns:
            (B, T, vocab_size) next-token logits
        """
        h = self.embed(token_ids)
        h, _ = self.rnn(h)
        return self.head(h)

    def score_sequence(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Score a complete sequence by its language model probability."""
        logits = self.forward(token_ids[:, :-1])
        targets = token_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum(dim=-1)


class TextDecoder(nn.Module):
    """Full text decoding pipeline: features -> words.

    Combines retrieval, neural decoding, and language model re-ranking
    to produce the most likely word or phrase the user is thinking.
    """

    def __init__(self, config: Optional[TextDecoderConfig] = None):
        super().__init__()
        if config is None:
            config = TextDecoderConfig()
        self.config = config

        # Build vocabulary from selected categories
        self.vocab = []
        for cat in config.vocabulary_categories:
            if cat in BCI_VOCABULARY:
                self.vocab.extend(BCI_VOCABULARY[cat])
        # Add letters and numbers
        self.vocab.extend(BCI_VOCABULARY.get("letters", []))
        self.vocab.extend(BCI_VOCABULARY.get("numbers", []))
        # Remove duplicates preserving order
        seen = set()
        unique_vocab = []
        for w in self.vocab:
            if w not in seen:
                seen.add(w)
                unique_vocab.append(w)
        self.vocab = unique_vocab

        config.vocab_size = len(self.vocab)
        logger.info("BCI vocabulary: %d words/phrases", config.vocab_size)

        # Retrieval-based decoder
        if config.use_retrieval:
            self.vocab_embed = VocabularyEmbedding(
                self.vocab, config.text_feature_dim
            )

        # Neural decoder
        if config.use_neural:
            self.neural_decoder = NeuralTextDecoder(
                feature_dim=config.text_feature_dim,
                vocab_size=config.vocab_size,
                hidden_dim=config.hidden_dim,
                n_layers=config.n_decoder_layers,
                dropout=config.dropout,
            )

        # Language model re-ranker
        if config.use_language_model:
            self.lm = LanguageModelReranker(self.vocab)

        # Confidence estimator
        self.confidence = nn.Sequential(
            nn.Linear(config.text_feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, text_features: torch.Tensor) -> dict:
        """Decode text features into words.

        Args:
            text_features: (B, text_feature_dim) recovered text features

        Returns:
            Dictionary with decoded words, confidence scores, and alternatives
        """
        B = text_features.shape[0]
        outputs = {
            "decoded_words": [],
            "confidence": [],
            "alternatives": [],
            "neural_logits": None,
        }

        # Confidence estimation
        conf = self.confidence(text_features).squeeze(-1)  # (B,)
        outputs["confidence"] = conf

        # Neural decoding
        if hasattr(self, "neural_decoder"):
            logits = self.neural_decoder(text_features)  # (B, vocab_size)
            outputs["neural_logits"] = logits
            probs = F.softmax(logits, dim=-1)

            for b in range(B):
                top_probs, top_idx = probs[b].topk(self.config.beam_size)
                words = [self.vocab[i] for i in top_idx.tolist()]
                scores = top_probs.tolist()
                outputs["decoded_words"].append(words[0])
                outputs["alternatives"].append(list(zip(words, scores)))

        # Retrieval-based decoding (for cross-validation)
        if hasattr(self, "vocab_embed"):
            retrieval_results = self.vocab_embed.retrieve(
                text_features, top_k=self.config.beam_size
            )
            if isinstance(retrieval_results[0], tuple):
                retrieval_results = [retrieval_results]
            outputs["retrieval_results"] = retrieval_results

            # If no neural decoder, use retrieval
            if not hasattr(self, "neural_decoder"):
                for b in range(B):
                    outputs["decoded_words"].append(retrieval_results[b][0][0])
                    outputs["alternatives"].append(retrieval_results[b])

        return outputs

    def decode_with_context(
        self,
        text_features: torch.Tensor,
        previous_words: list[str],
    ) -> dict:
        """Decode with language model context for better accuracy.

        Args:
            text_features: (B, text_feature_dim)
            previous_words: list of previously decoded words for context
        """
        base_output = self.forward(text_features)

        if not hasattr(self, "lm") or not previous_words:
            return base_output

        # Convert previous words to indices
        prev_ids = []
        for w in previous_words[-self.lm.context_size:]:
            if w in self.vocab_embed.word_to_idx:
                prev_ids.append(self.vocab_embed.word_to_idx[w])

        if not prev_ids:
            return base_output

        # Get language model predictions
        context = torch.tensor([prev_ids], device=text_features.device)
        lm_logits = self.lm(context)[:, -1, :]  # (1, vocab_size)
        lm_probs = F.softmax(lm_logits, dim=-1)

        # Combine neural decoder and language model scores
        if base_output["neural_logits"] is not None:
            neural_probs = F.softmax(base_output["neural_logits"], dim=-1)
            # Geometric mean combination
            combined = (neural_probs * lm_probs).sqrt()
            combined = combined / combined.sum(dim=-1, keepdim=True)

            for b in range(text_features.shape[0]):
                top_probs, top_idx = combined[b].topk(self.config.beam_size)
                words = [self.vocab[i] for i in top_idx.tolist()]
                scores = top_probs.tolist()
                base_output["decoded_words"][b] = words[0]
                base_output["alternatives"][b] = list(zip(words, scores))

        return base_output

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def get_word_index(self, word: str) -> Optional[int]:
        if hasattr(self, "vocab_embed"):
            return self.vocab_embed.word_to_idx.get(word)
        return None
