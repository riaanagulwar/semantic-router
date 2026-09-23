"""
Semantic router: routes free-text queries to handlers using local embedding
similarity (sentence-transformers, runs on CPU, no API cost). Only queries
that don't confidently match a known route fall through to the LLM.

Guardrails implemented here:
  1. Input validation      - reject empty / oversized / control-char input
                              before it reaches embedding or any handler.
  2. Similarity threshold   - a weak best-match is not trusted; falls back
                              instead of routing on a low-confidence guess.
  3. Ambiguity margin       - if the top two routes are nearly tied, treat
                              it as ambiguous rather than picking one.
  4. Fixed route set        - classify() can only ever return a route name
                              that was loaded from config; it never invents
                              one or executes anything itself.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from app.config.settings import settings

logger = logging.getLogger("router")

# Defaults below match the project's original hardcoded values; all three
# are now overridable via config/settings.py (env vars) without a code
# edit. The algorithm itself is unchanged.
MAX_QUERY_LEN = settings.max_query_len
SIMILARITY_THRESHOLD = settings.similarity_threshold  # best match below this -> not confident, fallback
AMBIGUITY_MARGIN = settings.ambiguity_margin           # top two routes closer than this -> ambiguous, fallback


@dataclass
class RouteMatch:
    route: str
    score: float
    runner_up: Optional[str]
    runner_up_score: float
    confident: bool
    reason: str  # "ok" | "below_threshold" | "ambiguous"


class InputRejected(Exception):
    pass


def validate_input(text: str) -> str:
    """Guardrail #1: reject bad input before it reaches embedding or a
    handler. Called by core.handle_query() before classify()."""
    if not text or not text.strip():
        raise InputRejected("empty query")
    if len(text) > MAX_QUERY_LEN:
        raise InputRejected(f"query too long ({len(text)} chars, max {MAX_QUERY_LEN})")
    # strip control characters, keep normal punctuation/unicode text
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return cleaned.strip()


class SemanticRouter:
    def __init__(self, config_path: str | None = None, model_name: str | None = None):
        config_path = config_path or settings.routes_config_path
        model_name = model_name or settings.embedding_model_name
        self.model = SentenceTransformer(model_name)
        self.routes: dict[str, list[str]] = {}
        self.handler_names: dict[str, str] = {}
        self._route_embeddings: dict[str, np.ndarray] = {}
        self._load_config(config_path)

    def _load_config(self, path: str) -> None:
        with open(path) as f:
            cfg = yaml.safe_load(f)

        for route in cfg["routes"]:
            name = route["name"]
            utterances = route["utterances"]
            self.routes[name] = utterances
            self.handler_names[name] = route["handler"]
            self._route_embeddings[name] = self.model.encode(
                utterances, normalize_embeddings=True
            )
        logger.info("loaded %d routes: %s", len(self.routes), list(self.routes))

    def classify(self, query: str) -> RouteMatch:
        """Returns the best-matching route plus enough context (runner-up,
        margin) for the caller to decide whether to trust it. Never
        dispatches anything itself."""
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]

        scores: dict[str, float] = {}
        for name, embeddings in self._route_embeddings.items():
            sims = embeddings @ query_vec        # cosine sim (vectors are normalized)
            scores[name] = float(np.max(sims))   # best-matching utterance per route

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_name, best_score = ranked[0]
        runner_up_name, runner_up_score = ranked[1] if len(ranked) > 1 else (None, 0.0)

        if best_score < SIMILARITY_THRESHOLD:
            return RouteMatch(best_name, best_score, runner_up_name,
                               runner_up_score, confident=False, reason="below_threshold")

        if best_score - runner_up_score < AMBIGUITY_MARGIN:
            return RouteMatch(best_name, best_score, runner_up_name,
                               runner_up_score, confident=False, reason="ambiguous")

        return RouteMatch(best_name, best_score, runner_up_name,
                           runner_up_score, confident=True, reason="ok")
