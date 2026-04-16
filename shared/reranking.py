import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool = True
    candidate_multiplier: int = 3
    similarity_weight: float = 0.7
    keyword_weight: float = 0.2
    metadata_weight: float = 0.1
    max_candidates: int = 50
    min_token_len: int = 3
    debug_logging: bool = False
    log_top_n: int = 5


def _parse_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_rerank_config_from_env() -> RerankConfig:
    return RerankConfig(
        enabled=_parse_bool(os.getenv("RERANK_ENABLED", "true"), True),
        candidate_multiplier=max(1, int(os.getenv("RERANK_CANDIDATE_MULTIPLIER", "3"))),
        similarity_weight=float(os.getenv("RERANK_SIMILARITY_WEIGHT", "0.7")),
        keyword_weight=float(os.getenv("RERANK_KEYWORD_WEIGHT", "0.2")),
        metadata_weight=float(os.getenv("RERANK_METADATA_WEIGHT", "0.1")),
        max_candidates=max(1, int(os.getenv("RERANK_MAX_CANDIDATES", "50"))),
        min_token_len=max(1, int(os.getenv("RERANK_MIN_TOKEN_LEN", "3"))),
        debug_logging=_parse_bool(os.getenv("RERANK_DEBUG_LOG", "false"), False),
        log_top_n=max(1, int(os.getenv("RERANK_LOG_TOP_N", "5"))),
    )


def candidate_pool_limit(top_k: int, config: RerankConfig) -> int:
    requested_top_k = max(1, int(top_k))
    if not config.enabled:
        return requested_top_k
    expanded = requested_top_k * max(1, config.candidate_multiplier)
    return min(max(requested_top_k, expanded), max(1, config.max_candidates))


def _tokenize_text(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


def _query_terms(query: str, min_token_len: int) -> set:
    return {token for token in _tokenize_text(query) if len(token) >= min_token_len}


def _keyword_overlap_score(query_terms: set, content: str) -> float:
    if not query_terms:
        return 0.0
    content_terms = set(_tokenize_text(content))
    overlap = query_terms.intersection(content_terms)
    return len(overlap) / len(query_terms)


def _metadata_score(query_terms: set, file_path: str, citation_url: str) -> float:
    if not query_terms:
        return 0.0
    metadata_terms = set(_tokenize_text(file_path)) | set(_tokenize_text(citation_url))
    overlap = query_terms.intersection(metadata_terms)
    return len(overlap) / len(query_terms)


def _extract_similarity(doc: Dict[str, Any]) -> float:
    if doc.get("similarity") is not None:
        return float(doc.get("similarity", 0.0))
    if doc.get("distance") is not None:
        return 1.0 - float(doc.get("distance", 0.0))
    return 0.0


def _log_docs(
    logger: Optional[logging.Logger],
    enabled: bool,
    stage: str,
    docs: List[Dict[str, Any]],
    top_n: int,
    log_prefix: str,
) -> None:
    if not logger or not enabled:
        return

    logger.info("[%s] %s top %s documents", log_prefix, stage, min(top_n, len(docs)))
    for idx, doc in enumerate(docs[:top_n], start=1):
        logger.info(
            "[%s] %s #%s path=%s similarity=%.4f rerank=%.4f keyword=%.4f metadata=%.4f",
            log_prefix,
            stage,
            idx,
            doc.get("file_path", ""),
            float(doc.get("similarity", 0.0)),
            float(doc.get("rerank_score", 0.0)),
            float(doc.get("keyword_score", 0.0)),
            float(doc.get("metadata_score", 0.0)),
        )


def rerank_documents(
    query: str,
    docs: List[Dict[str, Any]],
    config: RerankConfig,
    top_k: int,
    logger: Optional[logging.Logger] = None,
    log_prefix: str = "retrieval",
) -> List[Dict[str, Any]]:
    requested_top_k = max(1, int(top_k))
    if not docs:
        return []

    query_terms = _query_terms(query, config.min_token_len)

    normalized_docs: List[Dict[str, Any]] = []
    for doc in docs:
        normalized = dict(doc)
        normalized["similarity"] = _extract_similarity(normalized)
        normalized_docs.append(normalized)

    _log_docs(
        logger,
        config.debug_logging,
        "before_rerank",
        normalized_docs,
        config.log_top_n,
        log_prefix,
    )

    if not config.enabled:
        selected = normalized_docs[:requested_top_k]
        for doc in selected:
            doc["keyword_score"] = 0.0
            doc["metadata_score"] = 0.0
            doc["rerank_score"] = round(float(doc.get("similarity", 0.0)), 4)
        _log_docs(
            logger,
            config.debug_logging,
            "after_rerank_disabled",
            selected,
            config.log_top_n,
            log_prefix,
        )
        return selected

    reranked: List[Dict[str, Any]] = []
    for doc in normalized_docs:
        keyword_score = _keyword_overlap_score(query_terms, doc.get("content_text", ""))
        metadata_score = _metadata_score(query_terms, doc.get("file_path", ""), doc.get("citation_url", ""))
        final_score = (
            config.similarity_weight * float(doc.get("similarity", 0.0))
            + config.keyword_weight * keyword_score
            + config.metadata_weight * metadata_score
        )

        doc["keyword_score"] = round(keyword_score, 4)
        doc["metadata_score"] = round(metadata_score, 4)
        doc["rerank_score"] = round(final_score, 4)
        reranked.append(doc)

    reranked.sort(key=lambda item: float(item.get("rerank_score", 0.0)), reverse=True)
    selected = reranked[:requested_top_k]

    _log_docs(
        logger,
        config.debug_logging,
        "after_rerank",
        selected,
        config.log_top_n,
        log_prefix,
    )

    return selected
