# src/retrieve_client.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import requests


@dataclass
class RetrievedPassage:
    text: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PathwayRetrieverClient:
    """
    Small client for Pathway DocumentStoreServer.

    Pathway server endpoints can differ by version, so we try multiple common endpoints:
      - /v1/retrieve
      - /retrieve
      - /v1/query
      - /query
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 30.0,
        endpoints: Optional[Sequence[str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.endpoints = list(endpoints) if endpoints else [
            "/v1/retrieve",
            "/retrieve",
            "/v1/query",
            "/query",
        ]

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        book_slug: Optional[str] = None,
        strict_book_filter: bool = False,
    ) -> List[RetrievedPassage]:
        """
        Retrieve top-k passages for a query.

        - filters: sent to server if supported (some Pathway builds accept it)
        - book_slug: client-side filter on metadata/path/source (recommended for accuracy)
        - strict_book_filter: if True and no passages match book_slug, returns []
                             if False, falls back to unfiltered results
        """
        payload: Dict[str, Any] = {"query": query, "k": k}
        if filters:
            payload["filters"] = filters

        last_err: Optional[str] = None

        for ep in self.endpoints:
            url = f"{self.base_url}{ep}"

            # Try POST first
            try:
                r = requests.post(url, json=payload, timeout=self.timeout_s)
            except Exception as e:
                last_err = f"POST {url} failed: {e}"
                continue

            # Endpoint doesn't exist
            if r.status_code == 404:
                last_err = f"POST {url} -> 404"
                continue

            # Some servers only allow GET
            if r.status_code == 405:
                try:
                    r = requests.get(url, params=payload, timeout=self.timeout_s)
                except Exception as e:
                    last_err = f"GET {url} failed: {e}"
                    continue

            if r.status_code >= 400:
                last_err = f"{r.request.method} {url} -> {r.status_code}: {r.text[:500]}"
                continue

            try:
                data = r.json()
            except Exception as e:
                raise RuntimeError(
                    f"{r.request.method} {url} returned non-JSON: {e}\nRaw: {r.text[:500]}"
                )

            passages = self._parse_response(data)

            # Optional: book-scoped filtering (huge accuracy boost)
            if book_slug:
                filtered = self._filter_by_book_slug(passages, book_slug)
                if filtered:
                    return filtered[:k]
                return [] if strict_book_filter else passages[:k]

            return passages[:k]

        raise RuntimeError(
            "Could not retrieve from Pathway server. Tried endpoints: "
            f"{self.endpoints}. Last error: {last_err}"
        )

    def _filter_by_book_slug(self, passages: List[RetrievedPassage], book_slug: str) -> List[RetrievedPassage]:
        """
        Client-side filtering: keep passages whose metadata/path/source contains book_slug.
        """
        bs = book_slug.lower().strip()
        if not bs:
            return passages

        out: List[RetrievedPassage] = []
        for p in passages:
            meta = p.metadata or {}
            # Common metadata keys we might see
            path = str(meta.get("path") or meta.get("file_path") or meta.get("source") or meta.get("uri") or "")
            if bs in path.lower():
                out.append(p)
        return out

    def _parse_response(self, data: Any) -> List[RetrievedPassage]:
        """
        Supports several response shapes (Pathway versions vary).
        Attempts to locate a list of items containing:
          - text/content/chunk
          - score/similarity (optional)
          - metadata (optional)
        """
        items = None

        # Case 1: data is already a list
        if isinstance(data, list):
            items = data

        # Case 2: dict with one of common list keys
        elif isinstance(data, dict):
            for key in ("results", "documents", "docs", "matches", "passages", "data", "items"):
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break

            # Some servers return {"result": {...}} or nested
            if items is None and "result" in data and isinstance(data["result"], list):
                items = data["result"]

        if not items:
            return []

        out: List[RetrievedPassage] = []
        for it in items:
            # Sometimes items can be strings
            if isinstance(it, str):
                txt = it.strip()
                if txt:
                    out.append(RetrievedPassage(text=txt, score=0.0, metadata={}))
                continue

            if not isinstance(it, dict):
                continue

            # Text field fallbacks
            text = (
                it.get("text")
                or it.get("content")
                or it.get("chunk")
                or it.get("document")
                or it.get("page_content")
                or ""
            )
            text = str(text).strip()

            # Score fallbacks
            raw_score = it.get("score", it.get("similarity", it.get("distance", 0.0)))
            try:
                score = float(raw_score) if raw_score is not None else 0.0
            except Exception:
                score = 0.0

            # Metadata fallbacks
            meta = it.get("metadata")
            if isinstance(meta, dict):
                metadata = meta
            else:
                metadata = {}

            # Sometimes metadata is flattened at top-level
            if not metadata:
                metadata = {
                    k: v
                    for k, v in it.items()
                    if k not in ("text", "content", "chunk", "document", "page_content", "score", "similarity", "distance", "metadata")
                }

            if text:
                out.append(RetrievedPassage(text=text, score=score, metadata=metadata))

        return out
