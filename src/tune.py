# src/tune.py
from __future__ import annotations

import argparse
import itertools
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Optional

import pandas as pd
from tqdm import tqdm

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from src.retrieve_client import PathwayRetrieverClient, RetrievedPassage
from src.utils import split_into_claims, ground_claim, slugify
from src.verifier_nli import NLIVerifier


# ---------------------------
# Small text helpers
# ---------------------------

def _tok(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def bm25_rerank(query: str, passages: List[str]) -> List[int]:
    """Light hybrid rerank (on retrieved texts only)."""
    if not passages:
        return []
    try:
        from rank_bm25 import BM25Okapi
    except Exception:
        return list(range(len(passages)))

    q_tokens = _tok(query)
    docs = [_tok(p) for p in passages]
    if not docs or all(len(d) == 0 for d in docs):
        return list(range(len(passages)))

    bm25 = BM25Okapi(docs)
    scores = bm25.get_scores(q_tokens)
    return sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)


def sentence_focus(claim: str, passage: str, max_sentences: int) -> str:
    """Reduce premise length to top sentences most related to the claim."""
    if not passage:
        return passage
    sents = re.split(r"(?<=[.!?])\s+", passage)
    sents = [s.strip() for s in sents if s.strip()]
    if len(sents) <= max_sentences:
        return passage

    order = bm25_rerank(claim, sents)
    if not order:
        return " ".join(sents[:max_sentences])

    top = [sents[i] for i in order[:max_sentences]]
    return " ".join(top)


# ---------------------------
# Retrieval helpers
# ---------------------------

def _passage_in_book(p: RetrievedPassage, book_slug: str) -> bool:
    """Filter to correct book if metadata contains a usable path; otherwise keep."""
    if not book_slug:
        return True
    meta = p.metadata or {}
    path = str(meta.get("path", meta.get("file_path", meta.get("source", "")))).lower()
    if not path:
        return True
    return book_slug in path


def merge_and_dedupe(results_lists: Sequence[Sequence[RetrievedPassage]]) -> List[RetrievedPassage]:
    """Merge lists; dedupe by normalized text; keep best score."""
    best: Dict[str, RetrievedPassage] = {}
    for lst in results_lists:
        for r in lst:
            t = re.sub(r"\s+", " ", (r.text or "").strip())
            if not t:
                continue
            if t not in best or float(r.score or 0.0) > float(best[t].score or 0.0):
                best[t] = r
    return list(best.values())


# ---------------------------
# Claim selection
# ---------------------------

def claim_priority(claim: str, char_name: str) -> int:
    """Prefer claims that are more checkable (entities/relations/numbers/events)."""
    c = (claim or "").strip()
    lc = c.lower()
    score = 0
    if char_name and char_name.lower() in lc:
        score += 3
    if re.search(r"\b(father|mother|brother|sister|wife|husband|son|daughter|married|divorced)\b", lc):
        score += 2
    if re.search(r"\b\d+\b", lc):
        score += 2
    if re.search(r"\b(arrested|imprisoned|escaped|killed|born|died|returned|wrote|married)\b", lc):
        score += 1
    if any(w[:1].isupper() for w in c.split()[1:]):
        score += 1
    return score


def is_soft_claim(claim: str) -> bool:
    """Filter vibe/mental-state claims that often trigger false contradictions."""
    lc = (claim or "").lower()
    soft_markers = [
        "felt", "believed", "hoped", "wanted", "dreamed", "feared",
        "was kind", "was brave", "was generous", "was nice",
    ]
    return any(m in lc for m in soft_markers)


# ---------------------------
# Labels
# ---------------------------

def label_to_int(x: Any) -> int:
    """
    train.csv uses: consistent / contradict
    We map:
      consistent -> 1
      contradict -> 0
    """
    if isinstance(x, (int, float)) and not pd.isna(x):
        return int(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "consistent"}:
        return 1
    if s in {"0", "false", "no", "contradict", "contradiction", "inconsistent"}:
        return 0
    raise ValueError(f"Unknown label value: {x}")


# ---------------------------
# Config
# ---------------------------

@dataclass(frozen=True)
class Config:
    contradiction_threshold: float
    margin: float
    entail_veto_threshold: float
    min_contradicted_claims: int
    high_contra_threshold: float
    candidate_k: int
    top_k: int
    max_sentences: int
    bm25: int
    skip_soft: int


# ---------------------------
# Decision rules
# ---------------------------

def claim_is_contradicted(
    preds,
    *,
    contradiction_threshold: float,
    margin: float,
    entail_veto_threshold: float,
) -> tuple[bool, float, float]:
    """
    Claim-level contradiction:
      max_contra >= threshold
      max_contra - max_entail >= margin
      max_entail < entail_veto_threshold  (veto false contradictions)
    Returns: (is_contra, max_contra, max_entail)
    """
    if not preds:
        return False, 0.0, 0.0
    max_contra = max(p.p_contra for p in preds)
    max_entail = max(p.p_entail for p in preds)

    ok = (
        (max_contra >= contradiction_threshold)
        and ((max_contra - max_entail) >= margin)
        and (max_entail < entail_veto_threshold)
    )
    return ok, float(max_contra), float(max_entail)


# ---------------------------
# Core predictor (matches predict.py logic)
# ---------------------------

def predict_row(
    row: pd.Series,
    *,
    retriever: PathwayRetrieverClient,
    nli: NLIVerifier,
    cfg: Config,
    text_col: str,
    book_col: str,
    char_col: str,
    caption_col: str,
    max_claims: int,
    retrieval_cache: Dict[Tuple[str, int], List[RetrievedPassage]],
) -> int:
    """
    Returns:
      1 => consistent
      0 => contradict / inconsistent
    """
    backstory = str(row.get(text_col, "") or "")
    book_slug = slugify(str(row.get(book_col, "") or ""))
    char_name = str(row.get(char_col, "") or "").strip()
    caption = str(row.get(caption_col, "") or "").strip()

    claims = split_into_claims(backstory)
    claims = [ground_claim(c, char_name) for c in claims]
    claims = [c for c in claims if c.strip()]

    if cfg.skip_soft:
        claims = [c for c in claims if not is_soft_claim(c)]

    claims = sorted(claims, key=lambda c: claim_priority(c, char_name), reverse=True)
    claims = claims[:max_claims]

    contradicted_claims = 0

    for claim in claims:
        # Multi-query retrieval
        queries = [claim]
        if char_name:
            queries.append(f"{char_name} {claim}")
        if caption and char_name:
            queries.append(f"{char_name} {caption}")

        retrieved_lists: List[List[RetrievedPassage]] = []
        for q in queries:
            ck = (q, cfg.candidate_k)
            if ck not in retrieval_cache:
                retrieval_cache[ck] = retriever.retrieve(q, k=cfg.candidate_k)
            res = retrieval_cache[ck]
            res = [r for r in res if _passage_in_book(r, book_slug)]
            retrieved_lists.append(res)

        merged = merge_and_dedupe(retrieved_lists)
        if not merged:
            continue

        passages = [m.text for m in merged if (m.text or "").strip()]
        if not passages:
            continue

        if cfg.bm25 and len(passages) > 1:
            order = bm25_rerank(claim, passages)
            if order:
                passages = [passages[i] for i in order]

        passages = passages[: cfg.top_k]

        premises = [sentence_focus(claim, p, max_sentences=cfg.max_sentences) for p in passages]
        premises = [p for p in premises if p.strip()]
        if not premises:
            continue

        preds = nli.predict_batch(premises, [claim] * len(premises))
        is_contra, max_contra, max_entail = claim_is_contradicted(
            preds,
            contradiction_threshold=cfg.contradiction_threshold,
            margin=cfg.margin,
            entail_veto_threshold=cfg.entail_veto_threshold,
        )
        if not is_contra:
            continue

        contradicted_claims += 1
        if contradicted_claims >= cfg.min_contradicted_claims or max_contra >= cfg.high_contra_threshold:
            return 0

    return 1


# ---------------------------
# Parsing helpers
# ---------------------------

def _parse_floats(s: str) -> List[float]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    return out


def _parse_ints(s: str) -> List[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


# ---------------------------
# Main tuning loop
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--server_url", type=str, default="http://127.0.0.1:8765")
    ap.add_argument("--input_csv", type=str, default="data/train.csv")

    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--text_col", type=str, default="content")
    ap.add_argument("--book_col", type=str, default="book_name")
    ap.add_argument("--char_col", type=str, default="char")
    ap.add_argument("--caption_col", type=str, default="caption")

    # NLI
    ap.add_argument("--nli_model", type=str, default="roberta-large-mnli")
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--nli_batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=384)

    # Claims
    ap.add_argument("--max_claims", type=int, default=60)

    # --- Anti-overfit tuning approach ---
    ap.add_argument("--dev_size", type=float, default=0.2, help="Holdout fraction for each split (e.g., 0.2)")
    ap.add_argument("--seeds", type=str, default="13,37,101", help="Repeated stratified split seeds")
    ap.add_argument("--final_cv_splits", type=int, default=5, help="CV folds to evaluate best config")

    # Grid controls (kept small by default on purpose)
    ap.add_argument("--contradiction_thresholds", type=str, default="0.60,0.65,0.70,0.75,0.80")
    ap.add_argument("--margins", type=str, default="0.05,0.10")
    ap.add_argument("--entail_veto_thresholds", type=str, default="0.70,0.75,0.80")

    ap.add_argument("--min_contradicted_claims", type=str, default="1,2")
    ap.add_argument("--high_contra_thresholds", type=str, default="0.85,0.90")

    ap.add_argument("--candidate_ks", type=str, default="40,60")
    ap.add_argument("--top_ks", type=str, default="8,12")
    ap.add_argument("--max_sentences_options", type=str, default="2,3")

    ap.add_argument("--bm25_options", type=str, default="0,1")
    ap.add_argument("--skip_soft_options", type=str, default="0,1")

    # Safety: cap configs (random sample if too large)
    ap.add_argument("--max_configs", type=int, default=250)

    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    for c in [args.label_col, args.text_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Columns: {list(df.columns)}")

    y = [label_to_int(v) for v in df[args.label_col].tolist()]
    y_series = pd.Series(y)

    # Models
    retriever = PathwayRetrieverClient(args.server_url)
    nli = NLIVerifier(
        model_name=args.nli_model,
        device=args.device,
        batch_size=args.nli_batch_size,
        max_length=args.max_length,
        fp16=True,
    )

    # Grid build
    ths = _parse_floats(args.contradiction_thresholds)
    margins = _parse_floats(args.margins)
    veto_ths = _parse_floats(args.entail_veto_thresholds)
    min_ccs = _parse_ints(args.min_contradicted_claims)
    high_ths = _parse_floats(args.high_contra_thresholds)
    cand_ks = _parse_ints(args.candidate_ks)
    top_ks = _parse_ints(args.top_ks)
    max_sents = _parse_ints(args.max_sentences_options)
    bm25_opts = _parse_ints(args.bm25_options)
    soft_opts = _parse_ints(args.skip_soft_options)

    grid = [
        Config(th, m, v, mincc, highth, ck, tk, ms, bm25, soft)
        for th, m, v, mincc, highth, ck, tk, ms, bm25, soft in itertools.product(
            ths, margins, veto_ths, min_ccs, high_ths, cand_ks, top_ks, max_sents, bm25_opts, soft_opts
        )
    ]

    # Reduce grid if too big (deterministic sampling)
    if len(grid) > args.max_configs:
        rng = random.Random(0)
        grid = rng.sample(grid, args.max_configs)

    seeds = _parse_ints(args.seeds)
    splitter = StratifiedShuffleSplit(
        n_splits=len(seeds),
        test_size=args.dev_size,
        random_state=seeds[0] if seeds else 13,
    )

    # We will manually use different random states per seed for stability
    split_indices: List[Tuple[List[int], List[int]]] = []
    for sd in seeds:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.dev_size, random_state=sd)
        tr_idx, dv_idx = next(sss.split(df, y_series))
        split_indices.append((tr_idx.tolist(), dv_idx.tolist()))

    print(f"\nRows: {len(df)} | Grid (sampled): {len(grid)} configs")
    print(f"Repeated dev splits: {len(split_indices)} (dev_size={args.dev_size})")
    print(f"Final CV splits for best config: {args.final_cv_splits}\n")

    best_cfg: Optional[Config] = None
    best_score = -1.0
    best_macro = -1.0

    # Cache retrieval across entire run
    retrieval_cache: Dict[Tuple[str, int], List[RetrievedPassage]] = {}

    for cfg in grid:
        f1s_0: List[float] = []
        macros: List[float] = []

        for _, dv_idx in split_indices:
            y_true = [y[i] for i in dv_idx]
            preds: List[int] = []

            for i in dv_idx:
                p = predict_row(
                    df.iloc[i],
                    retriever=retriever,
                    nli=nli,
                    cfg=cfg,
                    text_col=args.text_col,
                    book_col=args.book_col,
                    char_col=args.char_col,
                    caption_col=args.caption_col,
                    max_claims=args.max_claims,
                    retrieval_cache=retrieval_cache,
                )
                preds.append(p)

            f1_0 = f1_score(y_true, preds, pos_label=0)
            macro = f1_score(y_true, preds, average="macro")
            f1s_0.append(float(f1_0))
            macros.append(float(macro))

        mean_f1_0 = sum(f1s_0) / len(f1s_0)
        mean_macro = sum(macros) / len(macros)

        print(
            f"cfg: th={cfg.contradiction_threshold:.2f}, m={cfg.margin:.2f}, veto={cfg.entail_veto_threshold:.2f}, "
            f"min_cc={cfg.min_contradicted_claims}, high={cfg.high_contra_threshold:.2f}, "
            f"cand_k={cfg.candidate_k}, top_k={cfg.top_k}, sents={cfg.max_sentences}, "
            f"bm25={cfg.bm25}, soft={cfg.skip_soft} "
            f"=> avgF1(class0)={mean_f1_0:.3f}, avgMacroF1={mean_macro:.3f}"
        )

        if (mean_f1_0 > best_score) or (abs(mean_f1_0 - best_score) < 1e-9 and mean_macro > best_macro):
            best_score = mean_f1_0
            best_macro = mean_macro
            best_cfg = cfg

    print("\n====================")
    print("BEST CONFIG FOUND (by avg dev F1(class0))")
    print("====================")
    print(best_cfg)
    print(f"Avg dev F1(class0)={best_score:.3f} | Avg dev macroF1={best_macro:.3f}\n")

    # ---- Final: unbiased-ish evaluation for best config using CV OOF preds ----
    if best_cfg is None:
        raise RuntimeError("No best config found (unexpected).")

    skf = StratifiedKFold(n_splits=args.final_cv_splits, shuffle=True, random_state=42)
    oof = [-1] * len(df)

    for train_idx, val_idx in skf.split(df, y_series):
        for i in val_idx:
            oof[i] = predict_row(
                df.iloc[i],
                retriever=retriever,
                nli=nli,
                cfg=best_cfg,
                text_col=args.text_col,
                book_col=args.book_col,
                char_col=args.char_col,
                caption_col=args.caption_col,
                max_claims=args.max_claims,
                retrieval_cache=retrieval_cache,
            )

    y_true_all = y
    y_pred_all = oof

    print("Classification report (OOF CV for best config):")
    print(classification_report(y_true_all, y_pred_all, digits=3))
    print(f"OOF F1(class0)={f1_score(y_true_all, y_pred_all, pos_label=0):.3f}")
    print(f"OOF macroF1={f1_score(y_true_all, y_pred_all, average='macro'):.3f}")

    print("\nUse these flags in predict.py (copy/paste pattern):")
    print(f"--candidate_k {best_cfg.candidate_k} --top_k {best_cfg.top_k} "
          f"--contradiction_threshold {best_cfg.contradiction_threshold} --margin {best_cfg.margin} "
          f"--entail_veto_threshold {best_cfg.entail_veto_threshold} "
          f"--min_contradicted_claims {best_cfg.min_contradicted_claims} "
          f"--high_contra_threshold {best_cfg.high_contra_threshold} "
          f"--max_sentences {best_cfg.max_sentences} "
          f"{'--bm25_rerank' if best_cfg.bm25 else ''} "
          f"{'--skip_soft_claims' if best_cfg.skip_soft else ''}".strip())


if __name__ == "__main__":
    main()
