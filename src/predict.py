# src/predict.py
from __future__ import annotations

import argparse
import re
from typing import Any, Dict, List, Sequence, Optional

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score

from src.retrieve_client import PathwayRetrieverClient, RetrievedPassage
from src.utils import split_into_claims, ground_claim, slugify
from src.verifier_nli import NLIVerifier


# ---------------------------
# Text helpers
# ---------------------------

def _tok(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def bm25_rerank(query: str, passages: List[str]) -> List[int]:
    """Rerank retrieved chunks using BM25 on just the retrieved texts."""
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


def sentence_focus(claim: str, passage: str, max_sentences: int = 3) -> str:
    """Reduce premise to top-N most claim-relevant sentences."""
    if not passage or not passage.strip():
        return ""

    sents = re.split(r"(?<=[.!?])\s+", passage)
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return passage

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
    """Filter by book slug using metadata path if available."""
    if not book_slug:
        return True
    meta = p.metadata or {}
    path = str(meta.get("path", meta.get("file_path", meta.get("source", "")))).lower()
    if not path:
        return True  # can't filter reliably
    return book_slug in path


def merge_and_dedupe(results_lists: Sequence[Sequence[RetrievedPassage]]) -> List[RetrievedPassage]:
    """Merge multiple lists, dedupe by normalized text, keep best score."""
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
    """Higher = more verifiable."""
    c = (claim or "").strip()
    lc = c.lower()
    score = 0
    if char_name and char_name.lower() in lc:
        score += 3
    if re.search(r"\b(father|mother|brother|sister|wife|husband|son|daughter|married|divorced)\b", lc):
        score += 2
    if re.search(r"\b\d+\b", lc):
        score += 2
    if re.search(r"\b(arrested|imprisoned|escaped|killed|born|died|returned|wrote|moved|traveled|travelled)\b", lc):
        score += 1
    if any(w[:1].isupper() for w in c.split()[1:]):
        score += 1
    return score


def is_soft_claim(claim: str) -> bool:
    """Skip emotion/intention-heavy claims if enabled."""
    lc = (claim or "").lower()
    soft_markers = [
        "felt", "believed", "hoped", "wanted", "dreamed", "feared",
        "thought", "seemed", "appeared"
    ]
    return any(m in lc for m in soft_markers)


def label_to_int(x: Any) -> int:
    """train.csv label is 'consistent'/'contradict'."""
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "consistent"}:
        return 1
    if s in {"0", "false", "no", "contradict", "contradiction", "inconsistent"}:
        return 0
    raise ValueError(f"Unknown label value: {x}")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--server_url", type=str, default="http://127.0.0.1:8765")
    ap.add_argument("--input_csv", type=str, default="data/test.csv")
    ap.add_argument("--output_csv", type=str, default="results.csv")

    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--text_col", type=str, default="content")
    ap.add_argument("--label_col", type=str, default="")  # optional for train eval

    ap.add_argument("--book_col", type=str, default="book_name")
    ap.add_argument("--char_col", type=str, default="char")
    ap.add_argument("--caption_col", type=str, default="caption")

    # Retrieval
    ap.add_argument("--candidate_k", type=int, default=40)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--bm25_rerank", action="store_true")

    # Claims
    ap.add_argument("--max_claims", type=int, default=60)
    ap.add_argument("--max_sentences", type=int, default=3)
    ap.add_argument("--skip_soft_claims", action="store_true")

    # NLI
    ap.add_argument("--nli_model", type=str, default="roberta-large-mnli")
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--nli_batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=384)

    # Optional confirmation model
    ap.add_argument("--confirm_nli_model", type=str, default="")
    ap.add_argument("--confirm_threshold", type=float, default=0.75)

    # Decision thresholds (MATCH tune.py)
    ap.add_argument("--contradiction_threshold", type=float, default=0.70)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--entail_veto_threshold", type=float, default=0.80)

    ap.add_argument("--min_contradicted_claims", type=int, default=1)
    ap.add_argument("--high_contra_threshold", type=float, default=0.85)

    ap.add_argument("--save_rationale", action="store_true")

    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    if args.text_col not in df.columns:
        raise ValueError(f"Missing text_col={args.text_col}. Found: {list(df.columns)}")
    if args.id_col not in df.columns:
        df[args.id_col] = range(len(df))

    retriever = PathwayRetrieverClient(args.server_url)

    nli = NLIVerifier(
        model_name=args.nli_model,
        device=args.device,
        batch_size=args.nli_batch_size,
        max_length=args.max_length,
        fp16=True,
    )

    confirm_nli = None
    if args.confirm_nli_model.strip():
        confirm_nli = NLIVerifier(
            model_name=args.confirm_nli_model.strip(),
            device=args.device,
            batch_size=max(4, args.nli_batch_size // 2),
            max_length=args.max_length,
            fp16=True,
        )

    outputs: List[Dict[str, Any]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        story_id = row[args.id_col]
        text = str(row.get(args.text_col, "") or "")

        book_slug = slugify(str(row.get(args.book_col, "") or ""))
        char_name = str(row.get(args.char_col, "") or "").strip()
        caption = str(row.get(args.caption_col, "") or "").strip()

        # 1) Claim extraction + grounding
        claims = split_into_claims(text)
        claims = [ground_claim(c, char_name) for c in claims]

        if args.skip_soft_claims:
            claims = [c for c in claims if not is_soft_claim(c)]

        # prioritize verifiable claims first
        claims = sorted(claims, key=lambda c: claim_priority(c, char_name), reverse=True)
        claims = claims[: args.max_claims]

        contradicted_claims = 0
        best_rationale = {"claim": "", "passage": "", "p": 0.0}
        story_inconsistent = False

        # 2) Per-claim retrieval + NLI
        for claim in claims:
            queries = [claim]
            if char_name:
                q2 = f"{char_name} {claim}".strip()
                if q2 != claim:
                    queries.append(q2)
            if caption and char_name:
                queries.append(f"{char_name} {caption}".strip())

            retrieved_lists: List[List[RetrievedPassage]] = []
            for q in queries:
                res = retriever.retrieve(q, k=args.candidate_k)
                res = [r for r in res if _passage_in_book(r, book_slug)]
                retrieved_lists.append(res)

            merged = merge_and_dedupe(retrieved_lists)
            passages = [m.text for m in merged if (m.text or "").strip()]
            if not passages:
                continue

            if args.bm25_rerank and len(passages) > 1:
                order = bm25_rerank(claim, passages)
                if order:
                    passages = [passages[i] for i in order]

            passages = passages[: args.top_k]

            premises = [sentence_focus(claim, p, max_sentences=args.max_sentences) for p in passages]
            hyps = [claim] * len(premises)
            preds = nli.predict_batch(premises, hyps)
            if not preds:
                continue

            # 3) best contradiction for this claim
            best_i = max(range(len(preds)), key=lambda i: preds[i].p_contra)
            p = preds[best_i]
            best_contra = float(p.p_contra)
            best_entail = float(p.p_entail)

            # entail veto (IMPORTANT: matches your tune knobs)
            if best_entail >= args.entail_veto_threshold:
                continue

            is_contra = (best_contra >= args.contradiction_threshold) and ((best_contra - best_entail) >= args.margin)
            if not is_contra:
                continue

            # optional confirmer only on best evidence
            if confirm_nli is not None:
                conf = confirm_nli.predict_batch([premises[best_i]], [claim])[0]
                if float(conf.p_contra) < args.confirm_threshold:
                    continue

            contradicted_claims += 1

            if best_contra > best_rationale["p"]:
                best_rationale = {
                    "claim": claim,
                    "passage": passages[best_i][:500],
                    "p": best_contra,
                }

            # story-level rule
            if contradicted_claims >= args.min_contradicted_claims or best_contra >= args.high_contra_threshold:
                story_inconsistent = True
                break

        pred = 0 if story_inconsistent else 1
        out: Dict[str, Any] = {args.id_col: story_id, "prediction": pred}

        if args.save_rationale:
            out.update(
                {
                    "contradicted_claim": best_rationale["claim"],
                    "evidence_passage": best_rationale["passage"],
                    "evidence_contradiction_prob": float(best_rationale["p"] or 0.0),
                }
            )

        outputs.append(out)

    out_df = pd.DataFrame(outputs)
    out_df.to_csv(args.output_csv, index=False)
    print(f"✅ Saved predictions to {args.output_csv}")

    # Optional: evaluate on train if label_col exists
    if args.label_col and args.label_col in df.columns:
        y_true = [label_to_int(x) for x in df[args.label_col].tolist()]
        y_pred = out_df["prediction"].tolist()
        print("\nTrain evaluation (using same pipeline params):")
        print(classification_report(y_true, y_pred, digits=3))
        print("F1(class0) =", f1_score(y_true, y_pred, pos_label=0))
        print("macroF1     =", f1_score(y_true, y_pred, average="macro"))


if __name__ == "__main__":
    main()
