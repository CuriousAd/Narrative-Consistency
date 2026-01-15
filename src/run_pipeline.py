# src/run_pipeline.py
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Optional

from src.retrieve_client import PathwayRetrieverClient


def wait_for_server(server_url: str, timeout_s: int = 240, poll_s: float = 2.0) -> None:
    """
    Wait until Pathway server responds to retrieve calls.
    """
    client = PathwayRetrieverClient(server_url, timeout_s=10.0)
    t0 = time.time()
    last_err: Optional[str] = None

    while time.time() - t0 < timeout_s:
        try:
            _ = client.retrieve("ping", k=1)
            return
        except Exception as e:
            last_err = str(e)
            time.sleep(poll_s)

    raise RuntimeError(f"Server did not become ready within {timeout_s}s. Last error: {last_err}")


def terminate_process(proc: subprocess.Popen) -> None:
    """
    Try graceful shutdown, then force kill.
    """
    if proc.poll() is not None:
        return

    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
        return
    except Exception:
        pass

    try:
        proc.terminate()
        proc.wait(timeout=10)
        return
    except Exception:
        pass

    try:
        proc.kill()
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--books_dir", type=str, default="data/books")
    ap.add_argument("--input_csv", type=str, default="data/test.csv")
    ap.add_argument("--output_csv", type=str, default="predictions.csv")

    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--embed_device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--embed_batch_size", type=int, default=64)

    # Predict settings (use YOUR best tuned config as defaults)
    ap.add_argument("--text_col", type=str, default="content")
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--book_col", type=str, default="book_name")
    ap.add_argument("--char_col", type=str, default="char")
    ap.add_argument("--caption_col", type=str, default="caption")

    ap.add_argument("--nli_model", type=str, default="roberta-large-mnli")
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    ap.add_argument("--candidate_k", type=int, default=40)
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--contradiction_threshold", type=float, default=0.70)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--min_contradicted_claims", type=int, default=1)

    ap.add_argument("--bm25_rerank", action="store_true")
    ap.add_argument("--save_rationale", action="store_true")

    ap.add_argument("--server_ready_timeout_s", type=int, default=240)

    args = ap.parse_args()

    server_url = f"http://{args.host}:{args.port}"

    # Helpful env defaults (works CPU or GPU)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # 1) Start server
    server_cmd = [
        sys.executable, "-m", "src.index_server",
        "--books_dir", args.books_dir,
        "--host", args.host,
        "--port", str(args.port),
        "--embed_device", args.embed_device,
        "--embed_model", args.embed_model,
        "--embed_batch_size", str(args.embed_batch_size),
    ]

    print("\n[run_pipeline] Starting index server:")
    print("  " + " ".join(server_cmd))

    server_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        # 2) Wait for server
        print(f"\n[run_pipeline] Waiting for server: {server_url}")
        wait_for_server(server_url, timeout_s=args.server_ready_timeout_s)
        print("[run_pipeline] Server is ready ✅")

        # 3) Run predict
        predict_cmd = [
            sys.executable, "-m", "src.predict",
            "--server_url", server_url,
            "--input_csv", args.input_csv,
            "--output_csv", args.output_csv,
            "--text_col", args.text_col,
            "--id_col", args.id_col,
            "--book_col", args.book_col,
            "--char_col", args.char_col,
            "--caption_col", args.caption_col,
            "--nli_model", args.nli_model,
            "--device", args.device,
            "--candidate_k", str(args.candidate_k),
            "--top_k", str(args.top_k),
            "--contradiction_threshold", str(args.contradiction_threshold),
            "--margin", str(args.margin),
            "--min_contradicted_claims", str(args.min_contradicted_claims),
        ]

        if args.bm25_rerank:
            predict_cmd.append("--bm25_rerank")
        if args.save_rationale:
            predict_cmd.append("--save_rationale")

        print("\n[run_pipeline] Running predict:")
        print("  " + " ".join(predict_cmd))

        subprocess.check_call(predict_cmd)
        print(f"\n[run_pipeline] Done ✅ Output: {args.output_csv}")

    finally:
        # 4) Stop server
        print("\n[run_pipeline] Stopping server...")
        terminate_process(server_proc)

        # If you want to see server logs on failure, uncomment:
        # if server_proc.stdout:
        #     print(server_proc.stdout.read())


if __name__ == "__main__":
    main()
