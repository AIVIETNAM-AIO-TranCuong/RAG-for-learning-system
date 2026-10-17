"""Real-provider release checks. Supply --url for a public deployment, or use
--deployment, --node and --vercel-cli for an authenticated Vercel preview.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url")
    parser.add_argument("--deployment")
    parser.add_argument("--node", default="node")
    parser.add_argument("--vercel-cli")
    args = parser.parse_args()
    if not args.url and not (args.deployment and args.vercel_cli):
        parser.error("Provide --url, or --deployment with --vercel-cli")

    def request(path, body=None):
        if args.url:
            response = httpx.request("POST" if body else "GET", args.url.rstrip("/") + "/api" + path,
                                     json=body, timeout=250)
            response.raise_for_status()
            return response.json()
        command = [args.node, args.vercel_cli, "curl", "/api" + path, "--deployment", args.deployment,
                   "--", "--silent", "--show-error", "--fail-with-body", "--max-time", "250"]
        with tempfile.TemporaryDirectory(prefix="rag-smoke-") as directory:
            if body:
                payload = Path(directory) / "request.json"
                payload.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
                command += ["--request", "POST", "--header", "Content-Type: application/json", "--data-binary", "@" + str(payload)]
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=270)
        if result.returncode:
            raise SystemExit(f"Preview request failed: {path} (exit {result.returncode})")
        try:
            return json.loads(result.stdout)
        except ValueError:
            raise SystemExit(f"Preview returned a non-JSON response: {path}") from None

    assert request("/health")["status"] == "healthy"
    documents = request("/documents")
    assert len(documents) >= 5
    document_id = next(d["document_id"] for d in documents if "LLM-Alignment" in d["filename"])
    print(f"Health/library OK: {len(documents)} documents", flush=True)
    answer = request("/ask", dict(question="Tài liệu trình bày những nội dung chính nào về LLM alignment?", document_id=document_id))
    assert answer["answer"] and answer["citations"]
    assert all(c["metadata"]["document_id"] == document_id for c in answer["chunks"])
    print(f"Q&A/citations OK: {len(answer['citations'])} citations, source filter verified", flush=True)
    summary = request("/summarize", dict(document_id=document_id))
    assert summary["content"] and summary["method"] == "map_reduce" and summary["total_chunks"] == 40
    print("Summary OK: all 40 chunks, Map-Reduce", flush=True)
    quiz = request("/quiz", dict(document_id=document_id, n_questions=3))
    assert len(quiz["items"]) == 3 and all(len(i["options"]) == 4 for i in quiz["items"])
    print("Quiz OK: 3 questions", flush=True)
    cards = request("/flashcards", dict(document_id=document_id, n_cards=5))
    assert len(cards["cards"]) == 5
    print("Flashcards OK: 5 cards", flush=True)


if __name__ == "__main__":
    main()
