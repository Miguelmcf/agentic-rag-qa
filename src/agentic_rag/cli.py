"""Command-line interface for ingesting documents and asking questions."""

from __future__ import annotations

import argparse
import sys

from .service import RagService


def _print_answer(service: RagService, question: str) -> None:
    answer = service.ask(question)
    print(f"\nQ: {answer.question}")
    print(f"A: {answer.answer}\n")
    if answer.citations:
        print("Sources:")
        for citation in answer.citations:
            print(f"  {citation.marker} {citation.source} - {citation.snippet}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentic-rag",
        description="Agentic RAG document Q&A.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Index a directory of documents.")
    ingest_parser.add_argument("directory", help="Path to a folder of .txt/.md files.")

    ask_parser = subparsers.add_parser("ask", help="Ask a single question.")
    ask_parser.add_argument("question", help="The question to answer.")
    ask_parser.add_argument(
        "--ingest",
        metavar="DIR",
        help="Optionally ingest this directory before asking.",
    )

    chat_parser = subparsers.add_parser("chat", help="Interactive Q&A session.")
    chat_parser.add_argument("--ingest", metavar="DIR", help="Directory to ingest first.")

    args = parser.parse_args(argv)
    service = RagService()

    if args.command == "ingest":
        count = service.ingest_directory(args.directory)
        print(f"Indexed {count} chunks (total: {service.document_count}).")
        return 0

    if args.command == "ask":
        if args.ingest:
            service.ingest_directory(args.ingest)
        if service.document_count == 0:
            print("No documents indexed. Use --ingest DIR or the 'ingest' command first.")
            return 1
        _print_answer(service, args.question)
        return 0

    if args.command == "chat":
        if args.ingest:
            count = service.ingest_directory(args.ingest)
            print(f"Indexed {count} chunks.")
        if service.document_count == 0:
            print("No documents indexed. Restart with --ingest DIR.")
            return 1
        print("Type your question (Ctrl-C or empty line to exit).")
        try:
            while True:
                question = input("> ").strip()
                if not question:
                    break
                _print_answer(service, question)
        except (KeyboardInterrupt, EOFError):
            print()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
