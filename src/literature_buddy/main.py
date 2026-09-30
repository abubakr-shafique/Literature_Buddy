"""Command line entry point:  literature-buddy [PDF|URL]   |   index | ask | check | profiles"""

from __future__ import annotations

import argparse
import logging
import sys
import threading

from . import __version__
from .config.settings import available_profiles, load_config
from .logging_setup import setup_logging

COMMANDS = {"index", "ask", "check", "profiles"}


def _common() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--profile", help=f"hardware profile ({', '.join(available_profiles())})")
    p.add_argument("--config", help="path to a YAML config file")
    p.add_argument("--data-dir", help="where papers/indexes/chat history are stored")
    p.add_argument("--offline", action="store_true", help="never contact the Hugging Face Hub")
    p.add_argument("--log-level", help="DEBUG | INFO | WARNING")
    return p


def _load(args: argparse.Namespace):
    over: dict = {}
    if args.data_dir:
        over["data_dir"] = args.data_dir
    if args.offline:
        over["offline"] = True
    if args.log_level:
        over["log_level"] = args.log_level
    return load_config(profile=args.profile, config_path=args.config, overrides=over)


def _cli_progress(f: float, m: str) -> None:
    print(f"\r[{int(f * 100):3d}%] {m:<40}", end="", file=sys.stderr, flush=True)


def _cmd_index(args) -> int:
    from .services import Services

    svc = Services(_load(args))
    paper = svc.load_paper(args.paper, _cli_progress)
    d = paper.document
    print(f"\n{d.title}\n{d.page_count} pages, {len(d.figures)} figures, {len(d.tables)} tables, "
          f"{len(d.references)} references, {len(paper.store.chunks)} chunks -> {paper.directory}")
    return 0


def _cmd_ask(args) -> int:
    from .services import Services

    svc = Services(_load(args))
    paper = svc.load_paper(args.paper, _cli_progress)
    print(file=sys.stderr)
    pipe = svc.pipeline_for(paper)
    stop = threading.Event()
    for ev in pipe.ask(args.question, [], stop):
        if ev.kind == "status":
            print(f"· {ev.text}", file=sys.stderr)
        elif ev.kind == "token":
            print(ev.text, end="", flush=True)
        elif ev.kind == "done" and ev.result:
            print("\n\nSources:")
            for s in ev.result.display_sources:
                print(f"  [{s.sid}] {s.title}: {s.excerpt}")
            for w in ev.result.warnings:
                print(f"  ! {w}")
    return 0


def _cmd_check(args) -> int:
    import platform

    import pymupdf

    cfg = _load(args)
    print(f"Literature Buddy {__version__} · Python {platform.python_version()} · PyMuPDF {pymupdf.VersionBind}")
    try:
        import PySide6

        print(f"PySide6 {PySide6.__version__}")
    except ImportError:
        print("PySide6: MISSING (pip install -r requirements.txt)")
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA GPU"
        vram = f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB" if torch.cuda.is_available() else ""
        print(f"torch {torch.__version__} · {gpu} {vram}")
    except ImportError:
        print("torch: not installed (fine for the Ollama profiles)")
    print(f"Profile: {cfg.profile} · LLM {cfg.llm.provider}:{cfg.llm.model} · embeddings {cfg.embedding.provider}:{cfg.embedding.model}")
    from .models.model_manager import ModelManager

    ok, msg = ModelManager(cfg).health()
    print(("OK   " if ok else "FAIL ") + msg)
    return 0 if ok else 1


def _gui(args) -> int:
    from PySide6.QtWidgets import QApplication

    from .gui import theme
    from .gui.main_window import LOGO, MainWindow
    from .services import Services

    cfg = _load(args)
    setup_logging(cfg.log_level, cfg.data_path / "logs")
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Literature Buddy")
    from PySide6.QtGui import QIcon

    app.setWindowIcon(QIcon(str(LOGO)))
    app.setStyleSheet(theme.stylesheet())
    win = MainWindow(Services(cfg), args.paper)
    win.show()
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv and argv[0] in COMMANDS else "gui"
    if cmd != "gui":
        argv = argv[1:]
    common = _common()
    ap = argparse.ArgumentParser(prog="literature-buddy", parents=[common],
                                 description="Local, evidence-grounded reading assistant for scientific papers.")
    ap.add_argument("--version", action="version", version=__version__)
    if cmd == "gui":
        ap.add_argument("paper", nargs="?", help="PDF path or URL to open on start")
    elif cmd == "index":
        ap.add_argument("paper")
    elif cmd == "ask":
        ap.add_argument("paper")
        ap.add_argument("question")
    args = ap.parse_args(argv)
    if cmd != "gui":
        setup_logging(args.log_level or "WARNING")
    try:
        if cmd == "profiles":
            print("\n".join(available_profiles()))
            return 0
        return {"gui": _gui, "index": _cmd_index, "ask": _cmd_ask, "check": _cmd_check}[cmd](args)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).debug("fatal", exc_info=True)
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
