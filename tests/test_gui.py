import pytest

pytestmark = pytest.mark.gui
pytest.importorskip("PySide6")

from fake_ollama import FakeOllama  # noqa: E402

from literature_buddy.config import load_config  # noqa: E402
from literature_buddy.gui import theme  # noqa: E402
from literature_buddy.gui.main_window import MainWindow  # noqa: E402
from literature_buddy.services import Services  # noqa: E402


@pytest.fixture()
def fake():
    f = FakeOllama()
    yield f
    f.stop()


@pytest.fixture()
def win(qtbot, fake, tmp_path):
    cfg = load_config(profile="16gb-ollama", environ={}, overrides={
        "data_dir": str(tmp_path), "llm": {"base_url": fake.url, "model": "fake"},
        "embedding": {"base_url": fake.url}})
    w = MainWindow(Services(cfg))
    qtbot.addWidget(w)
    w.setStyleSheet(theme.stylesheet())
    w.show()
    return w


def _open(qtbot, win, pdf):
    win.open_source(str(pdf))
    qtbot.waitUntil(lambda: win.paper is not None, timeout=30000)


def _wait_answer(qtbot, win, n_assistant):
    qtbot.waitUntil(lambda: sum(m["role"] == "assistant" and m.get("final") for m in win.chat._msgs) >= n_assistant,
                    timeout=30000)


def test_open_highlight_ask_cite(qtbot, win, sample_pdf, fake):
    _open(qtbot, win, sample_pdf)
    assert win.viewer.page_count == 3 and win.chat.send_btn.isEnabled()
    assert "Ready" in win.chat._msgs[-1]["html"] or any("Ready" in m.get("html", "") for m in win.chat._msgs)

    # highlight -> panel + persistence
    rects, text = win.viewer.select_words(0, (70, 452), (300, 462))
    win.viewer.selected.emit(0, rects, text)
    assert len(win.highlights._items) == 1 and win.highlights.context()[0]["text"] == text
    hid = next(iter(win.highlights._items))
    win.highlights._items[hid].use_in_chat = False  # unticked => not sent
    win._on_toggle(hid, False)
    assert win.highlights.context() == []
    win._on_toggle(hid, True)

    # ask
    win.chat.sendRequested.emit("How many samples were used for training?")
    _wait_answer(qtbot, win, 1)
    m = win.chat._msgs[-1]
    assert "12,450 samples" in m["text"] and m["sources"] and m["sources"][0]["sid"].startswith(("S", "H"))
    chat_req = [r for r in fake.requests if r["path"] == "/api/chat"][-1]
    assert "[H1]" in chat_req["messages"][-1]["content"]  # ticked highlight was sent as context

    # citation click -> viewer navigates + outlines evidence
    sid = next(s["sid"] for s in m["sources"] if s["sid"].startswith("S"))
    win.chat.citationClicked.emit(len(win.chat._msgs) - 1, sid)
    assert win.viewer.state.evidence

    # remove highlight
    win._remove_highlights([hid])
    assert not win.highlights._items and not win.viewer.highlights
    win.grab().save("/tmp/gui_main.png")


def test_figure_question_uses_vlm_and_shows_thumbnail(qtbot, win, sample_pdf, fake):
    _open(qtbot, win, sample_pdf)
    win.chat.sendRequested.emit("What does Figure 1 show?")
    _wait_answer(qtbot, win, 1)
    imgs = [r for r in fake.requests if r["path"] == "/api/chat" and r["messages"][-1].get("images")]
    assert len(imgs) == 1
    m = win.chat._msgs[-1]
    assert any(s["label"] == "Figure 1" and s["image_path"] for s in m["sources"])
    win.chat.citationClicked.emit(len(win.chat._msgs) - 1, next(s["sid"] for s in m["sources"] if s["label"] == "Figure 1"))
    assert win.viewer.state.evidence
    win.grab().save("/tmp/gui_figure.png")


def test_reopen_uses_cache_and_restores_state(qtbot, win, sample_pdf):
    _open(qtbot, win, sample_pdf)
    win.chat.sendRequested.emit("How many samples were used for training?")
    _wait_answer(qtbot, win, 1)
    win.paper = None
    _open(qtbot, win, sample_pdf)
    assert win.paper.from_cache
    assert any(m["role"] == "assistant" for m in win.chat._msgs)  # history restored


def test_bad_file_shows_error_not_crash(qtbot, win, tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_text("not a pdf")
    win.open_source(str(bad))
    qtbot.waitUntil(lambda: any(m.get("kind") == "error" for m in win.chat._msgs), timeout=10000)
    assert win.paper is None


def test_keyboard_enter_sends(qtbot, win, sample_pdf):
    from PySide6.QtCore import Qt

    _open(qtbot, win, sample_pdf)
    win.chat.input.setPlainText("How many samples were used for training?")
    qtbot.keyClick(win.chat.input, Qt.Key.Key_Return)
    _wait_answer(qtbot, win, 1)
    assert win.chat.input.toPlainText() == ""
