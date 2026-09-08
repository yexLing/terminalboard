"""Smoke + unit tests. Run with: pytest -q"""
from __future__ import annotations

import json
import math
import os
import re

import pytest

from terminalboard.app import App, match_filter, _prev_word, _next_word
from terminalboard.model import HistogramSeries, Run, ScalarSeries, TextSeries
from terminalboard.reader import make_reader
from terminalboard.render import TextRenderer, shorten_tag, ema


# -- parsers -----------------------------------------------------------------

def test_light_parser_reads_scalars(logdir):
    runs = make_reader(str(logdir)).poll()             # default = light
    s = runs["run_a"].series["train/loss"]
    assert s.kind == "scalar"
    assert s.steps == list(range(0, 100, 10))
    assert s.values[0] == pytest.approx(1.0, rel=1e-5)
    assert s.values[-1] == pytest.approx(math.exp(-90 / 50.0), rel=1e-5)


def test_light_parser_reads_text_and_histogram(logdir):
    runs = make_reader(str(logdir)).poll()
    series = runs["run_a"].series
    assert series["note/info"].kind == "text"
    assert series["note/info"].texts[-1] == "hello\nworld"
    h = series["weights/h"]
    assert h.kind == "histogram"
    assert len(h) == 10
    edges, counts = h.buckets[0]
    assert edges and counts


def test_tb_parser_matches_light(logdir):
    pytest.importorskip("tensorboard")
    light = make_reader(str(logdir)).poll()["run_a"].series["train/loss"]
    tb = make_reader(str(logdir), use_tb=True).poll()["run_a"].series["train/loss"]
    assert light.steps == tb.steps
    assert light.values == pytest.approx(tb.values, rel=1e-5)


# -- rendering ---------------------------------------------------------------

def test_text_render_nonempty(logdir):
    runs = make_reader(str(logdir)).poll()
    body = TextRenderer().frame(runs, ["train/loss", "train/acc"],
                                smooth=0.5, max_cols=2, width=90, height=20)
    assert isinstance(body, str) and body.strip()


def test_text_render_fits_height(logdir):
    runs = make_reader(str(logdir)).poll()
    body = TextRenderer().frame(runs, ["train/loss"], width=80, height=18)
    assert body.count("\n") + 1 <= 18


def test_render_all_kinds_mixed(logdir):
    runs = make_reader(str(logdir)).poll()
    body = TextRenderer().frame(
        runs, ["train/loss", "note/info", "weights/h"],
        max_cols=3, width=120, height=20,
    )
    assert isinstance(body, str) and body.strip()


def test_flat_series_does_not_hang():
    run = Run("r", "/tmp")
    run.series["flat"] = ScalarSeries("flat", [0, 1, 2, 3], [0.0, 0.0, 0.0, 0.0])
    body = TextRenderer().frame({"r": run}, ["flat"], width=60, height=16)
    assert isinstance(body, str) and body.strip()


def test_partial_page_no_crash(logdir):
    runs = make_reader(str(logdir)).poll()
    body = TextRenderer().frame(runs, ["train/loss"], max_cols=2, width=90, height=20)
    assert isinstance(body, str)


# -- filter grammar ----------------------------------------------------------

def test_match_filter_basic():
    assert match_filter("loss", "train/loss")
    assert match_filter("LOSS", "train/loss")            # case-insensitive
    assert match_filter("train/*acc*", "train/acc")      # glob
    assert not match_filter("xyz", "train/loss")
    assert match_filter(None, "anything")                # empty matches all


def test_match_filter_or_and_not_regex():
    assert match_filter("loss | acc", "train/acc")       # OR with |
    assert match_filter("a, b", "xbx")                   # OR with ,
    assert match_filter("train loss", "train/loss")      # AND (both present)
    assert not match_filter("train loss", "val/loss")    # AND fails (no train)
    assert match_filter("!val", "train/loss")            # NOT
    assert not match_filter("!loss", "train/loss")
    assert match_filter("/lo.s/", "train/loss")          # regex (per-word)
    assert not match_filter("/^loss$/", "train/loss")


def test_match_filter_global_exclude():
    # `!x` excludes globally, no matter which OR alternative it sits in:
    # (a or b or c) and NOT d
    flt = "action_token_acc | visual_token_accuracy | action_refine & !by_format"
    assert match_filter(flt, "val/action_token_acc")
    assert match_filter(flt, "train/action_refine_loss")
    assert not match_filter(flt, "by_format/action_refine")        # excluded
    assert not match_filter(flt, "by_format/visual_token_accuracy")  # matched OR, still excluded
    assert not match_filter(flt, "other/thing")                    # no positive match
    # exclude lands in the first alternative but still applies everywhere
    assert not match_filter("a !d | b", "b_has_d")
    assert match_filter("a !d | b", "b_clean")
    # only-excludes keeps everything else
    assert match_filter("!old", "train/loss") and not match_filter("!old", "old/x")


def test_match_filter_whole_pattern_regex():
    # a whole-pattern /.../ is real regex, so | and spaces work inside it
    assert match_filter("/(loss|lr)/", "train/lr")
    assert match_filter("/(loss|lr)/", "val/loss")
    assert not match_filter("/(loss|lr)/", "train/acc")
    assert match_filter(r"/^train\/(loss|lr)$/", "train/loss")
    assert not match_filter(r"/^train\/(loss|lr)$/", "train/visual_loss")
    assert not match_filter("/[unclosed/", "anything")   # bad regex -> no match


# -- word-edit helpers -------------------------------------------------------

def test_word_motion():
    buf = list("train/loss val")
    assert _prev_word(buf, len(buf)) == buf.index("v")   # back to 'val'
    assert _next_word(buf, 0) == len("train")            # forward over 'train'


# -- z-order -----------------------------------------------------------------

def test_run_order_cycles(logdir):
    app = App(make_reader(str(logdir)), TextRenderer())
    app.reader.poll()
    base = app._run_order()
    app._order_rot += 1
    rotated = app._run_order()
    assert set(base) == set(rotated)
    if len(base) > 1:
        assert base != rotated                            # different top


# -- drill-down (focus + detail) ---------------------------------------------

class _FakeScreen:
    def draw(self, frame, hard=False):
        pass


def _app(logdir):
    a = App(make_reader(str(logdir)), TextRenderer(), rows=2, cols=2)
    a.reader.poll()
    return a


def test_focus_navigation(logdir):
    a = _app(logdir)
    a._handle_grid_key(_FakeScreen(), None, "RIGHT")
    assert a._focus == 1
    a._handle_grid_key(_FakeScreen(), None, "DOWN")
    assert a._focus == 1 + a.cols


def test_detail_text_scroll_and_switch(logdir):
    a = _app(logdir)
    a.tag_filter = "note/info"
    a._handle_grid_key(_FakeScreen(), None, "\r")        # Enter -> detail
    assert a._detail == "note/info"
    frame = a._build_detail_frame()
    assert "note/info" in frame
    a._handle_detail_key(_FakeScreen(), None, "DOWN")
    assert a._scroll == 1
    a._handle_detail_key(_FakeScreen(), None, "RIGHT")                        # switch experiment (wraps)
    assert a._build_detail_frame().strip()
    a._handle_detail_key(_FakeScreen(), None, "ESC")                          # back to grid
    assert a._detail is None


def test_detail_histogram_and_scalar(logdir):
    a = _app(logdir)
    for tag in ("weights/h", "train/loss"):
        a.tag_filter = tag
        a._detail = None
        a._handle_grid_key(_FakeScreen(), None, "\r")
        assert a._detail == tag
        assert a._build_detail_frame().strip()
        a._handle_detail_key(_FakeScreen(), None, "ESC")


def test_scalar_detail_cursor(logdir):
    a = _app(logdir)
    a.tag_filter = "train/loss"
    a._handle_grid_key(_FakeScreen(), None, "\r")
    track = a._scalar_track("train/loss", a._detail_runs())
    mid = (len(track) - 1) // 2
    assert a._cursor == mid                       # starts in the middle
    a._handle_detail_key(_FakeScreen(), None, "LEFT")
    assert a._cursor == mid - 1
    a._handle_detail_key(_FakeScreen(), None, "HOME")
    assert a._cursor == 0
    a._handle_detail_key(_FakeScreen(), None, "END")
    assert a._cursor == len(track) - 1
    frame = a._build_detail_frame()
    assert "cursor @ step" in frame and "value" in frame and "smoothed" in frame


def _multi_type_logdir(tmp_path):
    """Two runs with a scalar, histogram, pr-curve, and hparams."""
    import conftest as C
    for exp, (lr, dp) in {"baseline": (0.01, 0.1), "high_lr": (0.03, 0.0)}.items():
        d = tmp_path / exp
        d.mkdir()
        recs = []
        for s in range(0, 30, 10):
            vals = [
                C.scalar_value("loss", 2.0 ** (-s / 10)),
                C.histogram_value("w/h", [-2., -1., 0., 1., 2.],
                                  [1., 4., 9., 4., 1.]),
                C.pr_curve_value("pr/cls", [1., .9, .7, .4], [0., .4, .7, 1.]),
            ]
            if s == 0:
                vals.append(C.hparams_experiment(["lr", "dropout"], ["loss"]))
                vals.append(C.hparams_session({"lr": lr, "dropout": dp, "opt": "adam"}))
            recs.append((s, vals))
        C.write_events(d / "events.out.tfevents.1.h.1.0", recs)
    return tmp_path


def test_parse_pr_curve_and_hparams(tmp_path):
    runs = make_reader(str(_multi_type_logdir(tmp_path))).poll()
    r = runs["baseline"]
    pr = r.series["pr/cls"]
    assert pr.kind == "pr_curve"
    assert pr.precision[-1] == pytest.approx([1., .9, .7, .4], rel=1e-5)
    assert pr.recall[-1] == pytest.approx([0., .4, .7, 1.], rel=1e-5)
    assert "_hparams_/session_start_info" not in r.series   # folded, not a tag
    assert r.hparams == {"lr": pytest.approx(0.01), "dropout": pytest.approx(0.1),
                         "opt": "adam"}
    assert r.hparam_info["hparams"] == ["lr", "dropout"]
    assert r.hparam_info["metrics"] == ["loss"]


def test_kind_filter_cycles(tmp_path):
    a = App(make_reader(str(_multi_type_logdir(tmp_path))), TextRenderer())
    a.reader.poll()
    assert set(a._matching_tags()) == {"loss", "w/h", "pr/cls"}
    a._handle_view_key("c")
    assert a._kind_filter == "scalar" and a._matching_tags() == ["loss"]
    a._handle_view_key("c")
    assert a._kind_filter == "histogram" and a._matching_tags() == ["w/h"]
    a._handle_view_key("c")
    assert a._kind_filter == "text" and a._matching_tags() == []
    a._handle_view_key("c")
    assert a._kind_filter == "pr_curve" and a._matching_tags() == ["pr/cls"]
    a._handle_view_key("c")
    assert a._kind_filter is None


def test_distribution_and_prcurve_render(tmp_path):
    a = App(make_reader(str(_multi_type_logdir(tmp_path))), TextRenderer())
    a.reader.poll()
    a._distmode = True                          # histograms -> bands
    assert a._build_frame().strip()
    # pr-curve detail with cursor stepping
    a.tag_filter = "pr/cls"
    a._handle_grid_key(_FakeScreen(), None, "\r")
    assert a._build_detail_frame().strip()
    a._handle_detail_key(_FakeScreen(), None, "END")
    track = a._scalar_track("pr/cls", a._detail_runs())
    assert a._cursor == len(track) - 1


def test_hparams_table_view(tmp_path):
    a = App(make_reader(str(_multi_type_logdir(tmp_path))), TextRenderer())
    a.reader.poll()
    a._handle_grid_key(_FakeScreen(), None, "P")
    assert a._hparams is True
    frame = a._build_hparams_frame()
    for token in ("lr", "dropout", "opt", "loss", "baseline", "high_lr"):
        assert token in frame
    a._handle_hparams_key(_FakeScreen(), None, "ESC")
    assert a._hparams is False


def test_scalar_cursor_track_is_union_of_runs(tmp_path):
    # two runs of different length: cursor must reach the longer run's last step
    import conftest as C
    for name, n in [("short", 5), ("long", 12)]:
        d = tmp_path / name
        d.mkdir()
        C.write_events(d / "events.out.tfevents.1.h.1.0",
                       [(i, [C.scalar_value("m", float(i))]) for i in range(n)])
    a = App(make_reader(str(tmp_path)), TextRenderer())
    a.reader.poll()
    a.tag_filter = "m"
    a._handle_grid_key(_FakeScreen(), None, "\r")
    track = a._scalar_track("m", a._detail_runs())
    assert track[-1] == 11                       # union reaches the long run
    assert a._cursor == (len(track) - 1) // 2    # starts in the middle
    a._handle_detail_key(_FakeScreen(), None, "END")
    assert track[a._cursor] == 11                # END lands on the furthest point


def test_config_diff(tmp_path):
    # two runs whose config differs in one key
    import conftest as C
    for name, lr in [("a", "0.001"), ("b", "0.003")]:
        d = tmp_path / name
        d.mkdir()
        cfg = '{\n  "lr": %s,\n  "model": "resnet"\n}' % lr
        C.write_events(d / "events.out.tfevents.1.h.1.0",
                       [(0, [C.text_value("config", cfg)])])
    a = App(make_reader(str(tmp_path)), TextRenderer())
    a.reader.poll()
    a.tag_filter = "config"
    a._handle_grid_key(_FakeScreen(), None, "\r")
    a._handle_detail_key(_FakeScreen(), None, "d")                       # toggle diff
    frame = a._build_detail_frame()
    assert "diff across 2 experiments" in frame
    assert "lr" in frame and "0.001" in frame and "0.003" in frame
    assert "model" not in frame                     # identical key is hidden


def test_csv_export(logdir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = _app(logdir)
    a.tag_filter = "train/loss"
    msg = a._export_csv()
    assert "wrote" in msg
    f = tmp_path / "train_loss.csv"
    assert f.exists()
    lines = f.read_text().splitlines()
    assert lines[0].startswith("step,") and len(lines) > 1


def test_csv_export_skips_nonscalar(logdir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = _app(logdir)
    a.tag_filter = "note/info"          # a text tag
    assert "not a scalar" in a._export_csv()


def test_csv_prompt_uses_csv_dir(logdir, tmp_path):
    out = tmp_path / "out"
    a = App(make_reader(str(logdir)), TextRenderer(), csv_dir=str(out))
    a.reader.poll()
    a.tag_filter = "train/loss"
    assert a._csv_default_path("train/loss") == str(out / "train_loss.csv")

    class _Keys:
        def __init__(self, seq):
            self.q = list(seq)

        def get(self, _t):
            return self.q.pop(0) if self.q else None

    a._do_csv(_FakeScreen(), _Keys(["\r"]))            # accept default
    assert (out / "train_loss.csv").exists()
    a._do_csv(_FakeScreen(), _Keys(["\x1b"]))          # Esc cancels
    assert a._status == ""


def test_view_state_persists(logdir, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    a = App(make_reader(str(logdir)), TextRenderer(), restore=True)
    a.reader.poll()
    a.tag_filter = "train/loss"
    a.run_filter = "run_a"
    a.smooth = 0.85
    a.xaxis = "time"
    a.logy = True
    a._zoom = 1
    a._focus = 0
    a._save_view()
    # a fresh app on the same logdir restores it
    b = App(make_reader(str(logdir)), TextRenderer(), restore=True)
    assert b.tag_filter == "train/loss"
    assert b.run_filter == "run_a"
    assert b.smooth == 0.85
    assert b.xaxis == "time"
    assert b.logy is True
    assert b._zoom == 1
    # restore_exclude lets an explicit CLI value win (not overwritten by saved)
    c = App(make_reader(str(logdir)), TextRenderer(), restore=True,
            tag_filter="other", restore_exclude={"tag_filter"})
    assert c.tag_filter == "other"
    assert c.smooth == 0.85                       # still restored


def test_load_view_tolerates_old_state_file(logdir, tmp_path, monkeypatch):
    # a state file written by an older version (pre-kind_filter / pre-chat) must
    # load without crashing — regression for KeyError: 'kind_filter'.
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    a = App(make_reader(str(logdir)), TextRenderer(), restore=True)
    p = a._view_state_file()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:                        # old schema: no kind_filter
        json.dump({"tag_filter": "train/loss", "smooth": 0.7, "zoom": 1}, f)
    b = App(make_reader(str(logdir)), TextRenderer(), restore=True)  # must not raise
    assert b.tag_filter == "train/loss"
    assert b.smooth == 0.7
    assert b._kind_filter is None                  # absent key → default (all)


# -- LLM assistant -----------------------------------------------------------

def _fake_completion(tool_calls=(), text="", usage=None):
    """A fake litellm.completion: returns an OpenAI-style response (dict form)."""
    def complete(**kwargs):
        return {
            "choices": [{"message": {
                "content": text,
                "tool_calls": [
                    {"function": {"name": n, "arguments": json.dumps(a)}}
                    for (n, a) in tool_calls],
            }}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
        }
    return complete


def test_llm_ask_parses_tool_calls():
    from terminalboard import llm
    cfg = llm.LLMConfig(model="x")
    out = llm.ask(cfg, [{"role": "user", "content": "hi"}], llm.build_tools(),
                  complete=_fake_completion(
                      tool_calls=[("set_tag_filter", {"pattern": "val/*"}),
                                  ("bogus_action", {})],
                      text="done"))
    assert out["text"] == "done"
    # unknown actions are dropped; known ones kept
    assert out["tool_calls"] == [("set_tag_filter", {"pattern": "val/*"})]


def test_llm_apply_actions(logdir):
    a = App(make_reader(str(logdir)), TextRenderer())
    a.reader.poll()
    assert a._llm_apply_action("set_tag_filter", {"pattern": "train/loss"})
    assert a.tag_filter == "train/loss"
    a._llm_apply_action("set_type", {"kind": "histogram"})
    assert a._kind_filter == "histogram"
    a._llm_apply_action("set_smoothing", {"value": 5})         # clamped
    assert a.smooth == 0.99
    a._llm_apply_action("set_distribution", {"on": True})
    assert a._distmode is True
    a._llm_apply_action("open_hparams", {})
    assert a._hparams is True
    a._llm_apply_action("set_tag_filter", {"pattern": None})
    assert a.tag_filter is None
    a._llm_apply_action("set_type", {"kind": "all"})           # clear kind filter
    assert a._kind_filter is None
    # open_detail only opens an existing (visible) tag
    assert a._llm_apply_action("open_detail", {"tag": "no/such"}) is None
    assert a._llm_apply_action("open_detail", {"tag": "train/loss"})
    assert a._detail == "train/loss"


def test_llm_run_navigates_and_records_history(logdir):
    a = App(make_reader(str(logdir)), TextRenderer())
    a.reader.poll()
    a._llm_config = __import__("terminalboard.llm", fromlist=["x"]).LLMConfig("m")
    a._llm_complete = _fake_completion(
        tool_calls=[("set_tag_filter", {"pattern": "train/acc"})],
        text="train/acc is rising.")
    text, applied, usage = a._llm_run("show accuracy")
    assert a.tag_filter == "train/acc"
    assert any("train/acc" in x for x in applied)
    assert "rising" in text
    assert a._llm_history[-2] == {"role": "user", "content": "show accuracy"}
    assistant = a._llm_history[-1]
    assert assistant["role"] == "assistant"
    assert "rising" in assistant["content"] and "[did:" in assistant["content"]


def test_llm_context_is_json(logdir):
    a = App(make_reader(str(logdir)), TextRenderer())
    a.reader.poll()
    ctx = json.loads(a._llm_context())
    assert "train/loss" in ctx["scalars"]
    assert ctx["tags_by_kind"]["scalar"]
    # Phase 2: trend + focus context
    assert ctx["scalars"]["train/loss"]["run_a"]["trend"] == "down"
    assert ctx["scalars"]["train/acc"]["run_a"]["trend"] == "up"
    # the chat must know the live view (focused/visible tags, counts)
    assert "focused_tag" in ctx["view"]
    assert ctx["view"]["mode"] == "grid"
    assert ctx["view"]["visible_count"] >= 1
    assert "train/loss" in ctx["view"]["visible_tags"]


def test_llm_followup_memory(logdir):
    from terminalboard import llm
    a = App(make_reader(str(logdir)), TextRenderer())
    a.reader.poll()
    a._llm_config = llm.LLMConfig("m")
    seen = {}

    def complete(**kw):
        seen["messages"] = kw["messages"]
        return {"choices": [{"message": {"content": "ok", "tool_calls": [
            {"function": {"name": "set_zoom", "arguments": json.dumps({"panels": 4})}}]}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    a._llm_complete = complete
    a._llm_run("zoom to four")
    # assistant turn records the action taken
    assert "[did:" in a._llm_history[-1]["content"]
    a._llm_run("now show losses")
    # the 2nd call's messages carry the 1st turn (follow-up memory)
    contents = [m.get("content", "") for m in seen["messages"]]
    assert any("zoom to four" in c for c in contents)
    assert any("[did:" in c for c in contents)


def _fake_stream(text_chunks, tool_calls=(), usage=None):
    """A fake streaming litellm.completion: yields OpenAI-style delta chunks."""
    def complete(**kwargs):
        chunks = [{"choices": [{"delta": {"content": c}}]} for c in text_chunks]
        for idx, (n, a) in enumerate(tool_calls):       # split name/args deltas
            s = json.dumps(a)
            chunks.append({"choices": [{"delta": {"tool_calls": [
                {"index": idx, "function": {"name": n, "arguments": s[:3]}}]}}]})
            chunks.append({"choices": [{"delta": {"tool_calls": [
                {"index": idx, "function": {"arguments": s[3:]}}]}}]})
        chunks.append({"choices": [{"delta": {}}],
                       "usage": usage or {"prompt_tokens": 3, "completion_tokens": 2}})
        return iter(chunks)
    return complete


def test_llm_ask_stream():
    from terminalboard import llm
    got = []
    out = llm.ask_stream(
        llm.LLMConfig("m"), [{"role": "user", "content": "hi"}], llm.build_tools(),
        complete=_fake_stream(["Hel", "lo ", "world"],
                              tool_calls=[("set_logy", {"on": True})]),
        on_delta=got.append)
    assert "".join(got) == "Hello world"
    assert out["text"] == "Hello world"
    assert out["tool_calls"] == [("set_logy", {"on": True})]
    assert out["usage"]["completion_tokens"] == 2


def test_llm_run_streaming(logdir):
    from terminalboard import llm
    a = App(make_reader(str(logdir)), TextRenderer())
    a.reader.poll()
    a._llm_config = llm.LLMConfig("m")
    a._llm_complete = _fake_stream(["analyzing… ", "loss looks good"],
                                   tool_calls=[("set_smoothing", {"value": 0.5})])
    got = []
    text, applied, usage = a._llm_run("how is it", on_delta=got.append)
    assert "".join(got).startswith("analyzing")
    assert a.smooth == 0.5
    assert "loss looks good" in text


def test_llm_friendly_error_and_cost():
    from terminalboard import llm
    assert "Auth" in llm.friendly_error(Exception("Invalid API key provided"))
    assert "Rate" in llm.friendly_error(Exception("rate limit exceeded (429)"))
    assert "Model not found" in llm.friendly_error(Exception("model does not exist"))
    # cost is None when litellm is absent, else a non-negative float — never raises
    cost = llm.estimate_cost("gpt-4o", {"prompt_tokens": 1, "completion_tokens": 1})
    assert cost is None or cost >= 0


def test_chat_split_renders(logdir, monkeypatch):
    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.setenv("LINES", "30")
    a = App(make_reader(str(logdir)), TextRenderer())
    a.reader.poll()
    a._chat_open = True
    a._chat_sessions[0]["messages"] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "loss is falling", "text": "loss is falling",
         "applied": ["tags=*loss*"]}]
    frame = a._build_frame()
    plain = re.sub(r"\x1b\[[0-9;]*m", "", frame)
    assert "chat 1" in plain and "loss is falling" in plain      # chat pane present
    assert "train/loss" in plain                                  # dashboard present
    assert "│" in frame                                          # divider


def test_chat_esc_closes_never_quits(logdir):
    a = App(make_reader(str(logdir)), TextRenderer())
    a.reader.poll()
    a._chat_open = True
    # Esc in the chat closes the sidebar and returns None (NOT a quit)
    assert a._handle_chat_key(_FakeScreen(), None, "ESC") is None
    assert a._chat_open is False


def test_chat_arrow_scroll(logdir):
    a = App(make_reader(str(logdir)), TextRenderer())
    a.reader.poll()
    a._chat_open = True
    a._chat_session()["messages"] = [
        {"role": "assistant", "content": f"l{i}", "text": f"l{i}"} for i in range(40)]
    a._build_frame()
    assert a._chat_scroll == 0                              # starts at the bottom
    a._handle_chat_key(_FakeScreen(), None, "UP")
    a._handle_chat_key(_FakeScreen(), None, "UP")
    assert a._chat_scroll == 2                              # arrows scroll up
    a._handle_chat_key(_FakeScreen(), None, "DOWN")
    assert a._chat_scroll == 1
    # ^P recalls a sent message (no longer the up-arrow)
    a._chat_drafts = ["earlier question"]
    a._chat_draft_idx = 1
    a._handle_chat_key(_FakeScreen(), None, "\x10")
    assert "".join(a._chat_input) == "earlier question"


def test_chat_full_screen_toggle(logdir, monkeypatch):
    monkeypatch.setenv("COLUMNS", "100")
    monkeypatch.setenv("LINES", "20")
    a = App(make_reader(str(logdir)), TextRenderer())
    a.reader.poll()
    a._chat_open = True
    assert "│" in a._build_frame()                          # split: has a divider
    a._handle_chat_key(_FakeScreen(), None, "\x06")         # ^F -> full screen
    assert a._chat_full is True
    assert "│" not in a._build_frame()                      # full: no divider
    a._chat_command("/split", _FakeScreen(), None)
    assert a._chat_full is False


def test_chat_input_rich_editing(logdir):
    a = App(make_reader(str(logdir)), TextRenderer())
    a.reader.poll()
    a._chat_open = True
    for c in "hello world":
        a._handle_chat_key(_FakeScreen(), None, c)
    a._handle_chat_key(_FakeScreen(), None, "\x17")        # ^W: delete word
    assert "".join(a._chat_input) == "hello "
    a._handle_chat_key(_FakeScreen(), None, "\x15")        # ^U: clear
    assert a._chat_input == []


def test_chat_greeting_on_empty(logdir, monkeypatch):
    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.setenv("LINES", "30")
    a = App(make_reader(str(logdir)), TextRenderer())
    a.reader.poll()
    a._chat_open = True
    plain = re.sub(r"\x1b\[[0-9;]*m", "", a._build_frame())
    assert "Ask about your runs" in plain                  # cold-start greeting


def test_chat_dashboard_cached_while_typing(logdir, monkeypatch):
    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.setenv("LINES", "30")
    a = App(make_reader(str(logdir)), TextRenderer())
    a.reader.poll()
    a._chat_open = True
    a._build_frame()
    sig1 = a._dash_cache[0]
    a._handle_chat_key(_FakeScreen(), None, "z")           # typed, not a zoom
    a._build_frame()
    assert a._dash_cache[0] == sig1                        # dashboard not rebuilt
    assert "z" in "".join(a._chat_input)


def test_chat_sessions_and_commands(logdir):
    a = App(make_reader(str(logdir)), TextRenderer())
    a.reader.poll()
    assert len(a._chat_sessions) == 1
    a._chat_command("/new", _FakeScreen(), None)
    assert len(a._chat_sessions) == 2 and a._chat_active == 1
    a._chat_command("/rename experiments", _FakeScreen(), None)
    assert a._chat_session()["name"] == "experiments"
    a._chat_command("/prev", _FakeScreen(), None)
    assert a._chat_active == 0
    a._chat_command("/delete", _FakeScreen(), None)
    assert len(a._chat_sessions) == 1


def test_chat_sessions_persist(logdir, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    a = App(make_reader(str(logdir)), TextRenderer(), restore=True)
    a.reader.poll()
    a._chat_sessions = [{"name": "alpha", "messages": [
        {"role": "user", "content": "hi"}]},
        {"name": "beta", "messages": []}]
    a._chat_active = 1
    a._save_view()
    b = App(make_reader(str(logdir)), TextRenderer(), restore=True)
    assert [s["name"] for s in b._chat_sessions] == ["alpha", "beta"]
    assert b._chat_active == 1
    assert b._chat_sessions[0]["messages"][0]["content"] == "hi"


def test_chat_complete_uses_session_history_and_drives_view(logdir):
    from terminalboard import llm
    a = App(make_reader(str(logdir)), TextRenderer())
    a.reader.poll()
    a._llm_config = llm.LLMConfig("m")
    seen = {}

    def complete(**kw):                              # streaming fake (ask_stream)
        seen["messages"] = kw["messages"]
        return iter([
            {"choices": [{"delta": {"content": "done"}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"name": "open_hparams", "arguments": "{}"}}]}}]},
            {"choices": [{"delta": {}}],
             "usage": {"prompt_tokens": 1, "completion_tokens": 1}}])

    a._llm_complete = complete
    sess = a._chat_session()
    sess["messages"].append({"role": "user", "content": "open hparams"})
    text, applied, usage = a._chat_complete(sess, on_delta=lambda c: None)
    assert a._hparams is True                                # drove the dashboard
    assert sess["messages"][-1]["role"] == "assistant"
    assert sess["messages"][-1]["applied"] == ["open hparams"]
    # the system+history messages end with the user's prompt (no duplicate)
    assert seen["messages"][0]["role"] == "system"
    assert seen["messages"][-1] == {"role": "user", "content": "open hparams"}


def test_model_catalog_and_picker(logdir):
    from terminalboard import llm
    cat = llm.model_catalog()
    assert ("deepseek/deepseek-v4-flash", "DeepSeek") in cat   # curated, searchable
    assert len(cat) >= len(llm.CURATED_MODELS)
    if llm.is_available():       # litellm (the [llm] extra) adds its full list
        assert len(cat) > 100
    a = App(make_reader(str(logdir)), TextRenderer())
    a.reader.poll()
    # search ranks curated picks first, cheap-first
    rows = a._model_list("deepseek")
    vals = [v for _d, v, _p in rows]
    assert vals[0] == "deepseek/deepseek-v4-flash"
    assert "deepseek/deepseek-v4-pro" in vals
    # empty query shows curated; non-matching query still allows a custom entry
    assert a._model_list("")[0][1] == "gpt-5.4-nano"
    custom = a._model_list("my/local-model")
    assert custom[-1][1] == "my/local-model" and custom[-1][2] == "custom"


def test_model_picker_flow_sets_model(logdir, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    from terminalboard import llm
    monkeypatch.setattr(llm, "is_available", lambda: True)
    monkeypatch.setattr(llm, "validate", lambda cfg, complete=None: (True, ""))
    a = App(make_reader(str(logdir)), TextRenderer())
    a.reader.poll()

    class _Keys:
        def __init__(self, seq):
            self.q = list(seq)

        def get(self, _t):
            return self.q.pop(0) if self.q else None

    # type 'deepseek', Enter picks the top (v4-flash), type key, Enter saves
    keys = _Keys(list("deepseek") + ["\r"] + list("sk-X") + ["\r"])
    assert a._llm_setup(_FakeScreen(), keys) is True
    assert a._llm_config.model == "deepseek/deepseek-v4-flash"
    assert a._llm_config.api_key == "sk-X"


def test_llm_local_cost_map_enforced():
    # importing terminalboard.llm must pre-set the offline-cost-map switch so
    # litellm never fetches the pricing JSON from GitHub (privacy: the only
    # allowed traffic is to the user's chosen provider).
    import os
    import terminalboard.llm  # noqa: F401  (already imported; idempotent)
    assert os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP") == "true"


def test_llm_config_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    from terminalboard import llm
    assert llm.load_config() is None
    llm.save_config(llm.LLMConfig(model="gpt-4o", api_key="sk-x", api_base=""))
    got = llm.load_config()
    assert got.model == "gpt-4o" and got.api_key == "sk-x"
    import os
    assert oct(os.stat(llm.config_path()).st_mode)[-3:] == "600"


def test_config_load(tmp_path, monkeypatch):
    pytest.importorskip("tomllib")      # stdlib on 3.11+
    cfg = tmp_path / "c.toml"
    cfg.write_text('[terminalboard]\nsmooth = 0.9\ngrid = "3x3"\nlogy = true\n')
    monkeypatch.setenv("TERMINALBOARD_CONFIG", str(cfg))
    from terminalboard.cli import load_config
    c = load_config()
    assert c["smooth"] == 0.9 and c["grid"] == "3x3" and c["logy"] is True


def test_logy_and_xaxis_toggle(logdir):
    a = _app(logdir)
    a._handle_view_key("l")
    assert a.logy is True
    a._handle_view_key("x")
    assert a.xaxis == "time"
    assert a._build_frame().strip()                 # renders without error


def test_detail_q_does_not_quit_esc_goes_back(logdir):
    a = _app(logdir)
    a.tag_filter = "train/loss"
    a._handle_grid_key(_FakeScreen(), None, "\r")
    assert a._detail == "train/loss"
    assert a._handle_detail_key(_FakeScreen(), None, "q") is None      # q does nothing in detail
    assert a._detail == "train/loss"              # still in detail
    a._handle_detail_key(_FakeScreen(), None, "ESC")                   # Esc -> back to grid
    assert a._detail is None
    # and from the grid, Esc quits
    assert a._handle_grid_key(_FakeScreen(), None, "ESC") is True


# -- helpers -----------------------------------------------------------------

def test_shorten_tag():
    assert shorten_tag("train/loss", 50) == "train/loss"
    assert shorten_tag("a/b/very_long_leaf_name", 6).startswith(("v", "…"))


def test_ema_smoothing():
    assert ema([1.0, 1.0, 1.0], 0.5) == [1.0, 1.0, 1.0]
    assert ema([], 0.5) == []
    assert ema([0.0, 10.0], 0.0) == [0.0, 10.0]


# -- CLI ---------------------------------------------------------------------

def test_cli_once_default(logdir, capsys):
    from terminalboard.cli import main
    rc = main([str(logdir), "--once", "--tags", "loss"])
    assert rc == 0
    assert "loss" in capsys.readouterr().out


def test_cli_list(logdir, capsys):
    from terminalboard.cli import main
    rc = main([str(logdir), "--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "train/loss" in out and "note/info" in out and "weights/h" in out


def test_cli_missing_logdir():
    from terminalboard.cli import main
    assert main(["/no/such/dir", "--once"]) == 2
