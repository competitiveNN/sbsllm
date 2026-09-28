"""Regression tests for OpenAI-compatible streaming: termination + thinking.

The local chat used to hang because the streaming handler never exited its
poll loop early: the "content went idle" fallback was unreachable dead code
and the site-level `done` flag was pinned false by over-broad loading
selectors, so every request burned the full browser timeout and the SSE
connection was never closed.
"""

import json
import socket
import threading
import time
import types
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest

from sbsllm.server import (
    DEFAULT_RESPONSE_IDLE_TIMEOUT,
    OpenAIHandler,
    _ModelLockRegistry,
    create_server,
)


class FakeClock:
    """Virtual clock: time only advances when the code under test sleeps."""

    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += max(seconds, 0.01)


def _result(content="", *, thinking=None, done=False, busy=False, found=True, count=1):
    return {
        "found": found,
        "content": content,
        "thinking": thinking,
        "busy": busy,
        "done": done,
        "count": count,
    }


def _generating(content="", *, thinking=None, count=1):
    """A poll taken while the site reports it is still generating."""
    return _result(content, thinking=thinking, busy=True, done=False, count=count)


def _finished(content="", *, thinking=None, count=1):
    """A poll taken after the site stopped generating."""
    return _result(content, thinking=thinking, busy=False, done=True, count=count)


def _handler(
    idle_timeout=3.0,
    poll_interval=0.25,
    done_confirm=0.75,
    thinking_patience=120.0,
):
    handler = OpenAIHandler.__new__(OpenAIHandler)
    handler.server = types.SimpleNamespace(
        model_locks=_ModelLockRegistry(),
        browser_lock_timeout=10,
        browser_timeout=30.0,
        response_idle_timeout=idle_timeout,
        response_done_confirm=done_confirm,
        thinking_patience=thinking_patience,
        poll_interval=poll_interval,
    )
    handler.wfile = MagicMock()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.close_connection = False
    handler.headers: dict[str, str] = {}
    handler._inject_and_submit_with_recovery = MagicMock(
        return_value=({"inject": "OK", "submit": "OK"}, MagicMock())
    )
    handler._recover_page = MagicMock(return_value=None)
    return handler


def _stream(handler, polls, browser_timeout=30.0, loop_cap=20000):
    """Drive _handle_streaming_chat_completions over a scripted poll sequence.

    The last entry repeats forever. Returns (virtual_elapsed, events) where
    events is a list of (delta, finish_reason) plus the string "[DONE]".
    """
    clock = FakeClock()
    calls = {"n": 0}

    def capture(page, js):
        calls["n"] += 1
        assert calls["n"] < loop_cap, "streaming loop never terminated"
        return dict(polls[min(calls["n"] - 1, len(polls) - 1)])

    events = []
    with (
        patch("sbsllm.server.capture_response", side_effect=capture),
        patch("sbsllm.server.extract_js", return_value="EXTRACT"),
        patch("sbsllm.server.time") as fake_time,
    ):
        fake_time.monotonic.side_effect = clock.monotonic
        fake_time.sleep.side_effect = clock.sleep
        fake_time.time.return_value = 1_700_000_000

        def send_sse(data):
            events.append(data)

        handler._send_sse = send_sse
        start = clock.t
        handler._handle_streaming_chat_completions(
            "rid", "zai", "zai", MagicMock(), 1, "hi", 0.0, browser_timeout
        )
        elapsed = clock.t - start

    parsed = []
    for raw in events:
        if raw == "[DONE]":
            parsed.append(("[DONE]", None))
            continue
        choice = json.loads(raw)["choices"][0]
        parsed.append((choice["delta"], choice["finish_reason"]))
    return elapsed, parsed


# The reported zai shape: the answer arrives and settles, but the page keeps
# a stale "loading" element around so the site's own `done` never flips.
STABLE_ANSWER: list[dict] = [
    _result("", found=False, count=0),  # baseline before submit
    _result("", found=False, count=0),  # nothing started yet
    _result("Hello"),
    _result("Hello! I'm GLM"),
    _result("Hello! I'm GLM, trained by Z.ai."),
]


class TestStreamTerminates:
    def test_terminates_on_idle_window_not_browser_timeout(self):
        """The old loop ran to browser_timeout whenever `done` stayed false."""
        elapsed, events = _stream(_handler(idle_timeout=3.0), STABLE_ANSWER)

        # 4 polls to drain the sequence + 3s idle, far below the 30s ceiling.
        assert elapsed < 10.0, f"stream ran for {elapsed}s of a 30s timeout"
        assert events[-1] == ("[DONE]", None)
        assert events[-2] == ({}, "stop")

    def test_finish_reason_is_length_only_on_real_timeout(self):
        # Content keeps growing, so the idle window never opens and only the
        # hard browser_timeout can end the stream.
        def capture(page, js):
            capture.n += 1
            return _result("x" * capture.n)

        capture.n = 0
        clock = FakeClock()
        handler = _handler(idle_timeout=1.0)
        events = []
        with (
            patch("sbsllm.server.capture_response", side_effect=capture),
            patch("sbsllm.server.extract_js", return_value="EXTRACT"),
            patch("sbsllm.server.time") as fake_time,
        ):
            fake_time.monotonic.side_effect = clock.monotonic
            fake_time.sleep.side_effect = clock.sleep
            fake_time.time.return_value = 1_700_000_000
            handler._send_sse = events.append
            start = clock.t
            handler._handle_streaming_chat_completions(
                "rid", "zai", "zai", MagicMock(), 1, "hi", 0.0, 3.0
            )
            elapsed = clock.t - start

        assert 3.0 <= elapsed <= 4.0
        assert events[-1] == "[DONE]"
        assert json.loads(events[-2])["choices"][0]["finish_reason"] == "length"

    def test_done_alone_does_not_finish_before_generation_is_seen(self):
        """A site renders its stop control a beat after the first token, so
        `done` is briefly true while the answer is still arriving."""
        polls = [
            _result("", found=False, count=0),
            _result("half an ans", done=True),
            _result("half an answ", done=True),
            _result("half an answer", done=True),
            _finished("half an answer"),
        ]
        elapsed, events = _stream(_handler(idle_timeout=2.0), polls)
        # Falls back to the idle window rather than cutting on the first `done`.
        assert 1.5 <= elapsed <= 3.0, f"took {elapsed}s"
        assert events[-1] == ("[DONE]", None)
        text = "".join(d.get("content", "") for d, _ in events if isinstance(d, dict))
        assert text == "half an answer"

    def test_finishes_promptly_once_generation_stops(self):
        polls = [
            _result("", found=False, count=0),
            _generating("work"),
            _finished("work"),
        ]
        elapsed, events = _stream(_handler(idle_timeout=30.0, done_confirm=0.5), polls)
        # done_confirm, not the 30s idle window.
        assert elapsed < 3.0, f"took {elapsed}s"
        assert events[-2] == ({}, "stop")

    def test_emits_no_duplicate_content(self):
        _t, events = _stream(_handler(idle_timeout=1.0), STABLE_ANSWER)
        text = "".join(d.get("content", "") for d, _ in events if isinstance(d, dict))
        assert text == "Hello! I'm GLM, trained by Z.ai."

    def test_capture_failure_ends_the_stream(self):
        handler = _handler()
        with (
            patch(
                "sbsllm.server.capture_response",
                side_effect=[
                    _result("", found=False, count=0),
                    __import__(
                        "sbsllm.browser", fromlist=["BrowserError"]
                    ).BrowserError("page gone"),
                ],
            ),
            patch("sbsllm.server.extract_js", return_value="EXTRACT"),
            patch("sbsllm.server.get_page_snapshot", return_value={}),
        ):
            handler._handle_streaming_chat_completions(
                "rid", "zai", "zai", MagicMock(), 1, "hi", 0.0, 30.0
            )
        written = [c.args[0] for c in handler.wfile.write.call_args_list]
        assert any(b"[DONE]" in w for w in written), "stream left open after error"


class TestThinkingStreaming:
    THINKING: ClassVar[list[dict]] = [
        _result("", found=False, count=0),
        _result("", thinking="Let me check the greeting."),
        _result("", thinking="Let me check the greeting. Then answer."),
        _result("Hi there", thinking="Let me check the greeting. Then answer."),
    ]

    def test_thinking_is_streamed_before_any_content(self):
        """Thinking used to be dropped until the answer text appeared."""
        _t, events = _stream(_handler(idle_timeout=1.0), self.THINKING)
        thinking_deltas = [
            d["thinking"]
            for d, _ in events
            if isinstance(d, dict) and d.get("thinking")
        ]
        # Thinking is delivered as it grows, not one lump on the first chunk.
        assert thinking_deltas == [
            "Let me check the greeting.",
            " Then answer.",
        ]

    def test_first_chunk_announces_assistant_role(self):
        """A thinking-only first chunk must still carry role=assistant."""
        _t, events = _stream(_handler(idle_timeout=1.0), self.THINKING)
        content_deltas = [d for d, _ in events if isinstance(d, dict) and d]
        assert content_deltas[0].get("role") == "assistant"
        assert "content" not in content_deltas[0]

    def test_thinking_and_content_are_both_delivered(self):
        _t, events = _stream(_handler(idle_timeout=1.0), self.THINKING)
        text = "".join(d.get("content", "") for d, _ in events if isinstance(d, dict))
        thinking = "".join(
            d.get("thinking", "") for d, _ in events if isinstance(d, dict)
        )
        assert text == "Hi there"
        assert thinking == "Let me check the greeting. Then answer."


class TestThinkingStallPatience:
    """A thinking-only stall (content empty, reasoning present, site still
    generating) gets far more patience than an answer stall: reasoning traces
    pause naturally between chunks and thinking models think for minutes."""

    def _stall(self, *, busy=True, content="", thinking="still thinking"):
        return _result(content, thinking=thinking, busy=busy, done=False, count=1)

    def test_thinking_stall_uses_thinking_patience_not_busy_patience(self):
        """20s of unchanged reasoning while generating must NOT end the stream."""
        # 30s of unchanged thinking with busy_patience=20s would have cut it.
        polls = [
            _result("", found=False, count=0),
            self._stall(),
            self._stall(),
            self._stall(),
        ]
        elapsed, events = _stream(
            _handler(
                idle_timeout=3.0,
                thinking_patience=30.0,
            ),
            polls,
            browser_timeout=30.0,
        )
        # 3 polls * 0.25s + 30s thinking_patience
        assert 29.0 <= elapsed <= 31.0, f"took {elapsed}s"
        assert events[-1] == ("[DONE]", None)
        assert events[-2][1] == "length"

    def test_answer_still_uses_busy_patience(self):
        """A stall with content present must still respect busy_patience."""
        polls = [
            _result("", found=False, count=0),
            _result("Hi", thinking="done", busy=True, done=False, count=1),
            _result("Hi", thinking="done", busy=True, done=False, count=1),
            _result("Hi", thinking="done", busy=True, done=False, count=1),
        ]
        elapsed, events = _stream(
            _handler(
                idle_timeout=3.0,
                thinking_patience=30.0,
            ),
            polls,
            browser_timeout=30.0,
        )
        # busy_patience defaults to 20.0
        assert 19.0 <= elapsed <= 21.0, f"took {elapsed}s"
        assert events[-2][1] == "length"


class TestIsNewResponse:
    def _handler(self):
        return _handler()

    def test_thinking_growth_counts_as_a_new_turn(self):
        handler = self._handler()
        baseline = _result("", found=False, count=0, thinking=None)
        assert handler._is_new_response(_result("", thinking="hmm"), baseline) is True

    def test_unchanged_baseline_is_not_new(self):
        handler = self._handler()
        baseline = _result("old", count=1)
        assert handler._is_new_response(_result("old", count=1), baseline) is False

    def test_increased_count_is_new(self):
        handler = self._handler()
        baseline = _result("a", count=1)
        assert handler._is_new_response(_result("a", count=2), baseline) is True

    def test_changed_content_is_new(self):
        handler = self._handler()
        baseline = _result("a", count=1)
        assert handler._is_new_response(_result("ab", count=1), baseline) is True

    def test_no_baseline_accepts_first_answer(self):
        handler = self._handler()
        assert handler._is_new_response(_result("a"), None) is True
        assert handler._is_new_response(_result("", found=False), None) is False


class TestSSEHeaders:
    def test_streaming_response_closes_the_connection(self):
        """`Connection: keep-alive` on HTTP/1.0 left clients waiting forever."""
        handler = _handler()
        handler.protocol_version = "HTTP/1.0"
        handler._send_sse_headers("rid")

        headers = {c.args[0]: c.args[1] for c in handler.send_header.call_args_list}
        assert headers["Connection"] == "close"
        assert handler.close_connection is True

    def test_sse_events_are_flushed(self):
        handler = _handler()
        handler._send_sse("{}")
        handler.wfile.write.assert_called_once()
        assert handler.wfile.write.call_args.args[0] == b"data: {}\n\n"
        handler.wfile.flush.assert_called_once()


class TestRuntimeSettingsPublished:
    def test_start_publishes_settings_for_handlers(self):
        """Handlers read settings off self.server; missing ones silently fell
        back to defaults, ignoring the configured browser_timeout and never
        taking the browser lock."""
        server = create_server(
            model_map={"zai": "zai"},
            tab_map={"zai": object()},
            browser_timeout=12,
            browser_lock_timeout=7,
            response_idle_timeout=1.5,
        )
        mock_http_server = MagicMock()
        mock_http_server.handle_request.side_effect = KeyboardInterrupt()

        with patch("sbsllm.server._SBSHTTPServer", return_value=mock_http_server):
            server.start(announce=False)

        published = mock_http_server
        assert published.browser_timeout == 12
        assert published.browser_lock_timeout == 7
        assert published.response_idle_timeout == 1.5
        assert published.model_locks is server.model_locks
        assert published.tab_map is server.tab_map
        server.stop()

    def test_default_idle_timeout_is_short(self):
        assert DEFAULT_RESPONSE_IDLE_TIMEOUT == 3.0

    def test_handler_reads_published_settings(self):
        handler = _handler()
        assert handler._server_setting("browser_timeout", 1) == 30.0
        assert handler._server_setting("nope", "fallback") == "fallback"

    def test_handler_recover_page_updates_shared_tab_map(self):
        handler = OpenAIHandler.__new__(OpenAIHandler)
        handler.tab_map = {}
        handler.server = types.SimpleNamespace(tab_map={})
        new_page = object()

        with (
            patch("sbsllm.browser.recover_page", return_value=new_page),
            patch("sbsllm.server.get_site", return_value={"url": "https://x/"}),
        ):
            assert handler._recover_page("zai", "zai") is new_page

        assert handler.tab_map == {"zai": new_page}
        assert handler.server.tab_map == {}


class TestStreamEndsOverRealHTTP:
    """End-to-end: a client must see a closed connection, not a hung socket."""

    def test_connection_closes_after_done(self):
        server = create_server(
            model_map={"zai": "zai"},
            tab_map={"zai": MagicMock()},
            port=0,
            browser_timeout=30,
            health_interval=9999,
            response_idle_timeout=1.0,
        )
        polls = list(STABLE_ANSWER)

        def capture(page, js):
            return dict(polls[min(capture.n, len(polls) - 1)])

        capture.n = 0

        def counting_capture(page, js):
            capture.n += 1
            return capture(page, js)

        thread = threading.Thread(target=server.start, daemon=True)
        with (
            patch("sbsllm.server.capture_response", side_effect=counting_capture),
            patch("sbsllm.server.extract_js", return_value="EXTRACT"),
            patch("sbsllm.server.check_page_health", return_value=True),
            patch(
                "sbsllm.server.inject_and_submit",
                return_value={"inject": "OK", "submit": "OK"},
            ),
        ):
            thread.start()
            server.wait_until_ready(5)
            port = server._server.server_address[1]
            body = json.dumps(
                {
                    "model": "zai",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ).encode()
            sock = socket.create_connection(("127.0.0.1", port), timeout=10)
            sock.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
            )
            sock.settimeout(20)
            buf = b""
            closed = False
            start = time.monotonic()
            try:
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        closed = True
                        break
                    buf += chunk
            except TimeoutError:
                pass
            finally:
                sock.close()
                server.stop()

        assert closed, "server never closed the streaming connection"
        assert b"data: [DONE]" in buf
        assert b"Connection: close" in buf
        assert time.monotonic() - start < 15


class TestNoDeadImports:
    def test_server_exposes_helpers(self):
        for name in ("_stream_web_chat", "_sse_chunk", "_server_setting"):
            assert hasattr(OpenAIHandler, name)


@pytest.mark.parametrize(
    "attrs",
    [
        {"content": "answer", "done": True},
        {"content": "answer", "thinking": "trace", "done": True},
    ],
)
def test_result_helper_shape(attrs):
    assert set(_result(**attrs)) == {
        "found",
        "content",
        "thinking",
        "busy",
        "done",
        "count",
    }


class TestNoPrematureTruncation:
    """The local chat used to receive only part of the real answer."""

    def test_long_mid_generation_pause_does_not_cut_the_answer(self):
        """Web chats pause while thinking / re-rendering. An unchanged payload
        during a pause must not be read as 'finished'."""
        polls = [
            _result("", found=False, count=0),
            _generating("The answer begins"),
            _generating("The answer begins"),  # pause
            _generating("The answer begins"),  # pause
            _generating("The answer begins here"),  # pause
            _generating("The answer begins here and continues"),
            _finished("The answer begins here and continues"),
        ]
        elapsed, events = _stream(_handler(idle_timeout=1.0, done_confirm=0.5), polls)
        text = "".join(d.get("content", "") for d, _ in events if isinstance(d, dict))
        assert text == "The answer begins here and continues"
        assert elapsed < 5.0, f"took {elapsed}s"

    def test_slow_trickle_survives_a_generous_idle_window(self):
        polls = [_result("", found=False, count=0)]
        polls += [_generating("t" * i) for i in range(1, 30)]
        polls.append(_finished("t" * 29))
        elapsed, events = _stream(_handler(idle_timeout=1.0, done_confirm=0.5), polls)
        text = "".join(d.get("content", "") for d, _ in events if isinstance(d, dict))
        assert text == "t" * 29
        # 29 polls plus a short confirm, nowhere near the 1s idle window.
        assert elapsed < 8.0, f"took {elapsed}s"
        assert events[-2] == ({}, "stop")

    def test_flickering_stop_control_does_not_cut_the_answer(self):
        """The stop control can blink out mid-render; `done` must be
        confirmed before it ends the stream."""
        polls = [
            _result("", found=False, count=0),
            _generating("part one"),
            _finished("part one"),  # blink: not busy
            _generating("part one and two"),  # still going
            _generating("part one and two"),
            _finished("part one and two"),
        ]
        _t, events = _stream(_handler(idle_timeout=1.0, done_confirm=1.0), polls)
        text = "".join(d.get("content", "") for d, _ in events if isinstance(d, dict))
        assert text == "part one and two"

    def test_thinking_gap_does_not_truncate(self):
        polls = [
            _result("", found=False, count=0),
            _generating(thinking="thinking hard"),
            _generating(thinking="thinking hard"),
            _generating(thinking="thinking hard", content="Answer"),
            _finished(thinking="thinking hard", content="Answer"),
        ]
        _t, events = _stream(_handler(idle_timeout=1.0, done_confirm=0.5), polls)
        thinking = "".join(
            d.get("thinking", "") for d, _ in events if isinstance(d, dict)
        )
        text = "".join(d.get("content", "") for d, _ in events if isinstance(d, dict))
        assert thinking == "thinking hard"
        assert text == "Answer"

    def test_idle_fallback_still_terminates_when_site_never_signals(self):
        polls = [
            _result("", found=False, count=0),
            _result("quiet site answer"),
            _result("quiet site answer"),
        ]
        elapsed, events = _stream(_handler(idle_timeout=1.0), polls)
        assert elapsed < 4.0
        assert events[-2] == ({}, "stop")


class TestNoSilentHang:
    """A request must always end, and say why.

    These guards exist because a silent stream is indistinguishable from a
    hang in the local chat, and gave nothing to debug with.
    """

    def _handler(self, **kw):
        kw.setdefault("first_token_timeout", 5.0)
        kw.setdefault("busy_patience", 3.0)
        handler = _handler()
        for k, v in kw.items():
            setattr(handler.server, k, v)
        return handler

    def test_no_output_terminates_with_a_reason(self):
        polls = [_result("", found=False, count=0)]
        _elapsed, events = _stream(
            self._handler(first_token_timeout=2.0), polls, browser_timeout=60
        )
        text = "".join(d.get("content", "") for d, _ in events if isinstance(d, dict))
        assert "produced no answer" in text, text
        assert events[-1] == ("[DONE]", None)

    def test_no_output_does_not_silence_the_client(self):
        """Keepalives keep the connection visibly alive while we wait."""
        polls = [_result("", found=False, count=0)] * 200
        handler = self._handler(first_token_timeout=30.0)
        handler.server.keepalive_interval = 0.5
        sent = []
        clock = FakeClock()
        handler._send_sse_comment = lambda: sent.append(1)
        with (
            patch(
                "sbsllm.server.capture_response",
                side_effect=lambda p, j: dict(polls[0]),
            ),
            patch("sbsllm.server.extract_js", return_value="EXTRACT"),
            patch("sbsllm.server.time") as fake_time,
        ):
            fake_time.monotonic.side_effect = clock.monotonic
            fake_time.sleep.side_effect = clock.sleep
            fake_time.time.return_value = 1_700_000_000
            handler._send_sse = lambda d: None
            handler._handle_streaming_chat_completions(
                "rid", "zai", "zai", MagicMock(), 1, "hi", 0.0, 60.0
            )
        assert len(sent) >= 3, f"only {len(sent)} keepalives"

    def test_busy_patience_terminates_a_stuck_spinner(self):
        """A site that never stops claiming to generate must not hold the tab
        until the whole budget expires."""
        polls = [
            _result("", found=False, count=0),
            _generating("partial answer"),
        ]
        elapsed, events = _stream(
            self._handler(busy_patience=2.0), polls, browser_timeout=60
        )
        assert elapsed < 20.0, f"waited {elapsed}s - patience guard did not fire"
        text = "".join(d.get("content", "") for d, _ in events if isinstance(d, dict))
        assert text == "partial answer"

    def test_budget_exhaustion_always_terminates(self):
        """Even content that keeps changing must stop at the budget."""

        def capture(page, js):
            capture.n += 1
            return _generating("x" * capture.n)

        capture.n = 0
        clock = FakeClock()
        handler = self._handler()
        with (
            patch("sbsllm.server.capture_response", side_effect=capture),
            patch("sbsllm.server.extract_js", return_value="EXTRACT"),
            patch("sbsllm.server.time") as fake_time,
        ):
            fake_time.monotonic.side_effect = clock.monotonic
            fake_time.sleep.side_effect = clock.sleep
            fake_time.time.return_value = 1_700_000_000
            written = []
            handler._send_sse = written.append
            start = clock.t
            handler._handle_streaming_chat_completions(
                "rid", "zai", "zai", MagicMock(), 1, "hi", 0.0, 10.0
            )
            elapsed = clock.t - start
        assert elapsed <= 12.0
        assert written[-1] == "[DONE]"

    def test_stop_reason_is_logged_and_reported(self):
        polls = [
            _result("", found=False, count=0),
            _generating("x"),
            _finished("x"),
        ]
        _elapsed, events = _stream(self._handler(), polls)
        assert events[-2] == ({}, "stop")

    def test_busy_patience_does_not_fire_while_text_still_moving(self):
        polls = [_result("", found=False, count=0)]
        polls += [_generating("t" * i) for i in range(1, 40)]
        polls.append(_finished("t" * 39))
        elapsed, events = _stream(
            self._handler(busy_patience=3.0), polls, browser_timeout=60
        )
        text = "".join(d.get("content", "") for d, _ in events if isinstance(d, dict))
        assert text == "t" * 39, "busy patience truncated a live answer"
        assert elapsed < 30.0
