#!/usr/bin/env python3
"""
aloud — a local, offline text-to-speech reader for the Mac.

Reads web articles, PDF sections, clipboard text, or literal strings aloud
using Kokoro TTS (mlx-community/Kokoro-82M-bf16) via MLX-Audio on Apple Silicon.

Everything runs locally; after the first model download it works fully offline.
"""

import argparse
import bisect
import contextlib
import logging
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import threading
import wave
from dataclasses import dataclass

import numpy as np

# Quiet the noisy third-party startup chatter before anything imports them.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# espeak's phonemizer emits frequent harmless "words count mismatch" warnings
# on the espeak-ng fallback path. It resets its own logger level every time a
# backend is created, so a plain setLevel() gets clobbered — attach a filter
# (filters survive setLevel/handler resets) that drops anything below ERROR.
class _DropPhonemizerNoise(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.ERROR


logging.getLogger("phonemizer").addFilter(_DropPhonemizerNoise())

MODEL_ID = "mlx-community/Kokoro-82M-bf16"
DEFAULT_VOICE = "af_bella"
MAX_CHARS = 500          # target characters per synthesis chunk
FIRST_CHARS = 120        # tiny first chunk so audio starts fast
SEEK_BIG = 10            # j / l jump (seconds), YouTube-style
SEEK_SMALL = 5           # ← / → jump (seconds), YouTube-style
BLOCK = 2048             # audio frames written per loop (~85 ms at 24 kHz)
REFRESH = 0.1            # read-along redraw interval (seconds)
WORDS_PER_SEC = 2.5      # rough Kokoro speaking rate at speed 1.0, for estimates

# Language code is derived from the first letter of the voice name.
#   a=American English  b=British English  e=Spanish  f=French  h=Hindi
#   i=Italian  j=Japanese  p=Portuguese  z=Mandarin Chinese
# This reader is tuned for English (a/b); other languages need extra G2P deps.


def err(msg):
    """Print a clean one-line error and exit (no stack trace)."""
    print(f"aloud: {msg}", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Text sources
# --------------------------------------------------------------------------- #
def text_from_url(url):
    try:
        import trafilatura
    except ImportError:
        err("trafilatura is not installed (run: uv sync)")
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        err(f"could not fetch URL: {url}")
    # Keep light markdown (headings, lists, emphasis) so we can show structure
    # on screen; it's stripped back to clean text before synthesis.
    text = trafilatura.extract(
        downloaded, include_comments=False, include_tables=False,
        include_formatting=True,
    )
    if not text or not text.strip():
        err(f"no readable article text found at: {url}")
    return text


def parse_pages(spec):
    """'3-7' -> (3, 7); '5' -> (5, 5). 1-indexed, inclusive."""
    spec = spec.strip()
    m = re.fullmatch(r"(\d+)(?:-(\d+))?", spec)
    if not m:
        err(f"invalid --pages value: {spec!r} (use e.g. 3-7 or 5)")
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else start
    if start < 1 or end < start:
        err(f"invalid page range: {spec!r}")
    return start, end


def text_from_pdf(path, pages=None):
    try:
        from pypdf import PdfReader
    except ImportError:
        err("pypdf is not installed (run: uv sync)")
    try:
        reader = PdfReader(path)
    except Exception as e:
        err(f"could not open PDF {path}: {e}")
    total = len(reader.pages)
    if pages:
        start, end = pages
        if start > total:
            err(f"--pages {start}-{end} but PDF has only {total} pages")
        end = min(end, total)
        selected = range(start - 1, end)
    else:
        selected = range(total)
    parts = []
    for i in selected:
        try:
            parts.append(reader.pages[i].extract_text() or "")
        except Exception:
            parts.append("")
    text = "\n\n".join(p.strip() for p in parts if p.strip())
    if not text.strip():
        err(f"no extractable text in PDF (it may be scanned images): {path}")
    return text


def text_from_clipboard():
    try:
        out = subprocess.run(["pbpaste"], capture_output=True, text=True, check=True)
    except Exception as e:
        err(f"could not read clipboard via pbpaste: {e}")
    if not out.stdout.strip():
        err("clipboard is empty")
    return out.stdout


# --------------------------------------------------------------------------- #
# Chunking into typed segments
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    """One unit of synthesis. `text` is the clean string we read aloud; `kind`
    and `level` drive how it's rendered on screen as it plays."""
    text: str
    kind: str = "para"      # "para" | "heading" | "list"
    level: int = 0          # heading depth (1=#, 2=##, …)


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(paragraph):
    return [s for s in _SENT_SPLIT.split(paragraph) if s.strip()]


def strip_inline(s):
    """Remove markdown inline markers so they aren't spoken or shown literally."""
    s = re.sub(r"`+([^`]*)`+", r"\1", s)               # `code`
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)      # [text](url) -> text
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)            # **bold**
    s = re.sub(r"(?<!\*)\*(?!\*)([^*]+)\*(?!\*)", r"\1", s)  # *italic*
    s = re.sub(r"__([^_]+)__", r"\1", s)                # __bold__
    return " ".join(s.split())


def hard_wrap(sentence, limit):
    """Break an over-long sentence on commas/clauses, then on whitespace."""
    if len(sentence) <= limit:
        return [sentence]
    pieces, buf = [], ""
    for part in re.split(r"(?<=[,;:])\s+", sentence):
        if len(buf) + len(part) + 1 <= limit:
            buf = f"{buf} {part}".strip()
        else:
            if buf:
                pieces.append(buf)
            if len(part) <= limit:
                buf = part
            else:  # still too long: split on whitespace
                words, line = part.split(), ""
                for w in words:
                    if len(line) + len(w) + 1 <= limit:
                        line = f"{line} {w}".strip()
                    else:
                        pieces.append(line)
                        line = w
                buf = line
    if buf:
        pieces.append(buf)
    return pieces


def split_paragraph(para, limit=MAX_CHARS):
    """Greedily merge sentences of one paragraph into <=limit-char pieces,
    never breaking mid-sentence (over-long sentences are hard-wrapped)."""
    pieces, buf = [], ""
    for sent in split_sentences(para):
        for piece in hard_wrap(sent, limit):
            if not buf:
                buf = piece
            elif len(buf) + len(piece) + 1 <= limit:
                buf = f"{buf} {piece}"
            else:
                pieces.append(buf)
                buf = piece
    if buf:
        pieces.append(buf)
    return pieces


def segment_text(text, is_markdown, limit=MAX_CHARS):
    """Turn extracted text into a list of typed Segments.

    Markdown input (from web pages) keeps heading/list structure for display;
    plain input (PDF/clipboard/--text) becomes paragraphs. In both cases the
    spoken text has markdown markers stripped and long paragraphs are split.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    segments = []

    def add_paragraph(para):
        clean = strip_inline(" ".join(para.split()))
        for piece in split_paragraph(clean, limit):
            segments.append(Segment(piece, "para"))

    if not is_markdown:
        for para in re.split(r"\n\s*\n+", text):
            if para.strip():
                add_paragraph(para)
        return segments

    in_fence = False
    for raw in text.split("\n"):
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            # Code blocks read terribly aloud; skip the body but leave a short
            # spoken/shown notice so nothing disappears without the listener
            # knowing it was there.
            if not in_fence:
                segments.append(Segment("Code block omitted.", "code"))
            in_fence = not in_fence
            continue
        if not stripped or in_fence:
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if m:
            heading = strip_inline(m.group(2))
            if heading:
                segments.append(Segment(heading, "heading", len(m.group(1))))
            continue
        m = re.match(r"^[-*+]\s+(.*)", stripped)
        if m:
            item = strip_inline(m.group(1))
            if not item:
                continue
            # a long bullet still gets split, but every piece stays a "list"
            for piece in split_paragraph(item, limit):
                segments.append(Segment(piece, "list"))
            continue
        add_paragraph(stripped)
    return segments


def shrink_first_segment(segments):
    """Split a short lead off the first paragraph so the very first synthesis
    call is tiny and audio starts as soon as possible."""
    if segments and segments[0].kind == "para" and len(segments[0].text) > FIRST_CHARS:
        head, *rest = split_paragraph(segments[0].text, FIRST_CHARS)
        tail = split_paragraph(" ".join(rest), MAX_CHARS) if rest else []
        segments[0:1] = [Segment(p, "para") for p in (head, *tail)]
    return segments


# --------------------------------------------------------------------------- #
# Synthesis & audio
# --------------------------------------------------------------------------- #
def start_model_load(voice, speed, lang_code):
    """Load and warm up the model in a background thread so it overlaps
    fetching and segmenting the text. Returns a function that joins and
    yields the ready model.

    The warm-up generate also forces the lazily-loaded weights concrete;
    MLX lazy arrays can only be evaluated on the thread that created them,
    so without it the synth worker thread would crash (mlx#3529)."""
    print("Loading Kokoro model (first run downloads ~330 MB, then it's offline)…",
          flush=True)
    box = {}

    def work():
        try:
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                from mlx_audio.tts.utils import load_model
                model = load_model(MODEL_ID)
                list(model.generate(text="ready", voice=voice, speed=speed,
                                    lang_code=lang_code))
                box["model"] = model
        except Exception as e:
            box["error"] = e

    thread = threading.Thread(target=work, daemon=True)
    thread.start()

    def get_model():
        thread.join()
        if "error" in box:
            err(f"could not load model {MODEL_ID}: {box['error']}")
        return box["model"]

    return get_model


def synth(model, text, voice, speed, lang_code):
    """Return (mono float32 numpy audio, sample_rate) for one chunk."""
    results = list(model.generate(text=text, voice=voice, speed=speed,
                                  lang_code=lang_code))
    if not results:
        return None, None
    audio = np.concatenate([np.asarray(r.audio).reshape(-1) for r in results])
    return audio, results[0].sample_rate


def write_wav(path, audio, sr):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------- #
# Streaming playback with interactive controls
# --------------------------------------------------------------------------- #
class AudioBuffer:
    """A growing in-memory stream of synthesized audio.

    The synth worker marks where each segment (paragraph) begins and appends
    audio pieces as the model yields them; the playback thread reads arbitrary
    sample windows (for seeking). All audio lives in RAM — a 15-minute article
    is well under 100 MB at 24 kHz mono float32 — so seeking backward is always
    instant and playback never touches disk.
    """

    def __init__(self):
        self.parts = []        # list[np.ndarray], in append order
        self.starts = []       # cumulative start sample of each part
        self.seg_starts = []   # start sample of each segment (paragraph)
        self.total = 0         # samples synthesized so far
        self.done = False      # all segments synthesized
        self.lock = threading.Lock()

    def begin_segment(self):
        with self.lock:
            self.seg_starts.append(self.total)

    def append(self, arr):
        with self.lock:
            self.starts.append(self.total)
            self.parts.append(arr)
            self.total += len(arr)

    def mark_done(self):
        with self.lock:
            self.done = True

    def synth_total(self):
        with self.lock:
            return self.total

    def num_segments(self):
        with self.lock:
            return len(self.seg_starts)

    def segment_at(self, pos):
        with self.lock:
            if not self.seg_starts:
                return 0
            i = bisect.bisect_right(self.seg_starts, pos) - 1
            return max(0, min(i, len(self.seg_starts) - 1))

    def segment_start(self, idx):
        with self.lock:
            if 0 <= idx < len(self.seg_starts):
                return self.seg_starts[idx]
            return 0

    def segment_length(self, idx):
        """Samples in segment idx so far (grows while it's still streaming)."""
        with self.lock:
            if not 0 <= idx < len(self.seg_starts):
                return 0
            end = (self.seg_starts[idx + 1] if idx + 1 < len(self.seg_starts)
                   else self.total)
            return end - self.seg_starts[idx]

    def get_block(self, pos, n):
        """Return (block, at_end). block is None when pos is past the
        synthesized region; at_end is True only once everything is done."""
        with self.lock:
            if pos >= self.total:
                return None, self.done
            out, need, p = [], n, pos
            i = bisect.bisect_right(self.starts, p) - 1
            while need > 0 and 0 <= i < len(self.parts):
                local = p - self.starts[i]
                part = self.parts[i]
                if 0 <= local < len(part):
                    take = part[local:local + need]
                    out.append(take)
                    need -= len(take)
                    p += len(take)
                i += 1
            if not out:
                return None, self.done
            return (out[0] if len(out) == 1 else np.concatenate(out)), False


def _fmt_time(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class Player:
    """Synthesize chunks in a worker thread and play them through sounddevice,
    with a cursor we control for pause, seeking, and paragraph jumps. The
    now-playing paragraph is redrawn in place with a moving word highlight.

    A single Ctrl-C (or 'q') sets the quit flag; the audio stream is stopped and
    every thread is a daemon, so nothing is left running on exit.
    """

    HELP = ("[space/k] play/pause   [j/l] ∓10s   [←/→] ∓5s   "
            "[ [ / ] ] prev/next ¶   [0] restart   [q] quit")

    def __init__(self, model, chunks, voice, speed, lang_code, est_total_sec):
        self.model = model
        self.chunks = chunks
        self.voice = voice
        self.speed = speed
        self.lang_code = lang_code
        self.est_total_sec = est_total_sec
        self.sr = 24000
        self.buf = AudioBuffer()
        self.cursor = 0
        self.cursor_lock = threading.Lock()
        self.paused = False
        self.at_end = False
        self.quit = threading.Event()
        self.started = threading.Event()   # first chunk ready (or nothing to do)
        self.out_lock = threading.Lock()   # serialize all terminal writes
        self.interactive = sys.stdin.isatty() and sys.stdout.isatty()
        self._above = 0                    # lines above cursor in the live region
        self._para_lines = 0               # current paragraph line count

    # --- cursor helpers ----------------------------------------------------- #
    def _get_cursor(self):
        with self.cursor_lock:
            return self.cursor

    def _set_cursor(self, pos):
        with self.cursor_lock:
            self.cursor = max(0, pos)

    def _seek_to(self, pos):
        end = max(0, self.buf.synth_total() - 1)
        self._set_cursor(min(max(0, pos), end))
        self.at_end = False
        self.paused = False

    # --- worker threads ----------------------------------------------------- #
    def _synth_worker(self):
        for i, seg in enumerate(self.chunks):
            if self.quit.is_set():
                break
            self.buf.begin_segment()
            # Append pieces as the model yields them so playback can start
            # before the whole segment is synthesized.
            for r in self.model.generate(text=seg.text, voice=self.voice,
                                         speed=self.speed,
                                         lang_code=self.lang_code):
                if self.quit.is_set():
                    break
                audio = np.asarray(r.audio, dtype=np.float32).reshape(-1)
                if len(audio):
                    self.sr = r.sample_rate
                    self.buf.append(audio)
                    self.started.set()
            if self.buf.segment_length(i) == 0:
                # keep every segment non-empty so paragraph tracking works
                self.buf.append(np.zeros(int(0.1 * self.sr), dtype=np.float32))
        self.buf.mark_done()
        self.started.set()

    def _playback(self):
        import sounddevice as sd
        self.started.wait()
        silence = np.zeros(BLOCK, dtype=np.float32)
        try:
            stream = sd.OutputStream(samplerate=self.sr, channels=1,
                                     dtype="float32", blocksize=BLOCK)
            stream.start()
        except Exception as e:
            self.quit.set()
            self._log(f"audio output failed: {e}")
            return
        try:
            while not self.quit.is_set():
                if self.paused:
                    stream.write(silence)
                    continue
                block, at_end = self.buf.get_block(self._get_cursor(), BLOCK)
                if block is None:
                    if at_end:
                        self.at_end = True
                        self.paused = True
                    else:
                        stream.write(silence)   # underrun: wait for synth
                    continue
                stream.write(block)
                with self.cursor_lock:
                    self.cursor += len(block)
        finally:
            with contextlib.suppress(Exception):
                stream.stop()
                stream.close()

    # --- key handling (YouTube-style) --------------------------------------- #
    def _handle_key(self, key):
        if key in (" ", "k"):                     # play / pause
            self.paused = not self.paused
            self.at_end = False
        elif key in ("q", "Q"):                   # quit
            self.quit.set()
        elif key == "l":                          # forward 10s
            self._seek_to(self._get_cursor() + SEEK_BIG * self.sr)
        elif key == "j":                          # back 10s
            self._seek_to(self._get_cursor() - SEEK_BIG * self.sr)
        elif key == "RIGHT":                      # forward 5s
            self._seek_to(self._get_cursor() + SEEK_SMALL * self.sr)
        elif key == "LEFT":                       # back 5s
            self._seek_to(self._get_cursor() - SEEK_SMALL * self.sr)
        elif key == "]":                          # next paragraph
            self._jump_paragraph(+1)
        elif key == "[":                          # previous paragraph
            self._jump_paragraph(-1)
        elif key == "0":                          # restart
            self._seek_to(0)

    def _jump_paragraph(self, direction):
        idx = self.buf.segment_at(self._get_cursor()) + direction
        idx = max(0, min(idx, self.buf.num_segments() - 1))
        self._seek_to(self.buf.segment_start(idx))

    def _read_key(self, fd):
        if not select.select([fd], [], [], 0.2)[0]:
            return None
        ch = os.read(fd, 1)
        if ch == b"\x1b":                         # escape: maybe an arrow key
            if not select.select([fd], [], [], 0.05)[0]:
                return "ESC"
            seq = os.read(fd, 2)
            return {b"[C": "RIGHT", b"[D": "LEFT",
                    b"[A": "UP", b"[B": "DOWN"}.get(seq)
        return ch.decode("utf-8", "ignore") or None

    def _keyboard(self, fd):
        while not self.quit.is_set():
            key = self._read_key(fd)
            if key:
                self._handle_key(key)

    # --- rendering ---------------------------------------------------------- #
    def _status_line(self):
        pos = self._get_cursor()
        cur = pos / self.sr
        if self.buf.done:
            total = self.buf.synth_total() / self.sr
        else:
            total = max(self.est_total_sec, cur)
        frac = 0.0 if total <= 0 else max(0.0, min(1.0, cur / total))
        fill = int(frac * 24)
        bar = "█" * fill + "░" * (24 - fill)
        state = "End    " if self.at_end else ("Paused " if self.paused
                                               else "Playing")
        idx = self.buf.segment_at(pos) + 1
        tail = "" if self.buf.done else " +"   # synthesis still running
        return (f"{'❚❚' if self.paused else '▶'}  {state}  "
                f"¶ {idx}/{len(self.chunks)}  {bar}  "
                f"{_fmt_time(cur)} / {_fmt_time(total)}{tail}")

    @staticmethod
    def _term_width():
        return min(shutil.get_terminal_size((80, 24)).columns, 100)

    @staticmethod
    def _wrap_words(words, width, hl, indent=""):
        """Greedy word-wrap to `width`, reverse-video the word at index `hl`.
        ANSI codes are added only after width is measured, so they don't
        affect wrapping."""
        lines, line, vis, first = [], indent, len(indent), True
        for i, w in enumerate(words):
            if not first and vis + 1 + len(w) > width:
                lines.append(line)
                line, vis, first = indent, len(indent), True
            shown = f"\x1b[7m{w}\x1b[0m" if i == hl else w
            line += shown if first else " " + shown
            vis += len(w) if first else 1 + len(w)
            first = False
        lines.append(line)
        return lines

    def _segment_lines(self, seg, hl):
        """Render a segment to a list of terminal lines (with styling)."""
        width = self._term_width()
        if seg.kind == "code":
            return ["\x1b[2m[code block omitted]\x1b[0m"]
        if seg.kind == "heading":
            lines = self._wrap_words(seg.text.split(), width, None)
            return [f"\x1b[1;36m{l}\x1b[0m" for l in lines]      # bold cyan
        if seg.kind == "list":
            wl = self._wrap_words(seg.text.split(), max(20, width - 4), None)
            return [(f"  \x1b[33m•\x1b[0m {l}" if j == 0 else f"    {l}")
                    for j, l in enumerate(wl)]
        return self._wrap_words(seg.text.split(), width, hl)   # paragraph

    def _current_word(self, idx):
        """Estimate which word index is playing, by elapsed fraction of the
        paragraph's audio (Kokoro gives no real per-word timestamps)."""
        seg = self.chunks[idx]
        nwords = len(seg.text.split())
        if nwords == 0 or seg.kind != "para":
            return None
        length = self.buf.segment_length(idx)
        if length <= 0:
            return 0
        frac = (self._get_cursor() - self.buf.segment_start(idx)) / length
        return max(0, min(nwords - 1, int(frac * nwords)))

    def _paint(self, idx):
        """Redraw the live region (current paragraph + status) in place."""
        para = self._segment_lines(self.chunks[idx], self._current_word(idx))
        lines = para + [self._status_line()]
        with self.out_lock:
            if self._above:
                sys.stdout.write(f"\x1b[{self._above}A")  # up to region top
            if len(para) != self._para_lines:             # layout changed
                sys.stdout.write("\x1b[J")                # full clear
            body = "".join(f"\r\x1b[K{l}\n" for l in lines[:-1])
            sys.stdout.write(body + f"\r\x1b[K{lines[-1]}")
            sys.stdout.flush()
        self._above = len(para)
        self._para_lines = len(para)

    def _commit(self, idx):
        """Leave a finished paragraph cleanly in the scrollback (no highlight,
        no frozen status line) before the next one becomes live."""
        para = self._segment_lines(self.chunks[idx], None)
        with self.out_lock:
            if self._above:
                sys.stdout.write(f"\x1b[{self._above}A")
            sys.stdout.write("\r\x1b[J")
            sys.stdout.write("".join(f"{l}\n" for l in para))
            sys.stdout.flush()
        self._above = 0
        self._para_lines = 0

    def _log(self, text):
        with self.out_lock:
            sys.stdout.write("\r\x1b[K" + text + "\n")
            sys.stdout.flush()

    # --- run ---------------------------------------------------------------- #
    def run(self):
        # A single Ctrl-C just sets the quit flag; every loop below checks it
        # and tears down. (More reliable than catching KeyboardInterrupt out of
        # a blocking Event.wait, which can be missed.)
        prev_sigint = signal.signal(signal.SIGINT, lambda *_: self.quit.set())
        worker = threading.Thread(target=self._synth_worker, daemon=True)
        play = threading.Thread(target=self._playback, daemon=True)
        worker.start()
        print("Synthesizing… audio starts in a moment.")
        # Wait for the first chunk (or empty finish), staying responsive to quit.
        while not self.started.is_set() and not self.quit.is_set():
            self.started.wait(0.1)
        try:
            if self.quit.is_set():
                return
            play.start()
            if self.interactive:
                self._run_interactive()
            else:
                self._run_plain()
        finally:
            self.quit.set()
            signal.signal(signal.SIGINT, prev_sigint)
            play.join(timeout=2)
            worker.join(timeout=2)

    def _run_interactive(self):
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        print(self.HELP)
        last_idx = -1
        try:
            tty.setcbreak(fd)            # cbreak keeps Ctrl-C working as SIGINT
            kb = threading.Thread(target=self._keyboard, args=(fd,), daemon=True)
            kb.start()
            while not self.quit.is_set():
                idx = self.buf.segment_at(self._get_cursor())
                if idx != last_idx:
                    if last_idx != -1:
                        self._commit(last_idx)   # finished paragraph -> scrollback
                    last_idx = idx
                self._paint(idx)                 # live region with word highlight
                self.quit.wait(REFRESH)
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, old)
            if last_idx != -1:
                self._commit(last_idx)           # leave last paragraph clean
            with self.out_lock:
                sys.stdout.flush()

    def _run_plain(self):
        # No TTY (piped/redirected): just stream to the end with chunk progress.
        last_idx = -1
        while not self.quit.is_set():
            idx = self.buf.segment_at(self._get_cursor())
            if idx != last_idx:
                last_idx = idx
                print(f"chunk {idx + 1}/{len(self.chunks)}", flush=True)
            if self.at_end:
                print("Done.", flush=True)
                break
            self.quit.wait(0.2)


def save_to_file(model, segments, voice, speed, lang_code, out_path):
    n = len(segments)
    pieces, sr = [], None
    try:
        for i, seg in enumerate(segments):
            print(f"\r♪ synthesizing chunk {i + 1}/{n}", end="", flush=True)
            audio, csr = synth(model, seg.text, voice, speed, lang_code)
            if audio is not None:
                pieces.append(audio)
                sr = csr
    except KeyboardInterrupt:
        print("\nStopping…")
        sys.exit(130)
    print()
    if not pieces:
        err("nothing was synthesized")
    write_wav(out_path, np.concatenate(pieces), sr)
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------- #
# Voices
# --------------------------------------------------------------------------- #
def list_voices():
    import glob
    base = os.path.expanduser(
        "~/.cache/huggingface/hub/"
        "models--mlx-community--Kokoro-82M-bf16/snapshots/*/voices/*"
    )
    found = sorted({os.path.splitext(os.path.basename(p))[0]
                    for p in glob.glob(base)})
    if not found:
        # Fallback to the known v1.0 voice list if the model isn't downloaded yet.
        found = [
            "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
            "am_michael", "am_onyx", "am_puck", "am_santa",
            "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        ]
    print("Available voices (prefix: a=US, b=UK female=f male=m):\n")
    groups = {"af": "US female", "am": "US male",
              "bf": "UK female", "bm": "UK male"}
    for pfx, label in groups.items():
        names = [v for v in found if v.startswith(pfx)]
        if names:
            print(f"  {label:9} {' '.join(names)}")
    other = [v for v in found if v[:2] not in groups]
    if other:
        print(f"  other     {' '.join(other)}")
    print(f"\nDefault: {DEFAULT_VOICE}   Recommended alt: af_heart")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        prog="aloud",
        description="Local offline TTS reader (Kokoro via MLX-Audio).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  aloud https://example.com/article\n"
            "  aloud paper.pdf --pages 3-7\n"
            "  aloud --text \"hello world\" --voice af_heart\n"
            "  aloud --clipboard --speed 1.2\n"
            "  aloud report.pdf --save out.wav\n"
            "  aloud --list-voices\n"
        ),
    )
    p.add_argument("source", nargs="?", help="a URL or a .pdf file path")
    p.add_argument("--text", help="read this literal string")
    p.add_argument("--clipboard", action="store_true",
                   help="read clipboard contents (pbpaste)")
    p.add_argument("--voice", default=DEFAULT_VOICE,
                   help=f"voice name (default: {DEFAULT_VOICE})")
    p.add_argument("--speed", type=float, default=1.0,
                   help="speech speed (default: 1.0)")
    p.add_argument("--pages", help="PDF page range, e.g. 3-7 or 5")
    p.add_argument("--save", metavar="FILE.wav",
                   help="save audio to a WAV file instead of playing")
    p.add_argument("--list-voices", action="store_true",
                   help="list available voices and exit")
    return p


def resolve_text(args):
    sources = sum(bool(x) for x in (args.source, args.text, args.clipboard))
    if sources == 0:
        err("nothing to read — give a URL/PDF, --text, or --clipboard "
            "(see: aloud --help)")
    if sources > 1:
        err("choose only one of: URL/PDF, --text, --clipboard")

    if args.text:
        if args.pages:
            err("--pages only applies to PDF files")
        return args.text, False
    if args.clipboard:
        if args.pages:
            err("--pages only applies to PDF files")
        return text_from_clipboard(), False

    src = args.source
    if src.startswith(("http://", "https://")):
        if args.pages:
            err("--pages only applies to PDF files")
        return text_from_url(src), True   # web extraction keeps markdown
    if src.lower().endswith(".pdf") or os.path.isfile(src):
        if not os.path.isfile(src):
            err(f"file not found: {src}")
        if not src.lower().endswith(".pdf"):
            err(f"only .pdf files are supported (got: {src})")
        pages = parse_pages(args.pages) if args.pages else None
        return text_from_pdf(src, pages), False
    err(f"don't know how to read {src!r} — expected a URL or a .pdf path")


def main():
    args = build_parser().parse_args()

    if args.list_voices:
        list_voices()
        return

    if args.speed <= 0:
        err("--speed must be greater than 0")

    lang_code = args.voice[0] if args.voice else "a"

    # Start loading and warming up the model now so it overlaps the fetch
    # and segmenting work below.
    get_model = start_model_load(args.voice, args.speed, lang_code)

    text, is_markdown = resolve_text(args)
    segments = shrink_first_segment(segment_text(text, is_markdown))
    if not segments:
        err("no readable text after extraction")

    spoken = " ".join(s.text for s in segments)
    words = len(spoken.split())
    est_sec = words / WORDS_PER_SEC / args.speed
    mins, secs = divmod(int(est_sec), 60)
    print(f"Text: {len(spoken):,} chars · {words:,} words · "
          f"{len(segments)} chunks · ~{mins}m{secs:02d}s of audio "
          f"(voice {args.voice}, speed {args.speed})")

    model = get_model()

    if args.save:
        save_to_file(model, segments, args.voice, args.speed, lang_code,
                     args.save)
    else:
        Player(model, segments, args.voice, args.speed, lang_code,
               est_sec).run()


if __name__ == "__main__":
    # Default SIGINT -> KeyboardInterrupt is what we rely on; keep it explicit.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping…")
        sys.exit(130)
