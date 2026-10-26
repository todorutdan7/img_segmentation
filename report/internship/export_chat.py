"""Turn the Claude session transcripts into readable Markdown.

    python export_chat.py                    # newest session only
    python export_chat.py --all              # every session, one file each
    python export_chat.py --tools            # also include tool calls

The raw transcripts are JSONL and total a few hundred MB, mostly base64 images.
This strips the images and the tool noise and keeps what you would actually want
to read later: what you asked and what the answer was.

Written for a Windows install; adjust TRANSCRIPT_DIR if the path differs.
"""

import argparse
import glob
import json
import os
import re
from datetime import datetime
from pathlib import Path

TRANSCRIPT_DIR = Path(os.path.expandvars(
    r"%APPDATA%\Claude\local-agent-mode-sessions"))


def find_transcripts(root: Path):
    """All *.jsonl under the Claude session folders, newest last."""
    if not root.exists():
        raise SystemExit(f"transcript folder not found: {root}\n"
                         f"open it in Explorer and check the path, then edit "
                         f"TRANSCRIPT_DIR at the top of this script.")
    files = [Path(p) for p in glob.glob(str(root / "**" / "*.jsonl"), recursive=True)]
    return sorted(files, key=lambda p: p.stat().st_mtime)


def blocks_to_text(content, keep_tools=False):
    """Flatten one message's content, dropping images and base64."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    out = []
    for b in content:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            out.append(b.get("text", "").strip())
        elif t == "image":
            out.append("_[image]_")
        elif t == "tool_use" and keep_tools:
            name = b.get("name", "?")
            arg = b.get("input", {})
            # show only the interesting field, truncated
            key = next((k for k in ("command", "file_path", "pattern", "query")
                        if k in arg), None)
            detail = str(arg.get(key, ""))[:160] if key else ""
            out.append(f"`[tool: {name}]` {detail}")
        elif t == "tool_result" and keep_tools:
            c = b.get("content")
            s = c if isinstance(c, str) else blocks_to_text(c, False)
            s = (s or "").strip()
            if s:
                out.append("```\n" + s[:800] + ("\n... (truncated)" if len(s) > 800 else "") + "\n```")
    return "\n\n".join(x for x in out if x)


def convert(path: Path, keep_tools=False):
    lines, n_user, n_asst = [], 0, 0
    stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

    lines.append(f"# Session {path.stem[:8]}")
    lines.append(f"\n_Last activity: {stamp} — source: `{path.name}`_\n")
    lines.append("---\n")

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                continue
            msg = d.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue

            text = blocks_to_text(msg.get("content"), keep_tools)
            if not text:
                continue
            # skip the auto-generated context-compaction summaries
            if role == "user" and text.startswith("This session is being continued"):
                text = "_[automatic summary of the earlier conversation]_"

            if role == "user":
                n_user += 1
                lines.append(f"\n## You\n\n{text}\n")
            else:
                n_asst += 1
                lines.append(f"\n### Claude\n\n{text}\n")

    lines.insert(2, f"_{n_user} messages from you, {n_asst} replies._\n")
    return "\n".join(lines), n_user, n_asst


def main():
    p = argparse.ArgumentParser(description="export Claude chats to Markdown")
    p.add_argument("--all", action="store_true", help="every session, not just the newest")
    p.add_argument("--tools", action="store_true", help="include tool calls and output")
    p.add_argument("--out", default="chat_export", help="output folder")
    a = p.parse_args()

    files = find_transcripts(TRANSCRIPT_DIR)
    if not files:
        raise SystemExit(f"no .jsonl transcripts under {TRANSCRIPT_DIR}")
    print(f"found {len(files)} transcript(s) in {TRANSCRIPT_DIR}")

    chosen = files if a.all else files[-1:]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    for f in chosen:
        md, nu, na = convert(f, a.tools)
        if nu + na == 0:
            continue
        stamp = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y%m%d_%H%M")
        dest = out / f"chat_{stamp}_{f.stem[:8]}.md"
        dest.write_text(md, encoding="utf-8")
        print(f"  {dest}   ({nu} + {na} messages, {len(md)/1024:.0f} KB)")
        total += 1

    print(f"\nwrote {total} file(s) to {out.resolve()}")


if __name__ == "__main__":
    main()
