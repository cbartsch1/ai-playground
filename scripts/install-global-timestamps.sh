#!/usr/bin/env bash
# Install prompt/response timestamps for ALL Claude Code sessions on this machine
# (every project, every model). Safe to re-run; existing ~/.claude/settings.json
# content is preserved and a backup is written first.
#   Usage (from the repo root): bash scripts/install-global-timestamps.sh
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
dest="$HOME/.claude/hooks/timestamp.sh"
settings="$HOME/.claude/settings.json"

mkdir -p "$HOME/.claude/hooks"
cp "$repo/.claude/hooks/timestamp.sh" "$dest"
chmod +x "$dest"

if [ -f "$settings" ]; then
  cp "$settings" "$settings.bak.$(date +%Y%m%d%H%M%S)"
fi

python3 - "$settings" "$dest" <<'PY'
import json, os, sys
path, script = sys.argv[1], sys.argv[2]
cfg = {}
if os.path.exists(path) and os.path.getsize(path):
    with open(path) as f:
        cfg = json.load(f)
cfg["showMessageTimestamps"] = True
hooks = cfg.setdefault("hooks", {})
for event, arg in (("UserPromptSubmit", "prompt"), ("Stop", "response")):
    cmd = f'"{script}" {arg}'
    groups = hooks.setdefault(event, [])
    if any(h.get("command") == cmd for g in groups for h in g.get("hooks", [])):
        continue
    groups.append({"hooks": [{"type": "command", "command": cmd, "timeout": 5}]})
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY

echo "Installed: $dest"
echo "Updated:   $settings"
echo "Start a new Claude Code session (or open /hooks once) to activate."
