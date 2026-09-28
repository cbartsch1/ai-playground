#!/usr/bin/env bash
# Timestamps every prompt and response, regardless of which model is serving the session.
# Usage: timestamp.sh prompt|response   (wired up in .claude/settings.json)
# Override the zone with AI_PLAYGROUND_TZ (default: US/Eastern, the ES/NQ session clock).
cat >/dev/null  # drain hook JSON on stdin
# If the global copy (scripts/install-global-timestamps.sh) is installed, let it do the stamping.
global="$HOME/.claude/hooks/timestamp.sh"
if [ -x "$global" ] && [ "$(cd "$(dirname "$0")" && pwd)/$(basename "$0")" != "$global" ]; then exit 0; fi
ts="$(TZ="${AI_PLAYGROUND_TZ:-America/New_York}" date '+%Y-%m-%d %H:%M:%S %Z')"
case "$1" in
  prompt)
    printf '{"systemMessage":"🕒 Prompt  %s","hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":"Prompt received at %s. Begin your response with a line of the form: 🕒 %s"}}\n' "$ts" "$ts" "$ts"
    ;;
  response)
    printf '{"systemMessage":"🕒 Response  %s"}\n' "$ts"
    ;;
esac
