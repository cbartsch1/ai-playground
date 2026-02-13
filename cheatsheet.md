# AI CLI Tools Cheat Sheet
**M3 Ultra - 96 GB RAM - Day Trading Algo Development**

## THE FULL TEAM

| # | Command | Model | Provider | Location | Cost | Best For |
|---|---------|-------|----------|----------|------|----------|
| 1 | `claude` | Claude Opus 4.6 | Anthropic | Cloud | Tokens | Complex architecture, hard debugging, agentic tasks |
| 2 | `claude-local` | GLM4.7-Flash (30B) | Ollama | Local | Free | General tasks, quick edits, boilerplate |
| 3 | `claude-coder` | Qwen3-Coder (30B) | Ollama | Local | Free | Python, pandas, trading logic, API integrations |
| 4 | `gemini` | Gemini 2.5 | Google | Cloud | Quota | Complex tasks, second opinion |
| 5 | `gemini-local` | GLM4.7-Flash (30B) | LiteLLM->Ollama | Local | Free | General tasks via Gemini CLI |
| 6 | `grok` | Grok 4 | xAI | Cloud | API key | Heavy reasoning, complex refactors |
| 7 | GLM Chat (Desktop app) | GLM4.7-Flash (30B) | Ollama | Local | Free | Quick chat without terminal |

3 cloud members (powerful, costs tokens) - 4 local members (free, unlimited, no timeouts)

## Performance Notes (Hybrid Strategy Review Test)

| Model | Found Real Bugs | Wrote Code | False Positives | Verdict |
|-------|----------------|------------|-----------------|---------|
| Qwen3-Coder (C+) | 0/4 but suggested ATR stops | No code despite being asked | 1 (wrong "critical bug") | Useful starting point, can't finish the job |
| GLM4.7-Flash (F) | 0/4 | Empty response | N/A | Couldn't handle 314-line strategy at all |
| Claude Opus (did the work) | 4/4 | All fixes written and committed | 0 | Found dashboard lie, ATR stops, session flatten, dead code |

**Bottom line**: Use local models for routine boilerplate and simple tasks. Strategy reviews, architecture, and real debugging still need the cloud team.

## WHEN TO USE WHAT

**Use `claude-coder` (free) for:**
- Python trading logic, strategy code, indicator functions
- Pandas/numpy data pipelines
- Broker API integrations
- Backtesting framework code
- Routine code generation and refactoring

**Use `claude-local` or `gemini-local` (free) for:**
- General-purpose tasks, boilerplate, config files
- Git commit messages, documentation
- Quick explanations and simple debugging

**Use `claude` or `gemini` or `grok` (cloud) for:**
- Complex strategy architecture and design decisions
- Debugging subtle order execution edge cases
- Large codebase refactors
- Multi-step agentic tasks
- When local models give poor results

## COMMON COMMANDS

### Claude Code
```
claude                    # Interactive chat (cloud)
claude-local              # Interactive chat (local GLM, free)
claude-coder              # Interactive chat (local Qwen3-Coder, free)
claude --print "prompt"   # One-shot answer (cloud)
claude-coder --print "prompt"  # One-shot answer (local coder, free)
claude --resume           # Resume last session
claude --model claude-sonnet-4-5-20250929  # Use Sonnet instead of Opus
```

### Gemini CLI
```
gemini                    # Interactive chat (cloud)
gemini-local              # Interactive chat (local, free)
gemini -p "prompt"        # One-shot answer (cloud)
gemini-local -p "prompt"  # One-shot answer (local, free)
gemini -r latest          # Resume last session
```

### Grok CLI
```
grok                      # Interactive chat (cloud only)
grok --model grok-4-latest  # Use specific model
grok --resume <sessionId>   # Resume a session
grok --continue           # Continue most recent session
```

### Ollama (model manager)
```
ollama list               # List downloaded models
ollama run glm-4.7-flash  # Chat directly with GLM
ollama run qwen3-coder    # Chat directly with Qwen3-Coder
ollama pull <model>       # Download a new model
ollama rm <model>         # Delete a model
ollama ps                 # Show running models
ollama show <model>       # Model details (size, quant, etc.)
```

## LOCAL MODELS INSTALLED

| Model | Size | Strengths |
|-------|------|-----------|
| GLM-4.7-Flash | 19 GB | General purpose, fast, good reasoning |
| Qwen3-Coder | 18 GB | Code-specialized, strong Python/pandas/numpy |

## SERVICES RUNNING IN BACKGROUND

| Service | Port | Auto-starts | Managed by |
|---------|------|-------------|------------|
| Ollama | 11434 | Yes (login) | brew services |
| LiteLLM | 4000 | Yes (login) | LaunchAgent |

### Service Management
```
# Ollama
brew services start ollama
brew services stop ollama
brew services restart ollama

# LiteLLM proxy
launchctl load ~/Library/LaunchAgents/com.local.litellm-proxy.plist    # Start
launchctl unload ~/Library/LaunchAgents/com.local.litellm-proxy.plist  # Stop

# Health checks
curl -s http://localhost:11434/api/tags | python3 -m json.tool   # Ollama
curl -s http://localhost:4000/health | python3 -m json.tool      # LiteLLM
```

## KEY FILES & CONFIGS

| File | Purpose |
|------|---------|
| `~/.zshrc` | Shell aliases & env vars |
| `~/.keys` | API keys (chmod 600) |
| `~/.litellm/config.yaml` | LiteLLM -> Ollama routing |
| `~/.gemini/settings.json` | Gemini CLI settings |
| `~/.grok/user-settings.json` | Grok CLI settings |
| `~/Library/LaunchAgents/com.local.litellm-proxy.plist` | LiteLLM auto-start |
| `/tmp/litellm.log` | LiteLLM proxy logs |

## ADDING MORE LOCAL MODELS
```
ollama pull deepseek-r1:14b   # Math/reasoning (good for quant work)
ollama pull llama3.3          # General purpose
```

## TROUBLESHOOTING

| Problem | Fix |
|---------|-----|
| `claude-coder` not found | Open a new terminal (aliases load on shell start) |
| Ollama not responding | `brew services restart ollama` |
| LiteLLM not responding | Check `/tmp/litellm.log` for errors |
| Gemini-local model not found | Verify LiteLLM is running: `curl localhost:4000/health` |
| Model running slow | Check `ollama ps` — only one model loaded at a time |
| Out of disk space | `ollama rm <unused-model>` to free space |
