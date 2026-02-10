# AI Playground - Isolated Environment

This Docker setup creates an isolated environment where AI agents can only access files within this directory.

## Setup

1. Build the container:
   ```bash
   docker-compose build
   ```

2. Start the container:
   ```bash
   docker-compose up -d
   ```

3. Enter the container:
   ```bash
   docker exec -it ai-playground /bin/bash
   ```

## Usage

- All your work should be done in the `/workspace` directory inside the container
- This directory maps to `./workspace` in your AI-playground folder
- The container has NO access to your home directory or other system files
- When working with AI agents, run commands from within this container

## Stop the container

```bash
docker-compose down
```

## Quick access alias (optional)

Add to your `~/.zshrc`:
```bash
alias ai-playground='docker exec -it ai-playground /bin/bash'
```

Then just run `ai-playground` to enter the isolated environment.
