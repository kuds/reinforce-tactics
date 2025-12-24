# Reinforce Tactics Tournament Docker Container

Run LLM bot tournaments in a containerized environment.

## Quick Start

### 1. Set up API Keys

Create a `.env` file in this directory with your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

Or export them directly:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

### 2. Configure the Tournament

Edit `config.json` to customize:
- **bots**: Which bots to include (remove LLM bots if you don't have the API keys)
- **maps**: Which maps to use
- **tournament settings**: games per matchup, max turns, etc.

### 3. Run the Tournament

```bash
# Build and run
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f
```

### 4. View Results

Results are saved to the `output/` directory:
- `output/results/` - Tournament results (JSON, CSV)
- `output/conversations/` - LLM conversation logs
- `output/replays/` - Game replays

## Configuration

### Tournament Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `games_per_matchup` | Games per side per map | 1 |
| `max_turns` | Maximum turns before draw | 100 |
| `map_pool_mode` | Map selection: `all`, `cycle`, `random` | `all` |
| `should_reason` | Enable LLM reasoning output | `true` |
| `log_conversations` | Save LLM conversations | `true` |
| `save_replays` | Save game replays | `true` |

### Bot Types

| Type | Description | Requirements |
|------|-------------|--------------|
| `simple` | Basic rule-based bot | None |
| `medium` | Medium difficulty bot | None |
| `advanced` | Advanced strategic bot | None |
| `llm` | LLM-powered bot | API key for provider |
| `model` | Trained RL model bot | Model file |

### LLM Providers

| Provider | Environment Variable | Supported Models |
|----------|---------------------|------------------|
| `openai` | `OPENAI_API_KEY` | gpt-4o, gpt-4o-mini, gpt-4-turbo |
| `anthropic` | `ANTHROPIC_API_KEY` | claude-3-5-sonnet, claude-3-5-haiku |
| `google` | `GOOGLE_API_KEY` | gemini-2.0-flash, gemini-1.5-pro |

## Example Configurations

### Simple Tournament (No API Keys)

```json
{
  "tournament": {
    "games_per_matchup": 2,
    "max_turns": 100,
    "map_pool_mode": "all"
  },
  "maps": ["maps/1v1/beginner.csv"],
  "bots": [
    {"name": "SimpleBot", "type": "simple"},
    {"name": "MediumBot", "type": "medium"},
    {"name": "AdvancedBot", "type": "advanced"}
  ]
}
```

### LLM Tournament

```json
{
  "tournament": {
    "games_per_matchup": 1,
    "max_turns": 100,
    "should_reason": true,
    "log_conversations": true
  },
  "maps": [
    "maps/1v1/beginner.csv",
    "maps/1v1/funnel_point.csv"
  ],
  "bots": [
    {"name": "SimpleBot", "type": "simple"},
    {"name": "GPT-4o Mini", "type": "llm", "provider": "openai", "model": "gpt-4o-mini"},
    {"name": "Claude Haiku", "type": "llm", "provider": "anthropic", "model": "claude-3-5-haiku-20241022"}
  ]
}
```

## Building Manually

```bash
# From the repository root
docker build -f docker/tournament/Dockerfile -t reinforce-tactics-tournament .

# Run with config
docker run --rm \
  -v $(pwd)/docker/tournament/config.json:/app/config/config.json:ro \
  -v $(pwd)/docker/tournament/output:/app/output \
  -v $(pwd)/maps:/app/maps:ro \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -e GOOGLE_API_KEY=$GOOGLE_API_KEY \
  reinforce-tactics-tournament
```

## Troubleshooting

### API Key Not Found

If you see warnings like "Skipping Bot: OPENAI_API_KEY not set":
1. Check that your `.env` file exists and contains the correct keys
2. Or export the environment variables before running docker-compose

### Not Enough Bots

The tournament requires at least 2 bots. If LLM bots are skipped due to missing API keys, add more rule-based bots or configure the API keys.

### Permission Denied on Output

Make sure the `output/` directory is writable:
```bash
chmod -R 777 output/
```
