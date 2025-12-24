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
| `max_turns` | Default maximum turns before draw | 100 |
| `map_pool_mode` | Map selection: `all`, `cycle`, `random` | `all` |
| `should_reason` | Enable LLM reasoning output | `true` |
| `log_conversations` | Save LLM conversations | `true` |
| `save_replays` | Save game replays | `true` |

### Map Configuration

Maps can be specified in two formats:

**Simple format** (uses default `max_turns`):
```json
"maps": ["maps/1v1/beginner.csv", "maps/1v1/funnel_point.csv"]
```

**Object format** (with per-map `max_turns`):
```json
"maps": [
  {"path": "maps/1v1/beginner.csv", "max_turns": 50},
  {"path": "maps/1v1/funnel_point.csv", "max_turns": 75},
  {"path": "maps/1v1/center_mountains.csv", "max_turns": 100}
]
```

You can mix both formats:
```json
"maps": [
  {"path": "maps/1v1/beginner.csv", "max_turns": 50},
  "maps/1v1/funnel_point.csv"
]
```

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

### Per-Map Turn Limits

```json
{
  "tournament": {
    "games_per_matchup": 1,
    "max_turns": 100,
    "map_pool_mode": "all"
  },
  "maps": [
    {"path": "maps/1v1/beginner.csv", "max_turns": 50},
    {"path": "maps/1v1/funnel_point.csv", "max_turns": 75},
    {"path": "maps/1v1/center_mountains.csv", "max_turns": 100}
  ],
  "bots": [
    {"name": "SimpleBot", "type": "simple"},
    {"name": "MediumBot", "type": "medium"},
    {"name": "AdvancedBot", "type": "advanced"}
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

## Advanced Features

### Resuming Interrupted Tournaments

If a tournament is interrupted (e.g., due to network issues, container restart, or manual stop), you can resume it by passing the `--resume` flag with the path to the previous output folder:

```bash
# Resume from previous output
docker run --rm \
  -v $(pwd)/docker/tournament/config.json:/app/config/config.json:ro \
  -v $(pwd)/docker/tournament/output:/app/output \
  -v $(pwd)/maps:/app/maps:ro \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  reinforce-tactics-tournament --resume /app/output
```

The resume feature:
- Scans existing replay files to determine which matches have been completed
- Skips already completed matches
- Plays only the remaining matches
- Logs resume statistics showing skipped vs new games

**Note**: Resume requires `save_replays: true` in the config so that completed match information is available.

### Google Cloud Storage Upload

For cloud deployments (e.g., Google Cloud Run, GCE), you can configure automatic upload of tournament output to Google Cloud Storage.

#### Configuration

Add a `gcs` section to your `config.json`:

```json
{
  "gcs": {
    "enabled": true,
    "bucket": "your-bucket-name",
    "prefix": "tournaments/run-001/",
    "credentials_file": null
  }
}
```

| Setting | Description | Default |
|---------|-------------|---------|
| `enabled` | Enable GCS upload | `false` |
| `bucket` | GCS bucket name (required if enabled) | - |
| `prefix` | Optional folder prefix in bucket | `""` |
| `credentials_file` | Path to service account JSON (optional) | `null` |

#### Authentication

GCS upload supports multiple authentication methods:

1. **Service Account JSON** (recommended for production):
   ```json
   {
     "gcs": {
       "enabled": true,
       "bucket": "my-bucket",
       "credentials_file": "/app/config/service-account.json"
     }
   }
   ```

2. **Environment Variable** (default credentials):
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
   ```

3. **GCE Metadata Server** (automatic on Google Cloud):
   When running on GCE, GKE, or Cloud Run, credentials are obtained automatically from the metadata server. Just ensure the service account has Storage Object Admin permissions.

#### Running on Google Cloud

Example for Google Cloud Run:

```bash
# Build and push image
docker build -f docker/tournament/Dockerfile -t gcr.io/my-project/tournament .
docker push gcr.io/my-project/tournament

# Deploy to Cloud Run
gcloud run deploy tournament \
  --image gcr.io/my-project/tournament \
  --set-env-vars OPENAI_API_KEY=$OPENAI_API_KEY,ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  --memory 4Gi \
  --timeout 3600
```

#### GCS Output Structure

When GCS upload is enabled, files are uploaded with this structure:

```
gs://bucket-name/prefix/
├── results/
│   ├── tournament_results_YYYYMMDD_HHMMSS.json
│   ├── tournament_standings_YYYYMMDD_HHMMSS.csv
│   └── ...
├── replays/
│   └── {map_name}/{bot1}_vs_{bot2}/
│       └── game_*.json
└── conversations/
    └── {map_name}/{bot1}_vs_{bot2}/
        └── game_*_player{N}_*.json
```

### Combined Resume + GCS

You can use both features together for robust cloud execution:

```json
{
  "tournament": {
    "save_replays": true,
    "log_conversations": true
  },
  "gcs": {
    "enabled": true,
    "bucket": "my-tournament-bucket",
    "prefix": "tournament-2024-01/"
  }
}
```

To resume a cloud tournament, download the previous output and mount it:

```bash
# Download previous output from GCS
gsutil -m cp -r gs://my-tournament-bucket/tournament-2024-01/replays ./output/replays

# Resume the tournament
docker run --rm \
  -v $(pwd)/config.json:/app/config/config.json:ro \
  -v $(pwd)/output:/app/output \
  reinforce-tactics-tournament --resume /app/output
```
