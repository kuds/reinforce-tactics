# Reinforce Tactics Examples

This directory contains example scripts demonstrating various features of Reinforce Tactics.

## LLM Bot Demo

**File:** `llm_bot_demo.py`

An interactive demo showing how to use LLM-powered bots (OpenAI, Claude, Gemini) to play the game.

### Prerequisites

1. Install an LLM provider package:
```bash
# For OpenAI
pip install openai>=1.0.0

# For Anthropic
pip install anthropic>=0.18.0

# For Google
pip install google-generativeai>=0.4.0
```

2. Set your API key:
```bash
export OPENAI_API_KEY='your-key-here'
# or
export ANTHROPIC_API_KEY='your-key-here'
# or
export GOOGLE_API_KEY='your-key-here'
```

### Usage

```bash
python examples/llm_bot_demo.py
```

The script will:
1. Prompt you to select an LLM provider
2. Load a map (or generate a random one)
3. Create an LLM bot
4. Run 3 demo turns showing the bot in action

### What You'll See

- Turn-by-turn game state (gold, current player)
- Bot thinking and executing actions
- Action logs showing what the bot decided to do
- Error handling if API keys are missing

### Troubleshooting

**"API key not provided"**
- Make sure you've exported the correct environment variable
- Check the variable name matches: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`

**"Missing dependency"**
- Install the required package: `pip install openai` (or `anthropic`, `google-generativeai`)

**Rate limiting**
- The bot has built-in retry logic with exponential backoff
- If you hit rate limits frequently, consider upgrading your API tier

## More Examples Coming Soon

- Headless training script
- Custom bot implementation tutorial
- Multi-player game setup
- Replay analysis tool
