# Reinforce Tactics Documentation Site

This is the **user-facing** documentation website for Reinforce Tactics, built
using [Docusaurus](https://docusaurus.io/), a modern static website generator.
It is deployed at [reinforcetactics.com](https://reinforcetactics.com).

> Contributor-facing notes (roadmap, internal code reviews, dev guides) live in
> the repo-level [`docs/`](../docs/) directory — see [`docs/README.md`](../docs/README.md).

## Installation

```bash
cd docs-site
npm install
```

## Local Development

```bash
cd docs-site
npm start
```

This command starts a local development server and opens up a browser window at `http://localhost:3000`. Most changes are reflected live without having to restart the server.

## Build

```bash
cd docs-site
npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Documentation Structure

The documentation includes:
- **Getting Started** (`intro.md`): Overview, features, and quick start guide
- **Game Mechanics** (`game-mechanics.md`): Units, combat system, structures, and terrain
- **Bot Tournaments** (`tournaments.mdx`): Official tournament results and analysis
- **Maps** (`maps.mdx`): Available maps with previews and descriptions
- **Tournament System** (`tournament-system.md`): Technical guide for running tournaments
- **Implementation Status** (`implementation-status.md`): Current state of features and development roadmap

## Deployment

The site is automatically deployed to GitHub Pages via GitHub Actions when changes are pushed to the main branch.

## Contributing to Documentation

To add or update documentation:

1. Edit markdown files in the `docs/` directory
2. Test locally with `npm start`
3. Commit and push your changes
4. The site will be automatically deployed

For more information about Docusaurus, visit the [official documentation](https://docusaurus.io/).
