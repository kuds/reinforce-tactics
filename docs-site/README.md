# Reinforce Tactics Documentation Site

This is the documentation website for Reinforce Tactics, built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

The documentation is deployed at [reinforcetactics.com](https://reinforcetactics.com).

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
- **Implementation Status** (`implementation-status.md`): Current state of features and development roadmap
- **Tournament System** (`tournament-system.md`): Bot tournament system documentation

## Deployment

The site is automatically deployed to GitHub Pages via GitHub Actions when changes are pushed to the main branch.

## Contributing to Documentation

To add or update documentation:

1. Edit markdown files in the `docs/` directory
2. Test locally with `npm start`
3. Commit and push your changes
4. The site will be automatically deployed

For more information about Docusaurus, visit the [official documentation](https://docusaurus.io/).
