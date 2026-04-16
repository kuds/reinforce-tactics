# Contributor & Dev Documentation

This directory holds **contributor-facing** documentation that lives with the
source code: roadmap, internal code reviews, and developer guides that aren't
part of the published user manual.

> **User-facing docs** (install, game rules, API reference, tournaments) are
> published at **[reinforcetactics.com](https://reinforcetactics.com)** and
> sourced from [`docs-site/`](../docs-site/).

## What's here

| File | Audience | Purpose |
|---|---|---|
| [`ROADMAP.md`](ROADMAP.md) | Contributors | Planned features, milestones, and open work |
| [`MAP_EDITOR.md`](MAP_EDITOR.md) | Contributors | How the in-game map editor works internally |
| [`REVIEW_maintainability.md`](REVIEW_maintainability.md) | Contributors | Code-quality review: duplication, bugs, refactor priorities |
| [`REVIEW_advancedbot.md`](REVIEW_advancedbot.md) | Contributors | Code review of the advanced rule-based bot |
| [`feudal_rl_review.md`](feudal_rl_review.md) | Contributors | Code review of the feudal RL implementation |

## When to add a doc here vs. in `docs-site/`

- **Here (`docs/`)** — internal notes, review findings, roadmap items, dev
  guides that change often alongside code. Not published.
- **In [`docs-site/docs/`](../docs-site/docs/)** — anything a user would read:
  how to install, play, train, configure bots, run tournaments. Published to
  [reinforcetactics.com](https://reinforcetactics.com) via Docusaurus.

If a contributor note in `docs/` matures into something useful to users,
promote it into `docs-site/docs/` and delete (or stub out) the original.
