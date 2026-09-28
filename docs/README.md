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
| [`vertex_training.md`](vertex_training.md) | Contributors | Run training on Google Cloud via Vertex AI custom jobs (Docker + GCS) |
| [`validation_run.md`](validation_run.md) | Contributors | Runbook of the 3-seed bootstrap validation run: launch (local, Colab, Vertex), resume, aggregate, report |
| [`validation_run_config.md`](validation_run_config.md) | Contributors | The validation run's config and its evidence: reward back-port, the 24 ladder-ordered stages, the Wilson gate, budgets and compute, the slice and self-play |
| [`MAP_EDITOR.md`](MAP_EDITOR.md) | Contributors | How the in-game map editor works internally |
| [`REVIEW_full_2026-09-26.md`](REVIEW_full_2026-09-26.md) | Contributors | Full codebase review (Sep 2026): prioritized fixes and development roadmap across engine, bots, RL pipeline, pygame UI and animations |
| [`REVIEW_full_2026-09-26_findings.md`](REVIEW_full_2026-09-26_findings.md) | Contributors | Appendix to the above: all 351 verified findings with locations and fixes |
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
