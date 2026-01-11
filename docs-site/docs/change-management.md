---
sidebar_position: 7
id: change-management
title: Change Management
---

# Change Management

This page outlines how changes are managed in the Reinforce Tactics project, including versioning, contribution workflows, and release processes.

## Versioning

Reinforce Tactics follows [Semantic Versioning](https://semver.org/) (SemVer):

- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality in a backwards-compatible manner
- **PATCH** version for backwards-compatible bug fixes

### Version Format

```
MAJOR.MINOR.PATCH
```

**Examples:**
- `1.0.0` - Initial stable release
- `1.1.0` - New feature added (e.g., new unit type)
- `1.1.1` - Bug fix (e.g., combat calculation fix)
- `2.0.0` - Breaking change (e.g., save file format change)

## Contributing Changes

### Workflow Overview

1. **Fork** the repository or create a feature branch
2. **Implement** your changes with clear, focused commits
3. **Test** your changes locally
4. **Submit** a pull request with a descriptive title and description
5. **Review** - maintainers will review and provide feedback
6. **Merge** - once approved, changes are merged to main

### Branch Naming Conventions

| Branch Type | Format | Example |
|-------------|--------|---------|
| Feature | `feature/<description>` | `feature/add-ranger-unit` |
| Bug Fix | `fix/<description>` | `fix/mage-attack-range` |
| Documentation | `docs/<description>` | `docs/update-api-guide` |
| Refactor | `refactor/<description>` | `refactor/combat-system` |

### Commit Message Guidelines

Write clear, descriptive commit messages:

```
<type>: <short description>

[optional body with more details]
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `refactor` - Code refactoring
- `test` - Adding or updating tests
- `chore` - Maintenance tasks

**Examples:**
```
feat: Add Ranger unit with long-range attacks

fix: Correct Cleric healing calculation for units at max HP

docs: Update game mechanics with status effect details
```

## Pull Request Process

### Before Submitting

:::tip Checklist
- [ ] Code follows the project's style guidelines
- [ ] All existing tests pass
- [ ] New features include appropriate tests
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear and descriptive
:::

### PR Description Template

When creating a pull request, include:

1. **Summary** - Brief description of changes
2. **Motivation** - Why this change is needed
3. **Changes** - List of specific modifications
4. **Testing** - How the changes were tested
5. **Screenshots** - If applicable (UI changes)

### Review Process

1. **Automated checks** - CI runs tests and linting
2. **Code review** - At least one maintainer reviews the code
3. **Feedback** - Address any requested changes
4. **Approval** - Maintainer approves the PR
5. **Merge** - Changes are merged to main

## Release Process

### Release Cycle

Releases are made when significant features or fixes accumulate:

1. **Feature Freeze** - No new features, only bug fixes
2. **Testing** - Comprehensive testing of all features
3. **Changelog** - Update changelog with all changes
4. **Version Bump** - Update version number
5. **Tag & Release** - Create GitHub release with notes

### Changelog Format

The changelog follows the [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
## [1.2.0] - 2025-01-15

### Added
- New Ranger unit with long-range attacks
- Tournament bracket visualization

### Changed
- Improved bot AI decision making
- Updated UI color scheme

### Fixed
- Mage attack range calculation
- Save file corruption on special characters

### Deprecated
- Old save file format (will be removed in 2.0.0)
```

## Breaking Changes

### Definition

A breaking change is any modification that:

- Changes the save file format
- Modifies the RL environment observation/action space
- Removes or renames public API methods
- Changes game mechanics that affect trained models

### Handling Breaking Changes

1. **Announce early** - Notify users in advance
2. **Document migration** - Provide clear upgrade instructions
3. **Deprecation period** - When possible, deprecate before removing
4. **Major version** - Always increment major version

### Deprecation Policy

:::warning Deprecation Timeline
1. **v1.x** - Feature marked as deprecated with warning
2. **v1.x+1** - Deprecated feature still works, warning emphasized
3. **v2.0** - Deprecated feature removed
:::

## API Stability

### Stable APIs

The following are considered stable and follow strict compatibility:

- **Gymnasium Environment** - `StrategyGameEnv` observation and action spaces
- **Game State** - `GameState` public methods
- **File Formats** - Save and replay file structures
- **CLI Interface** - Command-line arguments for `main.py`

### Experimental APIs

Features marked as experimental may change without notice:

- LLM bot prompts and configurations
- Internal training utilities
- Debug and profiling tools

## Hotfix Process

For critical bugs in production:

1. Create branch from latest release tag
2. Fix the issue with minimal changes
3. Test thoroughly
4. Release as patch version (e.g., `1.1.1`)
5. Cherry-pick fix to main branch

```bash
# Example hotfix workflow
git checkout -b hotfix/critical-save-bug v1.1.0
# ... make fix ...
git commit -m "fix: Prevent save file corruption"
git tag v1.1.1
git push origin v1.1.1
```

## Getting Help

If you have questions about contributing or the change management process:

- **GitHub Issues** - Open an issue with the `question` label
- **Pull Request** - Ask questions directly in your PR
- **Documentation** - Check existing docs for guidance

## Summary

| Aspect | Policy |
|--------|--------|
| Versioning | Semantic Versioning (SemVer) |
| Branches | Feature branches with descriptive names |
| Commits | Conventional commit messages |
| PRs | Review required before merge |
| Releases | Tagged releases with changelogs |
| Breaking Changes | Major version bump required |
| Deprecation | Minimum one minor version warning |
