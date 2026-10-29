uv_run := "uv run --extra all"

# List available recipes
help:
    @just --list

# Setup dev env
setup:
    uv sync --extra all

# Format code
format:
    {{ uv_run }} ruff format
    {{ uv_run }} ruff check --fix
    git ls-files "*.toml" | xargs {{ uv_run }} taplo fmt
    git ls-files "*.md" | xargs {{ uv_run }} mdformat
    git ls-files "*.yml" "*.yaml" | xargs {{ uv_run }} yamlfix
    {{ uv_run }} typos --write-changes

# Check code
lint:
    {{ uv_run }} ruff format --check
    {{ uv_run }} ruff check
    {{ uv_run }} gruff check .
    {{ uv_run }} ty check --no-progress
    git ls-files "*.toml" | xargs {{ uv_run }} taplo fmt --check
    git ls-files "*.md" | xargs {{ uv_run }} mdformat --check
    git ls-files "*.yml" "*.yaml" | xargs {{ uv_run }} yamlfix --check
    {{ uv_run }} typos

# Run tests
test:
    {{ uv_run }} pytest -n=auto -v tests

# Prepare a release: just release X.Y.Z
[positional-arguments]
release version="":
    #!/usr/bin/env sh
    set -eu
    version=$1
    test -n "$version" || {
        fragments=$(find changelog.d -maxdepth 1 -type f \( \
            -name "*.added.md" -o -name "*.changed.md" -o \
            -name "*.deprecated.md" -o -name "*.removed.md" -o \
            -name "*.fixed.md" -o -name "*.security.md" \))
        latest=$(git tag --sort=-v:refname |
            grep -E "^v[0-9]+\.[0-9]+\.[0-9]+$" | head -1)
        if test -n "$fragments" && test -n "$latest"; then
            version=${latest#v}
            major=${version%%.*}
            remainder=${version#*.}
            minor=${remainder%%.*}
            patch=${remainder#*.}
            if grep -q '\*\*Breaking:\*\*' $fragments; then
                next=$((major + 1)).0.0
            elif find changelog.d -maxdepth 1 -type f \( \
                -name "*.added.md" -o -name "*.changed.md" -o \
                -name "*.deprecated.md" -o -name "*.removed.md" \) | grep -q .; then
                next=$major.$((minor + 1)).0
            else
                next=$major.$minor.$((patch + 1))
            fi
            echo "suggested: just release $next" >&2
        else
            echo "usage: just release X.Y.Z" >&2
        fi
        echo "recent releases:" >&2
        git tag --sort=-v:refname | head -5 | sed "s/^/  /" >&2
        exit 1
    }
    uv run towncrier build --yes --version "$version"
    uv run mdformat CHANGELOG.md
    git add CHANGELOG.md
    printf '\nNext steps\n'
    echo "  git commit -am \"chore: prep $version release\""
    echo "  git tag v$version"
    echo "  git push origin main v$version"
    echo "  Create and publish the GitHub release for v$version"
