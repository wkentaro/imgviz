ifneq ($(OS),Windows_NT)
	# On Unix-based systems, use ANSI codes
	BLUE = \033[36m
	BOLD_BLUE = \033[1;36m
	BOLD_GREEN = \033[1;32m
	RED = \033[31m
	YELLOW = \033[33m
	BOLD = \033[1m
	NC = \033[0m
endif

escape = $(subst $$,\$$,$(subst ",\",$(subst ',\',$(1))))

define exec
	@echo "$(BOLD_BLUE)$(call escape,$(1))$(NC)"
	@$(1)
endef

help:
	@echo "$(BOLD_GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-].+:.*?# .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?# "}; \
		{printf "  $(BOLD_BLUE)%-20s$(NC) %s\n", $$1, $$2}'

PACKAGE_NAME := imgviz

UV_RUN := uv run --extra all

setup:  # Setup dev env
	$(call exec,uv sync --extra all)

format:  # Format code
	$(call exec,$(UV_RUN) ruff format)
	$(call exec,$(UV_RUN) ruff check --fix)
	$(call exec,git ls-files "*.toml" | xargs $(UV_RUN) taplo fmt)
	$(call exec,git ls-files "*.md" | xargs $(UV_RUN) mdformat)
	$(call exec,git ls-files "*.yml" "*.yaml" | xargs $(UV_RUN) yamlfix)
	$(call exec,$(UV_RUN) typos --write-changes)

lint:  # Check code
	$(call exec,$(UV_RUN) ruff format --check)
	$(call exec,$(UV_RUN) ruff check)
	$(call exec,$(UV_RUN) ty check --no-progress)
	$(call exec,git ls-files "*.toml" | xargs $(UV_RUN) taplo fmt --check)
	$(call exec,git ls-files "*.md" | xargs $(UV_RUN) mdformat --check)
	$(call exec,git ls-files "*.yml" "*.yaml" | xargs $(UV_RUN) yamlfix --check)
	$(call exec,$(UV_RUN) typos)

test:
	$(call exec,$(UV_RUN) pytest -n=auto -v tests)

release:  # Prepare a release: make release VERSION=X.Y.Z
	@test -n "$(VERSION)" || { \
		fragments=$$(find changelog.d -maxdepth 1 -type f \( \
			-name "*.added.md" -o -name "*.changed.md" -o \
			-name "*.deprecated.md" -o -name "*.removed.md" -o \
			-name "*.fixed.md" -o -name "*.security.md" \)); \
		latest=$$(git tag --sort=-v:refname | \
			grep -E "^v[0-9]+\.[0-9]+\.[0-9]+$$" | head -1); \
		if test -n "$$fragments" && test -n "$$latest"; then \
			version=$${latest#v}; \
			major=$${version%%.*}; \
			remainder=$${version#*.}; \
			minor=$${remainder%%.*}; \
			patch=$${remainder#*.}; \
			if grep -q '\*\*Breaking:\*\*' $$fragments; then \
				next=$$((major + 1)).0.0; \
			elif find changelog.d -maxdepth 1 -type f \( \
				-name "*.added.md" -o -name "*.changed.md" -o \
				-name "*.deprecated.md" -o -name "*.removed.md" \) | grep -q .; then \
				next=$$major.$$((minor + 1)).0; \
			else \
				next=$$major.$$minor.$$((patch + 1)); \
			fi; \
			echo "suggested: make release VERSION=$$next" >&2; \
		else \
			echo "usage: make release VERSION=X.Y.Z" >&2; \
		fi; \
		echo "recent releases:" >&2; \
		git tag --sort=-v:refname | head -5 | sed "s/^/  /" >&2; \
		exit 1; \
	}
	$(call exec,uv run towncrier build --yes --version $(VERSION))
	$(call exec,uv run mdformat CHANGELOG.md && git add CHANGELOG.md)
	@printf "\n$(BOLD_GREEN)Next steps$(NC)\n"
	@echo "  git commit -am \"chore: prep $(VERSION) release\""
	@echo "  git tag v$(VERSION)"
	@echo "  git push origin main v$(VERSION)"
	@echo "  Create and publish the GitHub release for v$(VERSION)"
