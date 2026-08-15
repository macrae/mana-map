# Mana Map — the commands you actually need.
#
# `make setup` exists because the one install constraint that will break your
# machine was, until now, prose in two files and enforced nowhere: llvmlite and
# numba must be installed BEFORE the project, or pacmap pulls numba itself and
# pip falls back to building LLVM from source. That is a twenty-minute detour
# ending in a compiler error, and nobody reads the sentence warning about it
# until afterwards.

VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
PYTEST  := $(PY) -m pytest
PORT    ?= 8000

# The 3.10 interpreter to build the venv from. Overridable, because plenty of
# people have 3.10 without having `python3.10` on PATH — this repo's own
# development venv is a conda build, so `make setup` would have refused to run on
# the machine that wrote it:
#
#   make setup PYTHON310=$$HOME/opt/miniconda3/envs/py310/bin/python3.10
#   make setup PYTHON310=$$(pyenv which python3.10)
PYTHON310 ?= python3.10

.DEFAULT_GOAL := help
.PHONY: help setup test test-all test-browser test-fresh serve manuals clean check

help:  ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Serve the map with no Python at all:"
	@echo "    python3 -m http.server $(PORT)   then open localhost:$(PORT)/viz/index.html"

$(VENV):
	@"$(PYTHON310)" -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,10) else 1)' \
	  2>/dev/null || { \
	  echo "ERROR: no usable Python 3.10 at '$(PYTHON310)'."; \
	  echo "  This project needs 3.10 EXACTLY — PyTorch publishes no wheels for"; \
	  echo "  3.13+, and the pins target torch 2.2.2."; \
	  echo ""; \
	  echo "  Install one:   brew install python@3.10   |   pyenv install 3.10"; \
	  echo "  Already have one that isn't called python3.10 on PATH?"; \
	  echo "    make setup PYTHON310=/path/to/python3.10"; \
	  exit 1; }
	"$(PYTHON310)" -m venv $(VENV)

setup: $(VENV)  ## Create .venv and install everything, in the order that works
	@echo "==> llvmlite + numba FIRST (prebuilt wheels; pacmap would source-build LLVM)"
	$(PIP) install --quiet --upgrade pip
	$(PIP) install llvmlite==0.41.1 numba==0.58.1
	@echo "==> the project and its dev extras"
	$(PIP) install -e ".[dev]"
	@echo "==> chromium for the browser tests (one time, ~150 MB)"
	$(PY) -m playwright install chromium
	@echo ""
	@echo "Done. Try:  make test"

test:  ## The inner loop: non-browser, parallel, cached (~20s)
	$(PYTEST)

test-fresh:  ## Same, but nothing served from the cache — trust this one
	$(PYTEST) --no-test-cache

test-browser:  ## The playwright suite (~4 min; needs `make setup`)
	$(PYTEST) -m "browser and not serial_only" -n 4
	$(PYTEST) -m "browser and serial_only"

test-all: test-fresh test-browser  ## Everything, uncached. What CI would run if it ran it all.

check: test  ## Alias for `make test` — what to run before opening a PR

serve:  ## Serve the map and the manuals (PORT=8000 by default)
	@echo "  map      http://localhost:$(PORT)/viz/index.html"
	@echo "  issues   http://localhost:$(PORT)/manuals/index.html"
	python3 -m http.server $(PORT)

manuals:  ## Re-render every published issue (deterministic; should be a no-op)
	@for slug in $$(ls data/decks | grep -v '^index.json$$'); do \
	  test -f data/decks/$$slug/issue.json && $(VENV)/bin/manamap pilot build-manual $$slug; \
	done
	$(VENV)/bin/manamap pilot build-index

clean:  ## Drop caches and bytecode. Never touches data/ or manuals/.
	rm -rf .pytest_cache
	find . -name __pycache__ -type d -not -path "./$(VENV)/*" -exec rm -rf {} +
