SHELL := /bin/bash

PY ?= python
ENV_FILE ?= .env
RUN := set -a; [ -f $(ENV_FILE) ] && source $(ENV_FILE); set +a; $(PY) gatherer.py

.PHONY: help list schedule all predictions env-check

help:
	@printf "Agent Seismograph\n"
	@printf "  make list                 # list predictions\n"
	@printf "  make schedule             # run today's schedule (from catalog)\n"
	@printf "  make all                  # scan all predictions\n"
	@printf "  make predictions PIDS=... # scan specific IDs\n"
	@printf "  make env-check            # verify API keys in env/.env\n"

list:
	@$(RUN) --list

schedule:
	@$(RUN) --schedule

all:
	@$(RUN) --all

predictions:
	@test -n "$(PIDS)" || (echo "PIDS required, e.g. PIDS='INF-01 SEC-05'" >&2; exit 1)
	@$(RUN) --predictions $(PIDS)

env-check:
	@set -a; [ -f $(ENV_FILE) ] && source $(ENV_FILE); set +a; \
	if [ -z "$$GROQ_API_KEY" ]; then echo "ERROR: GROQ_API_KEY missing"; exit 1; fi; \
	if [ -z "$$TAVILY_API_KEY" ]; then echo "ERROR: TAVILY_API_KEY missing"; exit 1; fi; \
	echo "OK: keys present"
