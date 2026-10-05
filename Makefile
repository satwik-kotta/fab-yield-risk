.PHONY: db load train score causal dashboard test all
db:        ; docker compose up -d --wait
load:      ; python -m src.stream_simulator --mode backfill
train:     ; python -W ignore -m src.model
score:     ; python -W ignore -m src.score
causal:    ; python -W ignore -m src.causal
dashboard: ; python -m src.export_dashboard
test:      ; python -W ignore -m pytest -q tests
all: load train score causal dashboard
