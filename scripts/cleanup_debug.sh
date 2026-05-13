#!/bin/bash
# SmartX Vision — Limpeza de arquivos debug
# Remove debug_*.jpg com mais de X horas da raiz /data

DATA_DIR="/opt/vision/data"
MAX_AGE_HOURS=24  # arquivos mais velhos que 24h serão removidos
LOG_FILE="/opt/vision/data/cleanup.log"

# Conta antes
BEFORE=$(find "$DATA_DIR" -maxdepth 1 -name "debug_*.jpg" | wc -l)

# Remove arquivos mais velhos que MAX_AGE_HOURS
find "$DATA_DIR" -maxdepth 1 -name "debug_*.jpg" -mmin +$((MAX_AGE_HOURS * 60)) -delete

# Conta depois
AFTER=$(find "$DATA_DIR" -maxdepth 1 -name "debug_*.jpg" | wc -l)
REMOVED=$((BEFORE - AFTER))

# Log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Removidos: $REMOVED arquivos debug (restam: $AFTER)" >> "$LOG_FILE"

# Mantém só as últimas 100 linhas do log
tail -100 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
