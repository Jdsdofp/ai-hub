#!/usr/bin/env python3
"""
Fix manual: injeta callback de epoch no detector.py
Procura pelo método _train_worker e insere o callback do YOLO.
"""
import re
from pathlib import Path
from datetime import datetime

# ── Localiza o detector.py ───────────────────────────────────────────────────
CANDIDATES = [
    Path("app/projects/epi_check/engine/detector.py"),
    Path("app/epi_check/engine/detector.py"),
    Path("detector.py"),
]
DETECTOR = next((p for p in CANDIDATES if p.exists()), None)
if not DETECTOR:
    print("❌ detector.py não encontrado. Rode na raiz do projeto.")
    raise SystemExit(1)

print(f"✅ Encontrado: {DETECTOR}")

# ── Backup ───────────────────────────────────────────────────────────────────
ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
bak = DETECTOR.with_suffix(f".py.bak_{ts}")
bak.write_bytes(DETECTOR.read_bytes())
print(f"📦 Backup: {bak}")

src = DETECTOR.read_text(encoding="utf-8")

# ════════════════════════════════════════════════════════════════════════════
# PATCH 1 — substitui a chamada model.train(...) pela versão com callback
# Padrão genérico: captura o bloco results = model.train(...)
# ════════════════════════════════════════════════════════════════════════════

CALLBACK_BLOCK = '''
        # ── Callback de progresso por epoch ──────────────────────────────
        def _on_epoch_end(trainer):
            try:
                ep   = int(getattr(trainer, "epoch", 0)) + 1
                tot  = int(getattr(trainer, "epochs", train_params.get("epochs", 60)))
                met  = getattr(trainer, "metrics", {}) or {}
                loss = getattr(trainer, "loss_items", None)

                box_loss = cls_loss = None
                if loss is not None:
                    try:
                        import torch
                        l = loss.cpu().tolist() if hasattr(loss, "cpu") else list(loss)
                        box_loss = round(float(l[0]), 4) if len(l) > 0 else None
                        cls_loss = round(float(l[1]), 4) if len(l) > 1 else None
                    except Exception:
                        pass

                map50    = round(float(met.get("metrics/mAP50(B)",   met.get("val/mAP50",   0))), 4) if met else None
                map5095  = round(float(met.get("metrics/mAP50-95(B)", met.get("val/mAP50-95", 0))), 4) if met else None

                import time as _time
                elapsed = int(_time.time() - _train_start)
                eta     = int((elapsed / ep) * (tot - ep)) if ep > 0 else 0

                _train_status[company_id].update({
                    "epoch":           ep,
                    "total_epochs":    tot,
                    "elapsed_seconds": elapsed,
                    "eta_seconds":     eta,
                    "box_loss":        box_loss,
                    "cls_loss":        cls_loss,
                    "map50":           map50,
                    "map50_95":        map5095,
                })
                print(f"[TRAIN] epoch {ep}/{tot}  box={box_loss}  cls={cls_loss}  mAP50={map50}", flush=True)
            except Exception as _e:
                print(f"[TRAIN-CB] erro no callback: {_e}", flush=True)

        import time as _time
        _train_start = _time.time()
        model.add_callback("on_train_epoch_end", _on_epoch_end)
        # ─────────────────────────────────────────────────────────────────
'''

# Procura por: results = model.train(   OU   model.train(
# e insere o bloco de callback ANTES dessa linha
pattern = re.compile(
    r'(\s+)(results\s*=\s*model\.train\(|model\.train\()',
    re.MULTILINE
)

match = pattern.search(src)
if not match:
    print("❌ Não encontrei 'model.train(' no detector.py")
    print("   Verifique manualmente o arquivo.")
    raise SystemExit(1)

insert_pos = match.start()
indent     = match.group(1)          # preserva indentação original

# Ajusta indentação do bloco de callback para combinar com o arquivo
cb_indented = "\n".join(
    (indent + line if line.strip() else line)
    for line in CALLBACK_BLOCK.strip().splitlines()
)

new_src = src[:insert_pos] + "\n" + cb_indented + "\n" + src[insert_pos:]

# ════════════════════════════════════════════════════════════════════════════
# PATCH 2 — garante que total_epochs é salvo no status ao iniciar
# Procura por: _train_status[company_id] = {"status": "training", ...}
# e adiciona total_epochs se não existir
# ════════════════════════════════════════════════════════════════════════════
def patch_initial_status(text):
    # Procura pelo dict de inicialização do status de treino
    pat = re.compile(
        r'(_train_status\[company_id\]\s*=\s*\{[^}]*"status"\s*:\s*"training"[^}]*\})',
        re.DOTALL
    )
    m = pat.search(text)
    if not m:
        print("⚠️  Não encontrei o dict inicial de _train_status — pulando PATCH 2")
        return text

    block = m.group(1)
    if "total_epochs" in block:
        print("✅ PATCH 2: total_epochs já existe no status inicial")
        return text

    # Adiciona total_epochs logo após "epoch": 0
    new_block = re.sub(
        r'("epoch"\s*:\s*0)',
        r'\1,\n                    "total_epochs": train_params.get("epochs", 60)',
        block
    )
    return text.replace(block, new_block)

new_src = patch_initial_status(new_src)

# ── Salva ────────────────────────────────────────────────────────────────────
DETECTOR.write_text(new_src, encoding="utf-8")
print(f"\n✅ Patch aplicado com sucesso em {DETECTOR}")
print("   Reinicie o servidor:  python main.py")
print("   Depois inicie um novo treinamento e acompanhe o progresso.\n")
