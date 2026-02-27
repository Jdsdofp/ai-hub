#!/usr/bin/env python3
"""
SmartX Vision Platform — Fix Training Progress Display
=======================================================
Problema:
  - _train_worker() nunca atualiza o campo 'epoch' no _train_status
  - O YOLOv8 possui callbacks (on_train_epoch_end) que permitem
    capturar o progresso em tempo real

Fixes:
  1. detector.py  — adiciona callback YOLO para atualizar epoch, loss,
                    elapsed_seconds e eta_seconds em tempo real
  2. dashboard.html — garante que a barra de progresso + badges apareçam
                      corretamente (já estava ok, apenas pequeno ajuste
                      no campo 'elapsed' que ficava undefined)

Uso: python3 fix_train_progress.py  (executar na raiz do projeto)
"""

import os, shutil, re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
DETECTOR   = ROOT / "app/projects/epi_check/engine/detector.py"
DASHBOARD  = ROOT / "app/ui/templates/dashboard.html"
BACKUP     = datetime.now().strftime(".bak_%Y%m%d_%H%M%S")


def backup(p):
    if p.exists():
        dst = str(p) + BACKUP
        shutil.copy2(str(p), dst)
        print(f"  [backup] {p.name} → {p.name}{BACKUP}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DETECTOR — adiciona YOLO callback para epoch progress
# ─────────────────────────────────────────────────────────────────────────────

OLD_TRAIN_WORKER = '''    def _train_worker(self, company_id, base_model, epochs, batch_size, img_size, patience):
        """FIX BUG-02: Todos os paths agora usam .resolve() para evitar falhas com caminhos relativos no YOLO/Docker."""
        try:
            from ultralytics import YOLO
            # FIX BUG-02: .resolve() garante path absoluto — YOLO falha com relativos dentro do container
            yaml_path = str(CompanyData.epi(company_id, "dataset", "data.yaml").resolve())
            if not Path(yaml_path).exists():
                self._train_status[company_id] = {"status": "error",
                                                    "error": "data.yaml not found. Generate dataset first."}
                return
            self._train_status[company_id] = {"status": "training", "epoch": 0, "total_epochs": epochs}
            model = YOLO(base_model)
            # FIX BUG-02: project também precisa de path absoluto
            project_path = str(CompanyData.epi(company_id, "models").resolve())
            model.train(
                data=yaml_path, epochs=int(epochs), imgsz=int(img_size),
                batch=int(batch_size), patience=int(patience),
                name="epi_detector", project=project_path,
                exist_ok=True, plots=True,
            )
            best_path = str(CompanyData.epi(company_id, "models", "epi_detector", "weights", "best.pt").resolve())
            self._train_status[company_id] = {"status": "complete", "model_path": best_path}
            self._models.pop(self._model_key(company_id, "best"), None)
            logger.info(f"[Company {company_id}] Training complete: {best_path}")
        except Exception as e:
            logger.error(f"[Company {company_id}] Training error: {e}")
            self._train_status[company_id] = {"status": "error", "error": str(e)}'''

NEW_TRAIN_WORKER = '''    def _train_worker(self, company_id, base_model, epochs, batch_size, img_size, patience):
        """FIX BUG-02: Todos os paths agora usam .resolve() para evitar falhas com caminhos relativos no YOLO/Docker.
        FIX PROGRESS: Usa callbacks do YOLO para atualizar epoch/loss em tempo real."""
        try:
            from ultralytics import YOLO
            yaml_path = str(CompanyData.epi(company_id, "dataset", "data.yaml").resolve())
            if not Path(yaml_path).exists():
                self._train_status[company_id] = {"status": "error",
                                                    "error": "data.yaml not found. Generate dataset first."}
                return

            _t0 = time.time()
            self._train_status[company_id] = {
                "status": "training",
                "epoch": 0,
                "total_epochs": epochs,
                "box_loss": None,
                "cls_loss": None,
                "elapsed_seconds": 0,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            model = YOLO(base_model)
            project_path = str(CompanyData.epi(company_id, "models").resolve())

            # ── Callbacks para progresso em tempo real ──────────────────────
            def on_epoch_end(trainer):
                cur_epoch  = trainer.epoch + 1          # 1-based
                metrics    = trainer.metrics or {}
                loss_items = getattr(trainer, "loss_items", None)

                box_loss = cls_loss = None
                if loss_items is not None:
                    try:
                        lv = loss_items.cpu().numpy() if hasattr(loss_items, "cpu") else list(loss_items)
                        box_loss = round(float(lv[0]), 4) if len(lv) > 0 else None
                        cls_loss = round(float(lv[1]), 4) if len(lv) > 1 else None
                    except Exception:
                        pass

                elapsed = int(time.time() - _t0)
                eta = 0
                if cur_epoch > 0:
                    eta = int((elapsed / cur_epoch) * (epochs - cur_epoch))

                self._train_status[company_id].update({
                    "status":          "training",
                    "epoch":           cur_epoch,
                    "total_epochs":    epochs,
                    "box_loss":        box_loss,
                    "cls_loss":        cls_loss,
                    "elapsed_seconds": elapsed,
                    "eta_seconds":     eta,
                    "map50":           round(float(metrics.get("metrics/mAP50(B)", 0) or 0), 4),
                    "map50_95":        round(float(metrics.get("metrics/mAP50-95(B)", 0) or 0), 4),
                })
                logger.debug(f"[Company {company_id}] Epoch {cur_epoch}/{epochs} | "
                             f"box={box_loss} cls={cls_loss} | "
                             f"elapsed={elapsed}s eta={eta}s")

            model.add_callback("on_train_epoch_end", on_epoch_end)
            # ────────────────────────────────────────────────────────────────

            model.train(
                data=yaml_path, epochs=int(epochs), imgsz=int(img_size),
                batch=int(batch_size), patience=int(patience),
                name="epi_detector", project=project_path,
                exist_ok=True, plots=True,
            )

            best_path = str(CompanyData.epi(company_id, "models", "epi_detector", "weights", "best.pt").resolve())
            elapsed_total = int(time.time() - _t0)
            self._train_status[company_id] = {
                "status":          "complete",
                "model_path":      best_path,
                "elapsed_seconds": elapsed_total,
                "total_epochs":    epochs,
            }
            self._models.pop(self._model_key(company_id, "best"), None)
            logger.info(f"[Company {company_id}] Training complete in {elapsed_total}s: {best_path}")

        except Exception as e:
            logger.error(f"[Company {company_id}] Training error: {e}")
            self._train_status[company_id] = {"status": "error", "error": str(e)}'''


def patch_detector():
    print("\n[1/2] Patching detector.py ...")
    backup(DETECTOR)
    content = DETECTOR.read_text(encoding="utf-8")

    if "on_train_epoch_end" in content:
        print("  [SKIP] Callback já existe em detector.py")
        return

    if OLD_TRAIN_WORKER in content:
        content = content.replace(OLD_TRAIN_WORKER, NEW_TRAIN_WORKER)
        print("  [OK] _train_worker() atualizado com callbacks de progresso")
    else:
        # fallback: substituição parcial via regex
        pattern = re.compile(
            r'def _train_worker\(self.*?except Exception as e:\s*'
            r'logger\.error\(f"\[Company \{company_id\}\] Training error: \{e\}"\)\s*'
            r'self\._train_status\[company_id\] = \{"status": "error", "error": str\(e\)\}',
            re.DOTALL
        )
        m = pattern.search(content)
        if m:
            content = content[:m.start()] + NEW_TRAIN_WORKER.lstrip() + content[m.end():]
            print("  [OK] _train_worker() substituído via regex fallback")
        else:
            print("  [ERROR] Não foi possível localizar _train_worker(). Edite manualmente.")
            return

    DETECTOR.write_text(content, encoding="utf-8")
    print("  [DONE] detector.py salvo")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DASHBOARD — melhorar exibição de progresso
#    O JS já estava quase certo; ajuste principal é no formatDuration e
#    na exibição de ETA + mAP50
# ─────────────────────────────────────────────────────────────────────────────

OLD_POLL_TRAINING = '''  if(r.status==='training'){
    setTrainLoading(true);sa.style.display='flex';
    const pct=r.total_epochs?Math.round((r.epoch/r.total_epochs)*100):5;
    const elapsed=r.elapsed_seconds?formatDuration(r.elapsed_seconds):'';
    sa.innerHTML='<div style="width:100%"><div class="train-epoch-row"><span class="train-epoch-label">EPOCH '+(r.epoch??'?')+' / '+(r.total_epochs??'?')+'</span><span class="train-epoch-pct">'+pct+'%</span></div><div class="progress-bar"><div class="progress-fill training-pulse" style="width:'+pct+'%"></div></div>'+(elapsed?'<div class="train-meta">⏱ '+elapsed+' elapsed</div>':'')+(r.box_loss?'<div class="train-meta" style="margin-top:2px">loss → box: '+r.box_loss+' · cls: '+r.cls_loss+'</div>':'')+'</div>';
    dl.style.display='none';_trainPollTimer=setTimeout(pollTrainStatus,4000);'''

NEW_POLL_TRAINING = '''  if(r.status==='training'){
    setTrainLoading(true);sa.style.display='flex';
    const epoch      = r.epoch        || 0;
    const totalEpochs= r.total_epochs || 1;
    const pct        = Math.max(2, Math.round((epoch / totalEpochs) * 100));
    const elapsed    = r.elapsed_seconds ? formatDuration(r.elapsed_seconds) : '';
    const eta        = r.eta_seconds     ? 'ETA ' + formatDuration(r.eta_seconds) : '';
    const lossInfo   = (r.box_loss != null)
      ? '<div class="train-meta" style="margin-top:4px">loss → box: <strong>'+r.box_loss+'</strong> · cls: <strong>'+(r.cls_loss??'—')+'</strong>'+(r.map50?' · mAP50: <strong>'+r.map50+'</strong>':'')+'</div>'
      : '';
    const timeRow    = (elapsed || eta)
      ? '<div class="train-meta" style="margin-top:2px">⏱ '+elapsed+(elapsed&&eta?' · ':'')+eta+'</div>'
      : '';
    sa.innerHTML=
      '<div style="width:100%">'+
        '<div class="train-epoch-row">'+
          '<span class="train-epoch-label">EPOCH '+epoch+' / '+totalEpochs+'</span>'+
          '<span class="train-epoch-pct">'+pct+'%</span>'+
        '</div>'+
        '<div class="progress-bar">'+
          '<div class="progress-fill training-pulse" style="width:'+pct+'%"></div>'+
        '</div>'+
        timeRow+
        lossInfo+
      '</div>';
    dl.style.display='none';_trainPollTimer=setTimeout(pollTrainStatus,3000);'''


def patch_dashboard():
    print("\n[2/2] Patching dashboard.html ...")
    backup(DASHBOARD)
    html = DASHBOARD.read_text(encoding="utf-8")

    if "eta_seconds" in html:
        print("  [SKIP] Dashboard já tem suporte a eta_seconds")
    elif OLD_POLL_TRAINING in html:
        html = html.replace(OLD_POLL_TRAINING, NEW_POLL_TRAINING)
        print("  [OK] pollTrainStatus() atualizado com epoch counter + ETA + mAP50")
    else:
        # regex fallback — localiza o bloco training e substitui apenas o innerHTML
        pattern = re.compile(
            r"(if\(r\.status===.training.\)\{.*?setTrainLoading\(true\);sa\.style\.display='flex';)"
            r".*?"
            r"(dl\.style\.display='none';_trainPollTimer=setTimeout\(pollTrainStatus,\d+\);)",
            re.DOTALL
        )
        m = pattern.search(html)
        if m:
            replacement = (
                "if(r.status==='training'){\n"
                "    setTrainLoading(true);sa.style.display='flex';\n"
                + NEW_POLL_TRAINING.split("setTrainLoading(true);sa.style.display='flex';", 1)[1]
            )
            html = html[:m.start()] + replacement + html[m.end():]
            print("  [OK] pollTrainStatus() atualizado via regex fallback")
        else:
            print("  [WARN] Bloco training não encontrado. Verifique manualmente.")

    DASHBOARD.write_text(html, encoding="utf-8")
    print("  [DONE] dashboard.html salvo")


def verify():
    print("\n" + "="*60)
    print("VERIFICAÇÃO")
    print("="*60)
    ok = True

    det = DETECTOR.read_text(encoding="utf-8") if DETECTOR.exists() else ""
    dash = DASHBOARD.read_text(encoding="utf-8") if DASHBOARD.exists() else ""

    checks = [
        ("detector: on_train_epoch_end callback",  "on_train_epoch_end" in det),
        ("detector: elapsed_seconds atualizado",   "elapsed_seconds" in det),
        ("detector: eta_seconds calculado",        "eta_seconds" in det),
        ("detector: box_loss capturado",           "box_loss" in det),
        ("dashboard: epoch counter no HTML",       "epoch / totalEpochs" in dash or "epoch / total" in dash.lower()),
        ("dashboard: ETA exibido",                 "eta_seconds" in dash or "ETA" in dash),
    ]
    for name, passed in checks:
        if not passed: ok = False
        print(f"  {'[OK]' if passed else '[FAIL]'} {name}")
    return ok


def main():
    print("="*60)
    print("SmartX Vision — Fix Training Progress Display")
    print("="*60)

    for p, n in [(DETECTOR, "detector.py"), (DASHBOARD, "dashboard.html")]:
        if not p.exists():
            print(f"[ERROR] {n} não encontrado: {p}")
            print("Execute na raiz do projeto SmartX Vision.")
            return

    patch_detector()
    patch_dashboard()

    ok = verify()
    print("\n" + "="*60)
    print("[SUCCESS] Patch aplicado!" if ok else "[WARNING] Revisar itens marcados [FAIL]")
    print("="*60)
    print("""
O que foi corrigido:

  detector.py (_train_worker):
    ✓ Callback on_train_epoch_end registrado no modelo YOLO
    ✓ _train_status atualizado a cada época com:
        epoch, total_epochs   → progresso "3 / 60"
        elapsed_seconds       → tempo decorrido
        eta_seconds           → tempo estimado restante
        box_loss, cls_loss    → losses da época
        map50, map50_95       → métricas de validação

  dashboard.html (pollTrainStatus):
    ✓ Barra de progresso mostra % real baseada em epoch/total_epochs
    ✓ Contador "EPOCH X / Y" atualizado em tempo real
    ✓ Linha de tempo "⏱ 2m 30s · ETA 12m 10s"
    ✓ Linha de loss "loss → box: 1.23 · cls: 0.45 · mAP50: 0.82"
    ✓ Poll a cada 3s durante treinamento (era 4s)

Reinicie o servidor para aplicar:
  python main.py
""")


if __name__ == "__main__":
    main()