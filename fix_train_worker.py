#!/usr/bin/env python3
"""
Substitui o _train_worker no detector.py com versão que registra
callback de epoch corretamente dentro da thread.
"""
from pathlib import Path
from datetime import datetime

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

# Backup
ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
bak = DETECTOR.with_suffix(f".py.bak_{ts}")
bak.write_bytes(DETECTOR.read_bytes())
print(f"📦 Backup: {bak}")

src = DETECTOR.read_text(encoding="utf-8")

# ── Trecho antigo (método completo) ──────────────────────────────────────────
OLD = '''    def _train_worker(self, company_id, base_model, epochs, batch_size, img_size, patience):
        """FIX BUG-02: Todos os paths agora usam .resolve() para evitar falhas com caminhos relativos no YOLO/Docker."""
        try:
            from ultralytics import YOLO
            # FIX BUG-02: .resolve() garante path absoluto ÔÇö YOLO falha com relativos dentro do container
            yaml_path = str(CompanyData.epi(company_id, "dataset", "data.yaml").resolve())
            if not Path(yaml_path).exists():
                self._train_status[company_id] = {"status": "error",
                                                    "error": "data.yaml not found. Generate dataset first."}
                return
            self._train_status[company_id] = {"status": "training", "epoch": 0, "total_epochs": epochs}
            model = YOLO(base_model)
            # FIX BUG-02: project tamb├®m precisa de path absoluto
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

# ── Trecho novo com callback ──────────────────────────────────────────────────
NEW = '''    def _train_worker(self, company_id, base_model, epochs, batch_size, img_size, patience):
        """FIX BUG-02 + CALLBACK: paths absolutos + progresso por epoch via callback."""
        try:
            from ultralytics import YOLO
            yaml_path = str(CompanyData.epi(company_id, "dataset", "data.yaml").resolve())
            if not Path(yaml_path).exists():
                self._train_status[company_id] = {"status": "error",
                                                    "error": "data.yaml not found. Generate dataset first."}
                return

            self._train_status[company_id] = {
                "status": "training",
                "epoch": 0,
                "total_epochs": int(epochs),
                "elapsed_seconds": 0,
                "eta_seconds": 0,
                "box_loss": None,
                "cls_loss": None,
                "map50": None,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            model = YOLO(base_model)
            project_path = str(CompanyData.epi(company_id, "models").resolve())
            train_start = time.time()
            status_ref = self._train_status  # referência direta ao dict do objeto

            def on_epoch_end(trainer):
                try:
                    ep  = int(getattr(trainer, "epoch", 0)) + 1
                    tot = int(epochs)
                    met = getattr(trainer, "metrics", {}) or {}
                    loss_items = getattr(trainer, "loss_items", None)

                    box_loss = cls_loss = None
                    if loss_items is not None:
                        try:
                            import torch as _torch
                            ll = loss_items.cpu().tolist() if hasattr(loss_items, "cpu") else list(loss_items)
                            box_loss = round(float(ll[0]), 4) if len(ll) > 0 else None
                            cls_loss = round(float(ll[1]), 4) if len(ll) > 1 else None
                        except Exception:
                            pass

                    map50 = None
                    for k in ("metrics/mAP50(B)", "val/mAP50", "mAP50"):
                        if k in met:
                            try:
                                map50 = round(float(met[k]), 4)
                            except Exception:
                                pass
                            break

                    elapsed = int(time.time() - train_start)
                    eta = int((elapsed / ep) * (tot - ep)) if ep > 0 else 0

                    status_ref[company_id].update({
                        "epoch":           ep,
                        "total_epochs":    tot,
                        "elapsed_seconds": elapsed,
                        "eta_seconds":     eta,
                        "box_loss":        box_loss,
                        "cls_loss":        cls_loss,
                        "map50":           map50,
                    })
                    logger.debug(f"[Company {company_id}] Epoch {ep}/{tot} | box={box_loss} cls={cls_loss} | elapsed={elapsed}s eta={eta}s")
                except Exception as cb_err:
                    logger.warning(f"[Company {company_id}] Callback error: {cb_err}")

            model.add_callback("on_train_epoch_end", on_epoch_end)

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

if OLD not in src:
    print("❌ Trecho original não encontrado exatamente.")
    print("   Verifique se o arquivo foi modificado por um patch anterior.")
    print("   Restaure o backup original e rode novamente.")
    raise SystemExit(1)

new_src = src.replace(OLD, NEW, 1)
DETECTOR.write_text(new_src, encoding="utf-8")
print(f"\n✅ _train_worker atualizado com callback de epoch!")
print("   Reinicie o servidor:  python main.py")
print("   Inicie um NOVO treinamento para testar.\n")
