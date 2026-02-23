#!/usr/bin/env python3
"""
SmartX Vision Platform — Dashboard UI Patch v3.1
=================================================
Corrige bugs de UI no dashboard.html:

BUG-12: .tabs-wrapper { top: 56px } mas .header tem height: 125px
         → Tabs ficam escondidas atrás do header no scroll
         Fix: top: 125px

BUG-13: annSave() não verifica se annCurrentFile é válido antes de salvar

BUG-14: switchCompany() não recarrega a aba ativa atual
         → Trocar de company na aba Faces não recarrega as pessoas

BUG-15: pollTrainStatus() setTimeout recursivo sem controle
         → Polling continua mesmo após sair da aba Train, acumulando calls

BUG-16 (JS lado): stream_feed usa generator síncrono — já corrigido em routes.py,
         mas o frontend também pode melhorar tratamento de erro de reconexão
         
BUG (extra): loadDashboard() não trata erro de rede — shimmer fica indefinidamente
"""

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


def find_dashboard():
    candidates = [
        Path("app/ui/templates/dashboard.html"),
        Path(__file__).parent / "app/ui/templates/dashboard.html",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def patch(html: str) -> str:
    changes = []

    # =========================================================
    # FIX BUG-12: tabs sticky top deve ser igual à altura do header (125px)
    # =========================================================
    old_tabs_top = ".tabs-wrapper {\n  position: sticky;\n  top: 56px;"
    new_tabs_top = ".tabs-wrapper {\n  position: sticky;\n  top: 125px; /* FIX BUG-12: igual à height do .header */"
    if old_tabs_top in html:
        html = html.replace(old_tabs_top, new_tabs_top)
        changes.append("BUG-12: tabs-wrapper top: 56px → 125px")
    else:
        # Tenta via regex
        html, n = re.subn(
            r'(\.tabs-wrapper\s*\{[^}]*?top\s*:\s*)56px',
            r'\g<1>125px /* FIX BUG-12 */',
            html
        )
        if n:
            changes.append("BUG-12: tabs-wrapper top corrigido via regex")

    # =========================================================
    # FIX BUG-15: pollTrainStatus() controle de intervalo para evitar múltiplos timers
    # Substitui o setTimeout recursivo por um mecanismo com flag de controle
    # =========================================================
    old_poll = (
        "async function pollTrainStatus(){"
        "const r=await(await fetch(API('/train/status'))).json();"
        "const b=document.getElementById('train-status');"
        "if(r.status==='training'){"
        "b.innerHTML='<div class=\"progress-bar\"><div class=\"progress-fill\" style=\"width:'+((r.epoch/r.total_epochs*100)||5)+'%\"></div></div>"
        "<div style=\"margin-top:8px;font-size:12px;color:var(--text-muted);font-family:Space Mono,monospace\">Epoch '+(r.epoch||'?')+'/'+(r.total_epochs||'?')+'</div>';"
        "setTimeout(pollTrainStatus,5000);"
        "}else if(r.status==='complete'){"
        "b.innerHTML='<span class=\"badge badge-ok\">✓ Complete</span>"
        "<div style=\"margin-top:8px;font-size:12px;color:var(--text-muted)\">'+(r.model_path||'')+'</div>';"
        "}else if(r.status==='error'){"
        "b.innerHTML='<span class=\"badge badge-err\">'+r.error+'</span>';"
        "}else{"
        "b.innerHTML='<span class=\"badge badge-off\">'+(r.status||'idle')+'</span>';"
        "}}"
    )

    new_poll = (
        "let _trainPollTimer=null;\n"
        "function stopTrainPoll(){if(_trainPollTimer){clearTimeout(_trainPollTimer);_trainPollTimer=null;}}\n"
        "async function pollTrainStatus(){\n"
        "  stopTrainPoll(); // FIX BUG-15: cancela timer anterior antes de criar novo\n"
        "  try{\n"
        "    const r=await(await fetch(API('/train/status'))).json();\n"
        "    const b=document.getElementById('train-status');\n"
        "    if(!b)return; // painel pode ter sido desmontado\n"
        "    if(r.status==='training'){\n"
        "      const pct=r.epoch&&r.total_epochs?Math.round(r.epoch/r.total_epochs*100):5;\n"
        "      b.innerHTML='<div class=\"progress-bar\"><div class=\"progress-fill\" style=\"width:'+pct+'%\"></div></div>"
        "<div style=\"margin-top:8px;font-size:12px;color:var(--text-muted);font-family:Space Mono,monospace\">Epoch '+(r.epoch||'?')+'/'+(r.total_epochs||'?')+'</div>';\n"
        "      _trainPollTimer=setTimeout(pollTrainStatus,5000); // FIX BUG-15: salva referência\n"
        "    }else if(r.status==='complete'){\n"
        "      b.innerHTML='<span class=\"badge badge-ok\">✓ Complete</span>"
        "<div style=\"margin-top:8px;font-size:12px;color:var(--text-muted)\">'+(r.model_path||'')+'</div>';\n"
        "    }else if(r.status==='error'){\n"
        "      b.innerHTML='<span class=\"badge badge-err\">'+(r.error||'Unknown error')+'</span>';\n"
        "    }else{\n"
        "      b.innerHTML='<span class=\"badge badge-off\">'+(r.status||'idle')+'</span>';\n"
        "    }\n"
        "  }catch(e){\n"
        "    console.warn('pollTrainStatus error:',e);\n"
        "  }\n"
        "}"
    )

    if old_poll in html:
        html = html.replace(old_poll, new_poll)
        changes.append("BUG-15: pollTrainStatus() timer com controle de cancelamento")
    else:
        changes.append("BUG-15: SKIP (padrão não encontrado — verifique manualmente)")

    # =========================================================
    # FIX BUG-14: switchCompany() recarrega a aba atualmente ativa
    # =========================================================
    old_switch = "function switchCompany(){CID=parseInt(document.getElementById('companySelect').value);loadDashboard();toast('Switched to Company '+CID)}"
    new_switch = (
        "function switchCompany(){\n"
        "  CID=parseInt(document.getElementById('companySelect').value);\n"
        "  toast('Switched to Company '+CID);\n"
        "  // FIX BUG-14: recarrega dados da aba atualmente ativa, não só o dashboard\n"
        "  const active=document.querySelector('.panel.active');\n"
        "  if(!active)return loadDashboard();\n"
        "  const panelId=active.id.replace('panel-','');\n"
        "  const reloadMap={\n"
        "    dashboard: loadDashboard,\n"
        "    config: loadPPEConfig,\n"
        "    annotate: loadAnnotationImages,\n"
        "    faces: loadPeople,\n"
        "    train: pollTrainStatus,\n"
        "  };\n"
        "  const fn=reloadMap[panelId];\n"
        "  if(fn)fn(); else loadDashboard();\n"
        "}"
    )

    if old_switch in html:
        html = html.replace(old_switch, new_switch)
        changes.append("BUG-14: switchCompany() agora recarrega aba ativa")
    else:
        changes.append("BUG-14: SKIP (padrão não encontrado — verifique manualmente)")

    # =========================================================
    # FIX BUG-13: annSave() valida annCurrentFile antes de salvar
    # =========================================================
    old_annsave = "async function annSave(){if(!annCurrentFile)return toast('No image selected','error');"
    new_annsave = (
        "async function annSave(){\n"
        "  // FIX BUG-13: valida que imagem está carregada E que o canvas existe\n"
        "  if(!annCurrentFile||annCurrentFile.trim()==='')return toast('No image selected','error');\n"
        "  const cv=document.getElementById('ann-cv');\n"
        "  if(!cv)return toast('Load an image first','error');\n"
        "  if(!annAnnotations.length){toast('No annotations to save','error');return;}\n"
        "  "
    )

    if old_annsave in html:
        html = html.replace(old_annsave, new_annsave)
        changes.append("BUG-13: annSave() validação de canvas e annotations")
    else:
        changes.append("BUG-13: SKIP (padrão não encontrado — verifique manualmente)")

    # =========================================================
    # FIX BUG-15 (complemento): showPanel() cancela timer de train ao trocar aba
    # =========================================================
    old_showpanel = (
        "function showPanel(n,el){"
        "document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));"
        "document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));"
        "document.getElementById('panel-'+n).classList.add('active');"
        "if(el)el.classList.add('active');"
        "if(n==='dashboard')loadDashboard();"
        "if(n==='config')loadPPEConfig();"
        "if(n==='annotate')loadAnnotationImages();"
        "if(n==='faces')loadPeople();"
        "if(n==='train')pollTrainStatus();"
        "}"
    )
    new_showpanel = (
        "function showPanel(n,el){\n"
        "  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));\n"
        "  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));\n"
        "  document.getElementById('panel-'+n).classList.add('active');\n"
        "  if(el)el.classList.add('active');\n"
        "  // FIX BUG-15: cancela poll de training se sair da aba\n"
        "  if(n!=='train' && typeof stopTrainPoll==='function')stopTrainPoll();\n"
        "  if(n==='dashboard')loadDashboard();\n"
        "  if(n==='config')loadPPEConfig();\n"
        "  if(n==='annotate')loadAnnotationImages();\n"
        "  if(n==='faces')loadPeople();\n"
        "  if(n==='train')pollTrainStatus();\n"
        "}"
    )

    if old_showpanel in html:
        html = html.replace(old_showpanel, new_showpanel)
        changes.append("BUG-15 (extra): showPanel() cancela timer ao trocar de aba")
    else:
        changes.append("BUG-15 (extra): SKIP showPanel")

    return html, changes


def main():
    path = find_dashboard()
    if not path:
        print("[ERROR] dashboard.html não encontrado. Execute do diretório raiz do projeto.")
        sys.exit(1)

    print(f"Patching: {path.resolve()}")
    bak = str(path) + datetime.now().strftime(".bak_%Y%m%d_%H%M%S")
    shutil.copy2(str(path), bak)
    print(f"Backup: {bak}")

    html = path.read_text(encoding="utf-8")
    html_patched, changes = patch(html)

    if html == html_patched:
        print("\n[WARN] Nenhuma alteração detectada. O arquivo pode já estar corrigido.")
    else:
        path.write_text(html_patched, encoding="utf-8")
        print(f"\n[OK] {len([c for c in changes if 'SKIP' not in c])} patches aplicados:")
        for c in changes:
            status = "✓" if "SKIP" not in c else "⚠"
            print(f"  {status} {c}")

    print("\nReinicie o servidor para aplicar as mudanças:")
    print("  python main.py")


if __name__ == "__main__":
    main()
