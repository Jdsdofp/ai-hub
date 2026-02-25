#!/usr/bin/env python3
"""
SmartX Vision Platform — Browser Camera Patch
==============================================
Adiciona suporte à câmera do dispositivo do usuário (browser) no Live Stream.

Mudanças:
  1. routes.py  — novo endpoint POST /api/v1/epi/detect/frame
                  Recebe JPEG bytes, detecta EPI, retorna imagem anotada em base64
  2. dashboard.html — aba Live Stream ganha botão "📷 Browser Camera"
                      Usa MediaDevices.getUserMedia + canvas para capturar frames
                      e enviar ao servidor a cada N ms

Uso: python3 add_browser_camera.py
"""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
ROUTES_PATH = ROOT / "app/projects/epi_check/api/routes.py"
DASHBOARD_PATH = ROOT / "app/ui/templates/dashboard.html"

BACKUP = datetime.now().strftime(".bak_%Y%m%d_%H%M%S")


def backup(p: Path):
    if p.exists():
        dst = str(p) + BACKUP
        shutil.copy2(str(p), dst)
        print(f"  [backup] {p.name} → {p.name}{BACKUP}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. BACKEND — novo endpoint /detect/frame
# ─────────────────────────────────────────────────────────────────────────────
FRAME_ENDPOINT = '''
# ======================================================================
# DETECTION — Single Frame (Browser Camera)
# ======================================================================
@router.post("/detect/frame", tags=["Detection — Image"],
    summary="Detect PPE from raw JPEG bytes (browser camera frames)")
async def detect_frame(
    file: UploadFile = File(...),
    model_name: str = Form("best"),
    confidence: float = Form(0.4),
    detect_faces: bool = Form(False),
    face_threshold: float = Form(0.45),
    annotate: bool = Form(True),
    company_id: int = Depends(get_ui_company),
):
    """
    Endpoint otimizado para receber frames de câmera do browser.
    Retorna JSON com resultado + imagem anotada em base64 (opcional).
    Menor overhead que /detect/upload pois não salva snapshot em disco.
    """
    try:
        data = await file.read()
        if not data:
            raise HTTPException(400, detail="Empty frame")
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, detail="Invalid frame — could not decode")

        if annotate:
            annotated, result = epi_engine.detect_and_annotate(
                company_id, img, model_name, confidence, detect_faces, face_threshold,
            )
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            result["annotated_base64"] = base64.b64encode(buf).decode()
        else:
            result = epi_engine.detect_image(
                company_id, img, model_name, confidence, detect_faces, face_threshold,
            )

        if not result["compliant"]:
            await mqtt_client.publish_alert(company_id, "EPI_NON_COMPLIANT", {
                "missing": result["missing"],
                "source": "browser_camera",
                "faces": result.get("faces", []),
            })

        return _sanitize_result(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] detect_frame error: {e}")
        raise HTTPException(500, detail=f"Frame detection failed: {str(e)}")
'''


def patch_routes():
    print("\n[1/2] Patching routes.py ...")
    backup(ROUTES_PATH)
    content = ROUTES_PATH.read_text()

    if "/detect/frame" in content:
        print("  [SKIP] /detect/frame já existe")
        return

    # Inserir antes do bloco de Face Recognition
    marker = "# ======================================================================\n# FACE RECOGNITION"
    if marker in content:
        content = content.replace(marker, FRAME_ENDPOINT + "\n" + marker)
        print("  [OK] Endpoint /detect/frame adicionado")
    else:
        # fallback: append antes do último bloco
        content += FRAME_ENDPOINT
        print("  [OK] Endpoint /detect/frame adicionado (fallback)")

    ROUTES_PATH.write_text(content)
    print("  [DONE] routes.py salvo")


# ─────────────────────────────────────────────────────────────────────────────
# 2. FRONTEND — Browser Camera no Live Stream
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_CSS = """
/* ==================== BROWSER CAMERA CSS ==================== */
.cam-container {
  position: relative;
  background: #0a0a0f;
  border-radius: 12px;
  overflow: hidden;
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--border);
}

.cam-container video,
.cam-container canvas,
.cam-container img.cam-result {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 0;
}

.cam-overlay {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.cam-status-bar {
  position: absolute;
  top: 10px;
  left: 10px;
  right: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
  pointer-events: none;
}

.cam-fps-badge {
  background: rgba(0,0,0,0.65);
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 6px;
  padding: 3px 8px;
  font-family: 'Space Mono', monospace;
  font-size: 11px;
  color: #a0aec0;
  backdrop-filter: blur(4px);
}

.cam-fps-badge.compliant  { color: #68d391; border-color: rgba(104,211,145,0.3); }
.cam-fps-badge.noncompliant { color: #fc8181; border-color: rgba(252,129,129,0.3); }

.cam-placeholder {
  color: #4a5568;
  font-size: 13px;
  font-family: 'Space Mono', monospace;
  text-align: center;
  padding: 40px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.cam-placeholder::before {
  content: '📷';
  font-size: 36px;
  opacity: 0.35;
}

.cam-selector {
  position: absolute;
  top: 10px;
  right: 10px;
  pointer-events: all;
  background: rgba(0,0,0,0.6);
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 6px;
  padding: 4px 8px;
  color: #e2e8f0;
  font-size: 12px;
  font-family: 'Space Mono', monospace;
  backdrop-filter: blur(4px);
  cursor: pointer;
  max-width: 180px;
  display: none;
}

.cam-selector.visible { display: block; }

.cam-mirror-btn {
  position: absolute;
  bottom: 10px;
  right: 10px;
  pointer-events: all;
  background: rgba(0,0,0,0.6);
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 6px;
  padding: 5px 9px;
  color: #a0aec0;
  font-size: 12px;
  cursor: pointer;
  display: none;
  backdrop-filter: blur(4px);
  transition: color 0.2s;
}
.cam-mirror-btn:hover { color: #fff; }
.cam-mirror-btn.visible { display: block; }

/* Scan line animation when active */
.cam-scanline {
  position: absolute;
  left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, rgba(26,115,232,0.6), transparent);
  animation: cam-scan 3s linear infinite;
  pointer-events: none;
  display: none;
}
.cam-scanning .cam-scanline { display: block; }

@keyframes cam-scan {
  0%   { top: 0%; }
  100% { top: 100%; }
}
/* ============================================================ */
"""

CAMERA_HTML_TAB = """
    <!-- BROWSER CAMERA TAB (dentro do painel stream) -->
    <div class="card" style="margin-top:16px">
      <div class="card-header">
        <div class="card-icon">📷</div>
        <div class="card-title">Browser Camera — Detecção em Tempo Real</div>
        <div style="margin-left:auto;display:flex;gap:8px;align-items:center">
          <span class="badge badge-off" id="cam-badge">INATIVO</span>
        </div>
      </div>

      <!-- CONTROLS ROW -->
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr auto;gap:12px;margin-bottom:14px;align-items:end">
        <div>
          <label>Confidence</label>
          <div class="range-row">
            <input type="range" id="cam-conf" min="0.1" max="0.9" step="0.05" value="0.4"
              oninput="document.getElementById('cam-conf-val').textContent=this.value">
            <span class="range-val" id="cam-conf-val">0.4</span>
          </div>
        </div>
        <div>
          <label>Intervalo (ms)</label>
          <select id="cam-interval" style="margin-bottom:0">
            <option value="100">100ms — ~10fps</option>
            <option value="200" selected>200ms — ~5fps</option>
            <option value="500">500ms — ~2fps</option>
            <option value="1000">1000ms — ~1fps</option>
          </select>
        </div>
        <div style="display:flex;flex-direction:column;gap:6px">
          <div class="checkbox-row" style="margin:0">
            <input type="checkbox" id="cam-faces" checked>
            <label for="cam-faces" style="font-size:12px">Face Recognition</label>
          </div>
          <div class="checkbox-row" style="margin:0">
            <input type="checkbox" id="cam-mirror" checked onchange="camToggleMirror()">
            <label for="cam-mirror" style="font-size:12px">Espelhar imagem</label>
          </div>
        </div>
        <div style="display:flex;flex-direction:column;gap:8px">
          <button class="btn btn-success" id="cam-start-btn" onclick="camStart()">📷 Iniciar Câmera</button>
          <button class="btn btn-danger" id="cam-stop-btn" onclick="camStop()" style="display:none">◼ Parar</button>
        </div>
      </div>

      <!-- CAMERA VIEW -->
      <div class="grid-2" style="gap:16px">
        <div class="cam-container" id="cam-container">
          <div class="cam-placeholder" id="cam-placeholder">Clique em "Iniciar Câmera" para<br>usar a câmera do seu dispositivo</div>
          <!-- video element (preview local) -->
          <video id="cam-video" autoplay playsinline muted style="display:none"></video>
          <!-- canvas oculto para capturar frames -->
          <canvas id="cam-canvas" style="display:none"></canvas>
          <!-- resultado anotado -->
          <img id="cam-result-img" class="cam-result" style="display:none" alt="resultado">
          <!-- overlay / status -->
          <div class="cam-overlay cam-scanline" id="cam-scanline"></div>
          <div class="cam-status-bar" id="cam-status-bar" style="display:none">
            <span class="cam-fps-badge" id="cam-fps-badge">0 FPS</span>
            <span class="cam-fps-badge" id="cam-compliance-badge">—</span>
            <span class="cam-fps-badge" id="cam-ms-badge">— ms</span>
          </div>
          <!-- device selector -->
          <select class="cam-selector" id="cam-device-sel" onchange="camSwitchDevice()"></select>
          <!-- mirror btn -->
          <button class="cam-mirror-btn" id="cam-mirror-btn" onclick="document.getElementById('cam-mirror').click()">⟺ Espelhar</button>
        </div>

        <!-- RESULT SIDEBAR -->
        <div>
          <div class="card" style="margin-bottom:0;height:100%">
            <div class="card-header" style="padding-bottom:10px;margin-bottom:12px">
              <div class="card-icon">◎</div>
              <div class="card-title">Último Resultado</div>
            </div>
            <div id="cam-result-detail" class="empty-state" data-icon="📷">
              Aguardando detecção...
            </div>
          </div>
        </div>
      </div>
    </div>"""

CAMERA_JS = """
// ===================== BROWSER CAMERA JS =====================
let _camStream = null;
let _camTimer = null;
let _camMirror = true;
let _camSending = false;
let _camFrameCount = 0;
let _camFpsTs = performance.now();
let _camFps = 0;

async function camStart() {
  try {
    // Pede permissão de câmera
    const constraints = {
      video: {
        width:  { ideal: 1280 },
        height: { ideal:  720 },
        facingMode: 'environment', // câmera traseira em mobile; fallback para frontal
      },
      audio: false
    };

    // Tenta câmera traseira; se falhar, tenta qualquer câmera
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch(_) {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }

    _camStream = stream;

    // Popula seletor de dispositivos
    await camPopulateDevices();

    const video = document.getElementById('cam-video');
    video.srcObject = stream;
    video.style.display = 'block';
    document.getElementById('cam-placeholder').style.display = 'none';
    document.getElementById('cam-result-img').style.display = 'none';
    document.getElementById('cam-status-bar').style.display = 'flex';
    document.getElementById('cam-start-btn').style.display = 'none';
    document.getElementById('cam-stop-btn').style.display = 'inline-flex';
    document.getElementById('cam-badge').className = 'badge badge-warn';
    document.getElementById('cam-badge').textContent = 'ATIVO';
    document.getElementById('cam-mirror-btn').classList.add('visible');

    camApplyMirror();

    // Container scanning animation
    document.querySelector('.cam-overlay.cam-scanline')?.parentElement?.classList.add('cam-scanning');

    // Start sending frames
    const interval = parseInt(document.getElementById('cam-interval').value);
    _camTimer = setInterval(camSendFrame, interval);

    toast('📷 Câmera iniciada');
  } catch (e) {
    if (e.name === 'NotAllowedError') {
      toast('Permissão de câmera negada. Verifique as configurações do browser.', 'error');
    } else if (e.name === 'NotFoundError') {
      toast('Nenhuma câmera encontrada neste dispositivo.', 'error');
    } else {
      toast('Erro ao acessar câmera: ' + e.message, 'error');
    }
    console.error('Camera error:', e);
  }
}

function camStop() {
  if (_camTimer) { clearInterval(_camTimer); _camTimer = null; }
  if (_camStream) {
    _camStream.getTracks().forEach(t => t.stop());
    _camStream = null;
  }
  const video = document.getElementById('cam-video');
  video.srcObject = null;
  video.style.display = 'none';

  document.getElementById('cam-placeholder').style.display = 'flex';
  document.getElementById('cam-result-img').style.display = 'none';
  document.getElementById('cam-status-bar').style.display = 'none';
  document.getElementById('cam-start-btn').style.display = 'inline-flex';
  document.getElementById('cam-stop-btn').style.display = 'none';
  document.getElementById('cam-badge').className = 'badge badge-off';
  document.getElementById('cam-badge').textContent = 'INATIVO';
  document.getElementById('cam-mirror-btn').classList.remove('visible');
  document.getElementById('cam-device-sel').classList.remove('visible');
  document.querySelector('.cam-overlay.cam-scanline')?.parentElement?.classList.remove('cam-scanning');
  document.getElementById('cam-result-detail').innerHTML = '<div class="empty-state" data-icon="📷">Câmera parada</div>';
  _camFrameCount = 0; _camFps = 0;
  toast('Câmera parada');
}

async function camSendFrame() {
  if (_camSending) return; // evita overlap de requests
  const video = document.getElementById('cam-video');
  if (!video.videoWidth) return; // frame ainda não disponível

  _camSending = true;
  const t0 = performance.now();

  try {
    const canvas = document.getElementById('cam-canvas');
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');

    // Se espelhado, desenha invertido no canvas (para o servidor receber não-espelhado)
    if (_camMirror) {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
      ctx.restore();
    } else {
      ctx.drawImage(video, 0, 0);
    }

    // Converte para JPEG blob
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));

    const fd = new FormData();
    fd.append('file', blob, 'frame.jpg');
    fd.append('confidence', document.getElementById('cam-conf').value);
    fd.append('detect_faces', document.getElementById('cam-faces').checked);
    fd.append('face_threshold', '0.45');
    fd.append('annotate', 'true');

    const resp = await fetch(API('/detect/frame'), { method: 'POST', body: fd });
    if (!resp.ok) { _camSending = false; return; }
    const r = await resp.json();

    const elapsed = Math.round(performance.now() - t0);

    // Exibe imagem anotada
    if (r.annotated_base64) {
      const img = document.getElementById('cam-result-img');
      const video2 = document.getElementById('cam-video');
      img.src = 'data:image/jpeg;base64,' + r.annotated_base64;
      if (_camMirror) {
        img.style.transform = 'scaleX(-1)';
      } else {
        img.style.transform = '';
      }
      img.style.display = 'block';
      video2.style.display = 'none'; // troca para imagem anotada
    }

    // FPS calc
    _camFrameCount++;
    const now = performance.now();
    const elapsed2 = (now - _camFpsTs) / 1000;
    if (elapsed2 >= 1.0) {
      _camFps = Math.round(_camFrameCount / elapsed2);
      _camFrameCount = 0;
      _camFpsTs = now;
    }

    // Update badges
    document.getElementById('cam-fps-badge').textContent = _camFps + ' FPS';
    const cbadge = document.getElementById('cam-compliance-badge');
    if (r.compliant) {
      cbadge.textContent = '✓ COMPLIANT';
      cbadge.className = 'cam-fps-badge compliant';
    } else {
      cbadge.textContent = '✗ ' + (r.missing || []).join(', ') || 'NON-COMPLIANT';
      cbadge.className = 'cam-fps-badge noncompliant';
    }
    document.getElementById('cam-ms-badge').textContent = elapsed + 'ms';

    // Sidebar result
    camRenderResult(r, elapsed);

  } catch(e) {
    console.warn('Frame send error:', e);
  } finally {
    _camSending = false;
  }
}

function camRenderResult(r, ms) {
  const el = document.getElementById('cam-result-detail');
  const compClass = r.compliant ? 'result-compliant' : 'result-noncompliant';
  const compLabel = r.compliant ? '✓ COMPLIANT' : '✗ NON-COMPLIANT';
  const compColor = r.compliant ? 'ok' : 'fail';

  let html = '<div class="result-box ' + compClass + '">';
  html += '<div class="result-title ' + compColor + '">' + compLabel + '</div>';
  html += '<div style="font-size:12px;color:var(--text-muted);margin-top:4px">';
  html += 'EPIs detectados: <strong>' + (r.detected_count||0) + '/' + (r.required_count||0) + '</strong></div>';

  if (r.missing && r.missing.length) {
    html += '<div style="margin-top:8px;font-size:12px">Faltando: <span style="color:var(--danger)">' + r.missing.join(', ') + '</span></div>';
  }

  if (r.detections && r.detections.length) {
    html += '<div style="margin-top:8px">';
    r.detections.forEach(d => {
      html += '<span class="badge badge-blue" style="margin:2px;font-size:10px">' + d.class_name + ' ' + Math.round(d.confidence*100) + '%</span>';
    });
    html += '</div>';
  }

  if (r.faces && r.faces.length) {
    html += '<div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(0,0,0,0.08)">';
    r.faces.forEach(f => {
      const fc = f.recognized ? 'badge-ok' : 'badge-off';
      html += '<div style="font-size:12px;margin-bottom:4px"><span class="badge ' + fc + '">' + f.person_name + '</span>';
      if (f.recognized) html += ' <span style="color:var(--text-muted);font-size:11px">' + Math.round(f.confidence*100) + '% match</span>';
      html += '</div>';
    });
    html += '</div>';
  }

  html += '<div style="margin-top:8px;font-size:10px;color:var(--text-dim);font-family:Space Mono,monospace">';
  html += '⏱ ' + ms + 'ms · model: ' + (r.model_name||'best') + '</div>';
  html += '</div>';

  el.innerHTML = html;
}

function camToggleMirror() {
  _camMirror = document.getElementById('cam-mirror').checked;
  camApplyMirror();
}

function camApplyMirror() {
  const video = document.getElementById('cam-video');
  video.style.transform = _camMirror ? 'scaleX(-1)' : '';
}

async function camPopulateDevices() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter(d => d.kind === 'videoinput');
    const sel = document.getElementById('cam-device-sel');
    sel.innerHTML = cams.map((d, i) =>
      '<option value="' + d.deviceId + '">' + (d.label || 'Câmera ' + (i+1)) + '</option>'
    ).join('');
    if (cams.length > 1) sel.classList.add('visible');
  } catch(e) {}
}

async function camSwitchDevice() {
  if (!_camStream) return;
  const deviceId = document.getElementById('cam-device-sel').value;
  if (!deviceId) return;
  // Para a câmera atual
  _camStream.getTracks().forEach(t => t.stop());
  // Abre nova câmera
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    _camStream = stream;
    document.getElementById('cam-video').srcObject = stream;
    toast('Câmera alterada');
  } catch(e) {
    toast('Erro ao trocar câmera: ' + e.message, 'error');
  }
}

// Para câmera automaticamente se o usuário sair da aba
const _origShowPanel = showPanel;
function showPanel(n, el) {
  if (n !== 'stream' && _camStream) camStop();
  _origShowPanel(n, el);
}
// =============================================================
"""


def patch_dashboard():
    print("\n[2/2] Patching dashboard.html ...")
    backup(DASHBOARD_PATH)
    html = DASHBOARD_PATH.read_text()

    # ── 1. CSS ──────────────────────────────────────────────────────────────
    if "cam-container" in html:
        print("  [SKIP] CSS de câmera já existe")
    else:
        # Inserir antes do closing </style>
        html = html.replace("</style>", CAMERA_CSS + "\n</style>", 1)
        print("  [OK] CSS de câmera adicionado")

    # ── 2. HTML (card dentro do painel stream) ───────────────────────────────
    if "cam-start-btn" in html:
        print("  [SKIP] HTML de câmera já existe")
    else:
        # Inserir antes do closing </div><!-- end panels -->
        marker = "</div><!-- end panels -->"
        if marker not in html:
            marker = "</div>\n\n<script>"  # fallback
        # Inserir depois do último </div> do painel stream
        # O painel stream termina com </div>\n</div> antes de "<!-- ===== FILE BROWSER"
        stream_end_marker = "<!-- ===== FILE BROWSER ===== -->"
        if stream_end_marker in html:
            # Achar o </div>\n</div> imediatamente antes do marker do file browser
            idx = html.index(stream_end_marker)
            # Regredir para achar o ponto de inserção
            insert_pos = html.rfind("</div>\n\n<!-- ===== FILE BROWSER", 0, idx + 50)
            if insert_pos == -1:
                insert_pos = html.rfind("</div>", 0, idx)
            html = html[:insert_pos] + "\n" + CAMERA_HTML_TAB + "\n\n" + html[insert_pos:]
            print("  [OK] Card de câmera adicionado no painel stream")
        else:
            print("  [WARN] Não encontrou marker do File Browser — adicionando antes de </div><!-- end panels -->")
            html = html.replace("</div><!-- end panels -->",
                                CAMERA_HTML_TAB + "\n</div><!-- end panels -->")
            print("  [OK] Card de câmera adicionado (fallback)")

    # ── 3. JavaScript ────────────────────────────────────────────────────────
    if "_camStream" in html:
        print("  [SKIP] JS de câmera já existe")
    else:
        # Inserir antes do </script> final
        html = html.rsplit("</script>", 1)
        html = (CAMERA_JS + "\n</script>").join(html)
        print("  [OK] JS de câmera adicionado")

    DASHBOARD_PATH.write_text(html)
    print("  [DONE] dashboard.html salvo")


def verify():
    print("\n" + "=" * 60)
    print("VERIFICAÇÃO")
    print("=" * 60)
    all_ok = True

    checks_routes = [
        ("/detect/frame", ROUTES_PATH.read_text() if ROUTES_PATH.exists() else ""),
    ]
    for name, content in checks_routes:
        ok = name in content
        if not ok: all_ok = False
        print(f"  {'[OK]' if ok else '[FAIL]'} routes.py: endpoint {name}")

    html = DASHBOARD_PATH.read_text() if DASHBOARD_PATH.exists() else ""
    checks_dash = [
        ("CSS cam-container",   "cam-container" in html),
        ("HTML cam-start-btn",  "cam-start-btn" in html),
        ("JS getUserMedia",     "getUserMedia" in html),
        ("JS camStart()",       "async function camStart" in html),
        ("JS camStop()",        "function camStop" in html),
        ("JS camSendFrame()",   "async function camSendFrame" in html),
    ]
    for name, ok in checks_dash:
        if not ok: all_ok = False
        print(f"  {'[OK]' if ok else '[FAIL]'} dashboard: {name}")

    return all_ok


def main():
    print("=" * 60)
    print("SmartX Vision Platform — Browser Camera Patch")
    print("=" * 60)
    print(f"Root:      {ROOT}")
    print(f"Routes:    {ROUTES_PATH}")
    print(f"Dashboard: {DASHBOARD_PATH}")
    print()

    for p, name in [(ROUTES_PATH, "routes.py"), (DASHBOARD_PATH, "dashboard.html")]:
        if not p.exists():
            print(f"[ERROR] {name} não encontrado: {p}")
            print("Execute este script a partir do diretório raiz do projeto.")
            return

    patch_routes()
    patch_dashboard()

    ok = verify()

    print("\n" + "=" * 60)
    if ok:
        print("[SUCCESS] Patch aplicado com sucesso!")
    else:
        print("[WARNING] Alguns itens podem precisar de revisão manual")
    print("=" * 60)
    print("""
O que foi adicionado:

  Backend (routes.py):
    POST /api/v1/epi/detect/frame
      • Recebe frame JPEG (do browser)
      • Roda detecção de EPI + face recognition
      • Retorna JSON + imagem anotada em base64
      • Menor overhead que /detect/upload (sem salvar em disco)

  Frontend (dashboard.html):
    Novo card "Browser Camera" na aba Live Stream:
      • Acessa câmera via navigator.mediaDevices.getUserMedia
      • Captura frames a cada 100–1000ms (configurável)
      • Envia para /detect/frame e exibe resultado anotado
      • Badges de FPS, compliance e latência em tempo real
      • Seletor de câmera (útil em notebooks/celulares com 2+ câmeras)
      • Espelhar imagem (toggle)
      • Para câmera automaticamente ao trocar de aba

Para aplicar: reinicie o servidor
  python main.py
""")


if __name__ == "__main__":
    main()
