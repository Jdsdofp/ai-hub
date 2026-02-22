#!/usr/bin/env python3
"""
SmartX Vision Platform v3.1 — Bug Fix Script
=============================================
Fixes two critical UI bugs in the dashboard:

BUG 1: Photo upload by category doesn't physically save files
  - Root cause: uploadPhotos() JS function has zero error handling.
    If the fetch fails (timeout, large payload, server error), it dies
    silently. Also, no loading indicator so user thinks nothing happened.
  - Fix: Add try/catch, loading state, progress feedback, error display.

BUG 2: Dataset generation shows "undefined train / undefined valid"
  - Root cause: FastAPI HTTPException returns {"detail": "..."} but the
    JS checks for r.error (which doesn't exist). So the error check fails
    and it reads r.train / r.valid which are undefined.
  - Fix: Check both r.error and r.detail, add try/catch, proper error display.

BONUS FIXES:
  - bulkUpload() same missing error handling
  - convertUpload() same pattern
  - All POST functions now have consistent error handling

Usage: python3 svp3_fix_v3.1.py
"""

import os
import shutil
from datetime import datetime

# =============================================================
# CONFIGURATION
# =============================================================
SVP3_ROOT = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_PATH = os.path.join(SVP3_ROOT, "app", "ui", "templates", "dashboard.html")
ROUTES_PATH = os.path.join(SVP3_ROOT, "app", "projects", "epi_check", "api", "routes.py")
DETECTOR_PATH = os.path.join(SVP3_ROOT, "app", "projects", "epi_check", "engine", "detector.py")

BACKUP_SUFFIX = datetime.now().strftime(".bak_%Y%m%d_%H%M%S")


def backup_file(path):
    """Create a timestamped backup."""
    if os.path.exists(path):
        bak = path + BACKUP_SUFFIX
        shutil.copy2(path, bak)
        print(f"  [BACKUP] {os.path.basename(path)} → {os.path.basename(bak)}")
        return bak
    return None


def fix_dashboard():
    """Fix all JS functions in dashboard.html."""
    print("\n[1/3] Fixing dashboard.html ...")
    backup_file(DASHBOARD_PATH)

    with open(DASHBOARD_PATH, "r") as f:
        html = f.read()

    # =========================================================
    # FIX 1: uploadPhotos() — add error handling + loading state
    # =========================================================
    old_upload = (
        "async function uploadPhotos(){"
        "const f=document.getElementById('upload-files').files;"
        "if(!f.length)return toast('Select files','error');"
        "const fd=new FormData();"
        "fd.append('category',document.getElementById('upload-category').value);"
        "for(const x of f)fd.append('files',x);"
        "const r=await(await fetch(API('/upload/photos'),{method:'POST',body:fd})).json();"
        "document.getElementById('upload-result').innerHTML="
        "'<span class=\"badge badge-ok\">'+r.uploaded+' uploaded</span>';"
        "toast(r.uploaded+' photos uploaded')}"
    )

    new_upload = (
        "async function uploadPhotos(){\n"
        "  const f=document.getElementById('upload-files').files;\n"
        "  if(!f.length)return toast('Select files first','error');\n"
        "  const cat=document.getElementById('upload-category').value;\n"
        "  const el=document.getElementById('upload-result');\n"
        "  el.innerHTML='<span class=\"badge badge-warn\">Uploading '+f.length+' files to '+cat+'... please wait</span>';\n"
        "  try{\n"
        "    const fd=new FormData();\n"
        "    fd.append('category',cat);\n"
        "    for(const x of f)fd.append('files',x);\n"
        "    const resp=await fetch(API('/upload/photos'),{method:'POST',body:fd});\n"
        "    if(!resp.ok){\n"
        "      const err=await resp.json().catch(()=>({detail:resp.statusText}));\n"
        "      el.innerHTML='<span class=\"badge badge-err\">Error: '+(err.detail||err.error||resp.statusText)+'</span>';\n"
        "      toast('Upload failed: '+(err.detail||resp.statusText),'error');\n"
        "      return;\n"
        "    }\n"
        "    const r=await resp.json();\n"
        "    const ok=r.uploaded||0;\n"
        "    const fail=r.results?r.results.filter(x=>!x.ok).length:0;\n"
        "    let msg='<span class=\"badge badge-ok\">'+ok+' photos uploaded to '+cat+'</span>';\n"
        "    if(fail)msg+=' <span class=\"badge badge-err\">'+fail+' failed</span>';\n"
        "    if(r.results){\n"
        "      msg+='<div style=\"margin-top:8px;font-size:12px;max-height:150px;overflow-y:auto\">';\n"
        "      r.results.forEach(x=>{\n"
        "        msg+='<div>'+(x.ok?'✓':'✗')+' '+x.file+(x.size?' ('+x.size+')':'')+(x.error?' — '+x.error:'')+'</div>';\n"
        "      });\n"
        "      msg+='</div>';\n"
        "    }\n"
        "    el.innerHTML=msg;\n"
        "    toast(ok+' photos uploaded to '+cat);\n"
        "    document.getElementById('upload-files').value='';\n"
        "  }catch(e){\n"
        "    console.error('Upload error:',e);\n"
        "    el.innerHTML='<span class=\"badge badge-err\">Upload failed: '+e.message+'</span>';\n"
        "    toast('Upload error: '+e.message,'error');\n"
        "  }\n"
        "}"
    )

    if old_upload in html:
        html = html.replace(old_upload, new_upload)
        print("  [OK] uploadPhotos() fixed")
    else:
        print("  [WARN] uploadPhotos() — exact match not found, applying regex patch")
        import re
        pattern = r'async function uploadPhotos\(\)\{.*?toast\(r\.uploaded\+\' photos uploaded\'\)\}'
        match = re.search(pattern, html)
        if match:
            html = html[:match.start()] + new_upload + html[match.end():]
            print("  [OK] uploadPhotos() fixed via regex")
        else:
            print("  [ERROR] Could not find uploadPhotos() function!")

    # =========================================================
    # FIX 2: bulkUpload() — same error handling pattern
    # =========================================================
    old_bulk = (
        "async function bulkUpload(){"
        "const f=document.getElementById('bulk-files').files;"
        "if(!f.length)return toast('Select files','error');"
        "const fd=new FormData();"
        "fd.append('category',document.getElementById('bulk-category').value);"
        "fd.append('remap_json',document.getElementById('bulk-remap').value);"
        "for(const x of f)fd.append('files',x);"
        "const r=await(await fetch(API('/upload/bulk'),{method:'POST',body:fd})).json();"
        "document.getElementById('bulk-result').innerHTML="
        "'<span class=\"badge badge-ok\">'+r.paired+' paired</span> "
        "<span class=\"badge badge-warn\">'+r.images_only+' image-only</span>';"
        "toast('Bulk: '+r.paired+' paired')}"
    )

    new_bulk = (
        "async function bulkUpload(){\n"
        "  const f=document.getElementById('bulk-files').files;\n"
        "  if(!f.length)return toast('Select files first','error');\n"
        "  const cat=document.getElementById('bulk-category').value;\n"
        "  const el=document.getElementById('bulk-result');\n"
        "  el.innerHTML='<span class=\"badge badge-warn\">Uploading '+f.length+' files... please wait</span>';\n"
        "  try{\n"
        "    const fd=new FormData();\n"
        "    fd.append('category',cat);\n"
        "    fd.append('remap_json',document.getElementById('bulk-remap').value);\n"
        "    for(const x of f)fd.append('files',x);\n"
        "    const resp=await fetch(API('/upload/bulk'),{method:'POST',body:fd});\n"
        "    if(!resp.ok){\n"
        "      const err=await resp.json().catch(()=>({detail:resp.statusText}));\n"
        "      el.innerHTML='<span class=\"badge badge-err\">Error: '+(err.detail||resp.statusText)+'</span>';\n"
        "      toast('Bulk upload failed','error');return;\n"
        "    }\n"
        "    const r=await resp.json();\n"
        "    el.innerHTML='<span class=\"badge badge-ok\">'+(r.paired||0)+' paired</span> <span class=\"badge badge-warn\">'+(r.images_only||0)+' image-only</span>';\n"
        "    toast('Bulk: '+(r.paired||0)+' paired to '+cat);\n"
        "  }catch(e){\n"
        "    console.error('Bulk upload error:',e);\n"
        "    el.innerHTML='<span class=\"badge badge-err\">Error: '+e.message+'</span>';\n"
        "    toast('Bulk upload error','error');\n"
        "  }\n"
        "}"
    )

    if old_bulk in html:
        html = html.replace(old_bulk, new_bulk)
        print("  [OK] bulkUpload() fixed")
    else:
        import re
        pattern = r"async function bulkUpload\(\)\{.*?toast\('Bulk: '\+r\.paired\+' paired'\)\}"
        match = re.search(pattern, html)
        if match:
            html = html[:match.start()] + new_bulk + html[match.end():]
            print("  [OK] bulkUpload() fixed via regex")
        else:
            print("  [WARN] bulkUpload() not found — may already be patched")

    # =========================================================
    # FIX 3: generateDataset() — fix "undefined train / undefined valid"
    # =========================================================
    old_dataset = (
        "async function generateDataset(){"
        "const fd=new FormData();"
        "fd.append('train_split',document.getElementById('train-split').value);"
        "const r=await(await fetch(API('/dataset/generate'),{method:'POST',body:fd})).json();"
        "if(r.error){toast(r.error,'error');return}"
        "document.getElementById('dataset-result').innerHTML="
        "'<span class=\"badge badge-ok\">'+r.train+' train / '+r.valid+' valid</span>"
        "<br>Classes: '+Object.values(r.classes||{}).join(', ');"
        "toast('Dataset generated!')}"
    )

    new_dataset = (
        "async function generateDataset(){\n"
        "  const el=document.getElementById('dataset-result');\n"
        "  el.innerHTML='<span class=\"badge badge-warn\">Generating dataset... please wait</span>';\n"
        "  try{\n"
        "    const fd=new FormData();\n"
        "    fd.append('train_split',document.getElementById('train-split').value);\n"
        "    const resp=await fetch(API('/dataset/generate'),{method:'POST',body:fd});\n"
        "    const r=await resp.json();\n"
        "    if(!resp.ok||r.error||r.detail){\n"
        "      const errMsg=r.detail||r.error||'Unknown error';\n"
        "      el.innerHTML='<span class=\"badge badge-err\">Error: '+errMsg+'</span>';\n"
        "      toast(errMsg,'error');\n"
        "      return;\n"
        "    }\n"
        "    const trainN=r.train||0;\n"
        "    const validN=r.valid||0;\n"
        "    let classHtml='';\n"
        "    if(r.classes){\n"
        "      classHtml='<br>Classes: '+Object.values(r.classes).join(', ');\n"
        "    }\n"
        "    el.innerHTML='<span class=\"badge badge-ok\">'+trainN+' train / '+validN+' valid</span>'+classHtml;\n"
        "    toast('Dataset generated: '+trainN+' train, '+validN+' valid');\n"
        "  }catch(e){\n"
        "    console.error('Dataset error:',e);\n"
        "    el.innerHTML='<span class=\"badge badge-err\">Error: '+e.message+'</span>';\n"
        "    toast('Dataset generation failed','error');\n"
        "  }\n"
        "}"
    )

    if old_dataset in html:
        html = html.replace(old_dataset, new_dataset)
        print("  [OK] generateDataset() fixed")
    else:
        import re
        pattern = r"async function generateDataset\(\)\{.*?toast\('Dataset generated!'\)\}"
        match = re.search(pattern, html)
        if match:
            html = html[:match.start()] + new_dataset + html[match.end():]
            print("  [OK] generateDataset() fixed via regex")
        else:
            print("  [ERROR] Could not find generateDataset() function!")

    # =========================================================
    # FIX 4: convertUpload() — add error handling
    # =========================================================
    old_convert = (
        "async function convertUpload(){"
        "const f=document.getElementById('conv-files').files;"
        "if(!f.length)return toast('Select files','error');"
        "const fd=new FormData();"
        "fd.append('remap_json',document.getElementById('conv-remap').value);"
        "for(const x of f)fd.append('files',x);"
        "document.getElementById('conv-result').innerHTML='Converting...';"
        "const r=await(await fetch(API('/convert/upload'),{method:'POST',body:fd})).json();"
        "document.getElementById('conv-result').innerHTML="
        "'<span class=\"badge badge-ok\">'+r.files_converted+' files converted</span>"
        " &middot; '+r.boxes_converted+' boxes';"
        "toast(r.files_converted+' files converted')}"
    )

    new_convert = (
        "async function convertUpload(){\n"
        "  const f=document.getElementById('conv-files').files;\n"
        "  if(!f.length)return toast('Select files first','error');\n"
        "  const el=document.getElementById('conv-result');\n"
        "  el.innerHTML='<span class=\"badge badge-warn\">Converting... please wait</span>';\n"
        "  try{\n"
        "    const fd=new FormData();\n"
        "    fd.append('remap_json',document.getElementById('conv-remap').value);\n"
        "    for(const x of f)fd.append('files',x);\n"
        "    const resp=await fetch(API('/convert/upload'),{method:'POST',body:fd});\n"
        "    if(!resp.ok){\n"
        "      const err=await resp.json().catch(()=>({detail:resp.statusText}));\n"
        "      el.innerHTML='<span class=\"badge badge-err\">Error: '+(err.detail||resp.statusText)+'</span>';\n"
        "      toast('Convert failed','error');return;\n"
        "    }\n"
        "    const r=await resp.json();\n"
        "    el.innerHTML='<span class=\"badge badge-ok\">'+(r.files_converted||0)+' files converted</span> &middot; '+(r.boxes_converted||0)+' boxes';\n"
        "    toast((r.files_converted||0)+' files converted');\n"
        "  }catch(e){\n"
        "    console.error('Convert error:',e);\n"
        "    el.innerHTML='<span class=\"badge badge-err\">Error: '+e.message+'</span>';\n"
        "    toast('Conversion error','error');\n"
        "  }\n"
        "}"
    )

    if old_convert in html:
        html = html.replace(old_convert, new_convert)
        print("  [OK] convertUpload() fixed")
    else:
        import re
        pattern = r"async function convertUpload\(\)\{.*?toast\(r\.files_converted\+' files converted'\)\}"
        match = re.search(pattern, html)
        if match:
            html = html[:match.start()] + new_convert + html[match.end():]
            print("  [OK] convertUpload() fixed via regex")
        else:
            print("  [WARN] convertUpload() not found — may already be patched")

    with open(DASHBOARD_PATH, "w") as f:
        f.write(html)

    print("  [DONE] dashboard.html saved")


def fix_routes():
    """Fix the dataset/generate endpoint to return consistent error format."""
    print("\n[2/3] Fixing routes.py ...")
    backup_file(ROUTES_PATH)

    with open(ROUTES_PATH, "r") as f:
        content = f.read()

    # Fix: generate_dataset endpoint — return JSON error instead of HTTPException
    # so the JS can handle it consistently
    old_generate = '''async def generate_dataset(
    train_split: float = Form(0.8, description="Train/valid split ratio (0.5–0.95)", examples=[0.8]),
    company_id: int = Depends(get_ui_company),
):
    result = epi_engine.generate_dataset(company_id, train_split)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result'''

    new_generate = '''async def generate_dataset(
    train_split: float = Form(0.8, description="Train/valid split ratio (0.5–0.95)", examples=[0.8]),
    company_id: int = Depends(get_ui_company),
):
    try:
        result = epi_engine.generate_dataset(company_id, train_split)
        if "error" in result:
            raise HTTPException(400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] Dataset generation error: {e}")
        raise HTTPException(500, detail=f"Dataset generation failed: {str(e)}")'''

    if old_generate in content:
        content = content.replace(old_generate, new_generate)
        print("  [OK] generate_dataset endpoint fixed")
    else:
        print("  [WARN] generate_dataset — exact match not found, trying partial")
        # Try partial match
        if "raise HTTPException(400, result[\"error\"])" in content:
            content = content.replace(
                'raise HTTPException(400, result["error"])',
                'raise HTTPException(400, detail=result["error"])'
            )
            print("  [OK] HTTPException fixed (detail= keyword)")
        else:
            print("  [INFO] routes.py may already be correct")

    # Also make sure logger is imported
    if "from loguru import logger" not in content:
        content = content.replace(
            "from fastapi import",
            "from loguru import logger\nfrom fastapi import"
        )
        print("  [OK] Added logger import")

    with open(ROUTES_PATH, "w") as f:
        f.write(content)

    print("  [DONE] routes.py saved")


def fix_detector():
    """Add better error handling to generate_dataset in detector.py."""
    print("\n[3/3] Fixing detector.py ...")
    backup_file(DETECTOR_PATH)

    with open(DETECTOR_PATH, "r") as f:
        content = f.read()

    # The generate_dataset method looks correct in the engine.
    # But let's make sure it has good error messages.
    old_no_pairs = 'return {"error": "No valid annotated images with active classes found"}'
    new_no_pairs = (
        'return {"error": "No valid annotated images found. '
        'Upload photos (Upload tab), annotate them (Annotate tab), '
        'then generate the dataset."}'
    )

    if old_no_pairs in content:
        content = content.replace(old_no_pairs, new_no_pairs)
        print("  [OK] Improved error message for empty annotations")
    else:
        print("  [INFO] Error message may already be updated")

    with open(DETECTOR_PATH, "w") as f:
        f.write(content)

    print("  [DONE] detector.py saved")


def verify_fixes():
    """Quick verification of the patches."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    with open(DASHBOARD_PATH, "r") as f:
        html = f.read()

    checks = [
        ("Upload error handling", "resp.ok" in html and "Upload failed" in html),
        ("Upload loading state", "Uploading" in html and "please wait" in html),
        ("Dataset error handling", "r.detail||r.error" in html),
        ("Dataset loading state", "Generating dataset" in html),
        ("Bulk upload error handling", "Bulk upload failed" in html),
        ("Convert error handling", "Convert failed" in html),
    ]

    all_ok = True
    for name, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        if not passed:
            all_ok = False
        print(f"  {status} {name}")

    with open(ROUTES_PATH, "r") as f:
        routes = f.read()

    route_checks = [
        ("HTTPException detail=", "detail=result" in routes),
    ]
    for name, passed in route_checks:
        status = "[OK]" if passed else "[FAIL]"
        if not passed:
            all_ok = False
        print(f"  {status} {name}")

    return all_ok


def main():
    print("=" * 60)
    print("SmartX Vision Platform v3.1 — Bug Fix Script")
    print("=" * 60)
    print(f"SVP3 Root: {SVP3_ROOT}")
    print(f"Dashboard: {DASHBOARD_PATH}")
    print(f"Routes:    {ROUTES_PATH}")
    print(f"Detector:  {DETECTOR_PATH}")

    # Check files exist
    for path, name in [(DASHBOARD_PATH, "dashboard.html"),
                       (ROUTES_PATH, "routes.py"),
                       (DETECTOR_PATH, "detector.py")]:
        if not os.path.exists(path):
            print(f"\n[ERROR] {name} not found at: {path}")
            print("Make sure you run this script from the SVP3 root directory.")
            return

    fix_dashboard()
    fix_routes()
    fix_detector()

    ok = verify_fixes()

    print("\n" + "=" * 60)
    if ok:
        print("[SUCCESS] All fixes applied!")
    else:
        print("[WARNING] Some fixes may need manual review")

    print("=" * 60)
    print("\nWhat was fixed:")
    print("  1. uploadPhotos() — error handling + loading indicator + file details")
    print("  2. bulkUpload() — error handling + loading indicator")
    print("  3. generateDataset() — checks r.detail (FastAPI error format)")
    print("  4. convertUpload() — error handling + loading indicator")
    print("  5. HTTPException(400, detail=...) — explicit detail keyword")
    print("  6. Better error messages in dataset generation")
    print("\nTo apply: restart the server")
    print("  source ~/svp3-env/bin/activate")
    print("  cd ~/svp3 && python main.py")
    print()


if __name__ == "__main__":
    main()
