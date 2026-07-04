"""
End-to-end Kaggle late submission for the 0.4784 ensemble, driven entirely by
the bearer token in ~/.kaggle/access_token (the official CLI can't use it — it
does HTTP Basic auth; our token only works as Bearer).

Steps (each idempotent/resumable):
  1. upload models + code as dataset  neel999/nfl-bdb26-models   (kagglehub)
  2. push script kernel               neel999/nfl-bdb26-submit    (REST)
  3. poll kernel run until complete                               (REST)
  4. submit the kernel version to the competition                (REST /submit-notebook)
  5. poll our submissions list for the private/public score      (REST)

Run in stages with STAGE=1|2|3|4|5|all (default all).
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
TOKEN = Path(os.path.expanduser("~/.kaggle/access_token")).read_text().strip()
API = "https://www.kaggle.com/api/v1"
H = {"Authorization": f"Bearer {TOKEN}"}

USER = "neel999"
COMP = "nfl-big-data-bowl-2026-prediction"
DS_SLUG = "nfl-bdb26-models"
DS_HANDLE = f"{USER}/{DS_SLUG}"
K_SLUG = "nfl-bdb26-submit"
K_REF = f"{USER}/{K_SLUG}"
DS_DIR = ROOT / "kaggle_pkg/model_ds"
NB_FILE = ROOT / "kaggle_pkg/notebook/nfl_bdb26_submit.py"
STATE = ROOT / "kaggle_pkg/state.json"

STAGE = os.environ.get("STAGE", "all")


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(**kw):
    s = load_state(); s.update(kw); STATE.write_text(json.dumps(s, indent=2)); return s


def api(method, path, **kw):
    r = requests.request(method, f"{API}{path}", headers=H, timeout=120, **kw)
    return r


# ---------- 1. dataset ----------
def _dataset_version():
    r = api("GET", f"/datasets/view/{DS_HANDLE}")
    try:
        return r.json().get("currentVersionNumber")
    except Exception:
        return None


def stage_dataset():
    os.environ["KAGGLE_API_TOKEN"] = TOKEN
    import kagglehub
    # metadata file lets kagglehub/CLI name it; harmless if ignored
    (DS_DIR / "dataset-metadata.json").write_text(json.dumps({
        "title": "NFL BDB26 Models", "id": DS_HANDLE,
        "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    prev_ver = _dataset_version()
    print(f"[1] uploading {DS_DIR} -> {DS_HANDLE} (prev version {prev_ver})", flush=True)
    try:
        kagglehub.dataset_upload(DS_HANDLE, str(DS_DIR),
                                 version_notes=os.environ.get("DS_NOTES", "kf0 single (1st-place scale, 37k steps, val 0.4685)"))
    except Exception as e:
        print(f"[1] dataset_upload raised {type(e).__name__}: {e}", flush=True)
        raise
    # CRITICAL: Kaggle processes new versions async. If we run the kernel before
    # the new version is ready, it silently attaches the PREVIOUS version (this
    # cost submission B: it loaded only kf0, not the ensemble). Wait for both a
    # version bump AND status "ready" before returning.
    deadline = time.time() + 600
    while time.time() < deadline:
        time.sleep(15)
        ver = _dataset_version()
        status = None
        try:
            status = api("GET", f"/datasets/status/{DS_HANDLE}").json()
        except Exception:
            pass
        ready = status == "ready" and (prev_ver is None or (ver or 0) > prev_ver)
        print(f"[1] dataset version={ver} status={status} ready={ready}", flush=True)
        if ready:
            save_state(dataset=DS_HANDLE, dataset_version=ver)
            print(f"[1] dataset v{ver} READY", flush=True)
            return
    raise RuntimeError("[1] dataset did not become ready within 600s")


# ---------- 2. push kernel ----------
def stage_push():
    src = NB_FILE.read_text()
    body = {
        "slug": K_REF,
        "newTitle": "NFL BDB26 Submit",
        "text": src,
        "language": "python",
        "kernelType": "script",
        "isPrivate": True,
        "enableGpu": False,
        "enableTpu": False,
        "enableInternet": False,
        "datasetDataSources": [DS_HANDLE],
        "competitionDataSources": [COMP],
        "kernelDataSources": [],
        "modelDataSources": [],
        "categoryIds": [],
    }
    print(f"[2] pushing kernel {K_REF}", flush=True)
    r = api("POST", "/kernels/push", json=body)
    print("[2] status", r.status_code, r.text[:500], flush=True)
    r.raise_for_status()
    d = r.json()
    ver = d.get("versionNumber") or d.get("versionNumber".lower())
    save_state(kernel=K_REF, version=ver, push_url=d.get("url"))
    print(f"[2] pushed version={ver} url={d.get('url')}", flush=True)


# ---------- 3. poll kernel run ----------
def stage_poll_kernel():
    print(f"[3] polling kernel status {K_REF}", flush=True)
    for _ in range(240):  # up to ~2h at 30s
        r = api("GET", "/kernels/status", params={"userName": USER, "kernelSlug": K_SLUG})
        if r.status_code == 200:
            st = r.json()
            status = st.get("status", "?")
            print(f"[3] {time.strftime('%H:%M:%S')} status={status} {st.get('failureMessage') or ''}", flush=True)
            if str(status).lower() in ("complete", "error", "cancelacknowledged", "cancelrequested"):
                save_state(kernel_status=status)
                return status
        else:
            print(f"[3] status http {r.status_code} {r.text[:200]}", flush=True)
        time.sleep(30)
    return "timeout"


# ---------- 4. submit notebook to competition ----------
def stage_submit():
    s = load_state()
    ver = s.get("version")
    body = {
        "competitionName": COMP,
        "kernelOwner": USER,
        "kernelSlug": K_SLUG,
        "fileName": "submission.parquet",
        "submissionDescription": "autoresearch ensemble (val 0.4784) late submission",
    }
    if ver:
        body["kernelVersion"] = int(ver)
    print(f"[4] submitting {K_REF} v{ver} to {COMP}", flush=True)
    # NB: this endpoint requires a form-encoded body (JSON is rejected with
    # "requires an output FileName"); send with data=, not json=.
    r = api("POST", f"/competitions/submissions/submit-notebook/{COMP}", data=body)
    print("[4] status", r.status_code, r.text[:600], flush=True)
    save_state(submit_response=r.text[:600], submit_code=r.status_code)


# ---------- 5. poll score ----------
def stage_score():
    print("[5] polling submissions for score", flush=True)
    for _ in range(240):
        r = api("GET", f"/competitions/submissions/list/{COMP}")
        if r.status_code == 200:
            subs = r.json()
            if subs:
                latest = subs[0]
                st = latest.get("status")
                pub = latest.get("publicScore") or latest.get("publicScoreNullable")
                pri = latest.get("privateScore") or latest.get("privateScoreNullable")
                print(f"[5] {time.strftime('%H:%M:%S')} status={st} public={pub} private={pri}", flush=True)
                if str(st).lower() == "complete":
                    save_state(public=pub, private=pri)
                    print(f"[5] DONE public={pub} private={pri}", flush=True)
                    return
                if str(st).lower() == "error":
                    print(f"[5] submission errored: {latest.get('errorDescription')}", flush=True)
                    return
        time.sleep(30)
    print("[5] timeout waiting for score", flush=True)


STAGES = {"1": stage_dataset, "2": stage_push, "3": stage_poll_kernel,
          "4": stage_submit, "5": stage_score}

if __name__ == "__main__":
    if STAGE == "all":
        stage_dataset()
        stage_push()
        status = stage_poll_kernel()
        if str(status).lower() != "complete":
            print(f"[!] kernel did not complete (status={status}); not submitting", flush=True)
            sys.exit(1)
        stage_submit()
        stage_score()
    else:
        for st in STAGE.split(","):
            STAGES[st.strip()]()
