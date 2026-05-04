#!/usr/bin/env bash
# VoiceAgent installer — multi-distro (Ubuntu/Debian/Arch), AMD+NVIDIA,
# TUI mit gum. Idempotent, jeder sudo-Aufruf einzeln (kein Cache).
#
# Bei "Nein" auf einen Install-Confirm: Step wird uebersprungen und am
# Ende als Warnung mit manueller Anleitung gesammelt.
set -uo pipefail   # KEIN -e: wir wollen weitermachen wenn ein Skip passiert

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Globale Skip-Liste: format "TITLE|||MANUAL_INSTRUCTIONS"
SKIPPED=()

# =============================================================================
# Detection
# =============================================================================
detect_distro() {
    if [ -f /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        case "$ID" in
            ubuntu|debian) echo "debian" ;;
            arch|manjaro|endeavouros) echo "arch" ;;
            *)
                case "${ID_LIKE:-}" in
                    *debian*) echo "debian" ;;
                    *arch*)   echo "arch" ;;
                    *)        echo "unknown" ;;
                esac
                ;;
        esac
    else
        echo "unknown"
    fi
}

detect_gpu() {
    local has_amd=0 has_nv=0
    if command -v lspci >/dev/null 2>&1; then
        if lspci | grep -iE "vga|3d|display" | grep -iqE "amd|advanced micro|radeon"; then
            has_amd=1
        fi
        if lspci | grep -iE "vga|3d|display" | grep -iq "nvidia"; then
            has_nv=1
        fi
    fi
    if [ "$has_amd" = "1" ] && [ "$has_nv" = "1" ]; then echo "both"
    elif [ "$has_amd" = "1" ]; then echo "amd"
    elif [ "$has_nv" = "1" ]; then echo "nvidia"
    else echo "none"
    fi
}

DISTRO="$(detect_distro)"
GPU_VENDOR="$(detect_gpu)"

# =============================================================================
# gum bootstrap (TUI)
# =============================================================================
ensure_gum() {
    if command -v gum >/dev/null 2>&1; then return 0; fi
    echo "[bootstrap] gum (TUI) wird installiert..."
    case "$DISTRO" in
        debian)
            sudo mkdir -p /etc/apt/keyrings
            curl -fsSL https://repo.charm.sh/apt/gpg.key | \
                sudo gpg --dearmor -o /etc/apt/keyrings/charm.gpg
            echo "deb [signed-by=/etc/apt/keyrings/charm.gpg] https://repo.charm.sh/apt/ * *" | \
                sudo tee /etc/apt/sources.list.d/charm.list >/dev/null
            sudo apt-get update -qq
            sudo apt-get install -y gum
            ;;
        arch)
            sudo pacman -S --needed --noconfirm gum
            ;;
        *)
            echo "ERROR: Unsupported distro: $DISTRO"
            exit 1
            ;;
    esac
}

# =============================================================================
# UI Helpers
# =============================================================================
ui_header() {
    gum style --foreground 212 --border-foreground 212 --border double \
        --align center --width 70 --margin "1 0" --padding "1 4" "$1"
}
ui_step()   { gum style --foreground 99  --bold "▶ $1"; }
ui_ok()     { gum style --foreground 46  "  ✓ $1"; }
ui_warn()   { gum style --foreground 220 "  ! $1"; }
ui_err()    { gum style --foreground 196 "  ✗ $1"; }
ui_info()   { gum style --foreground 244 "    $1"; }
ui_confirm(){ gum confirm --default=true "$1"; }
ui_run() {
    local title="$1"; shift
    [ "$1" = "--" ] && shift
    gum spin --spinner dot --title "$title" -- "$@"
}

skip_step() {
    # skip_step "Title" "Multi-line manual instructions"
    SKIPPED+=("$1|||$2")
    ui_warn "Uebersprungen: $1"
}

# =============================================================================
# Steps
# =============================================================================

# 0b. Storage-Auswahl: User darf .venv und models/ auf eine andere Disk
# auslagern (Symlinks im Repo zeigen dorthin).
#
# Drei Modi:
#   - "Symlinks bereits da"   -> skip stillschweigend
#   - "fresh"  (leeres Repo)  -> Symlinks auf leere Verzeichnisse (vor Install)
#   - "moved" (lokal installiert) -> Daten cross-device verschieben + Symlinks
choose_storage_location() {
    # Bereits Symlinks → fertig
    if [ -L "$ROOT/.venv" ] && [ -L "$ROOT/models" ]; then
        return 0
    fi

    local has_venv=0 has_models=0
    [ -e "$ROOT/.venv" ] && has_venv=1
    { [ -d "$ROOT/models/qwen3-asr" ] || [ -d "$ROOT/models/qwen3-tts-base" ]; } && has_models=1

    local mode="fresh"
    [ "$has_venv" = "1" ] || [ "$has_models" = "1" ] && mode="moved"

    # Mountpoint-Liste mit avail >= 30 GB
    local min_kb=$((30 * 1024 * 1024))
    declare -a opts paths
    local here_target here_kb here_gb
    here_target=$(df -k --output=target "$ROOT" | tail -1 | tr -d ' ')
    here_kb=$(df -k --output=avail "$ROOT" | tail -1 | tr -d ' ')
    here_gb=$((here_kb / 1024 / 1024))
    opts+=("Hier bleiben (Repo-Disk: $here_gb GB frei)")
    paths+=("$ROOT")
    while IFS= read -r line; do
        local target avail gb
        target=$(awk '{print $1}' <<<"$line")
        avail=$(awk '{print $2}' <<<"$line")
        [ "$target" = "$here_target" ] && continue
        if [ "$avail" -ge "$min_kb" ]; then
            gb=$((avail / 1024 / 1024))
            opts+=("$target  ($gb GB frei)")
            paths+=("$target")
        fi
    done < <(df -k --output=target,avail \
                | tail -n +2 \
                | grep -vE '/snap|/boot|tmpfs|/proc|/sys|/run|/dev|/var/lib/docker' \
                | sort -u)

    if [ ${#opts[@]} -le 1 ]; then
        # Nichts zur Auswahl
        return 0
    fi

    if [ "$mode" = "moved" ]; then
        ui_step "Storage-Ort: .venv + models verschieben?"
        local sz
        sz=$(du -sh "$ROOT/.venv" "$ROOT/models" 2>/dev/null | tail -1 | awk '{print $1}')
        ui_info "Aktuell im Repo (~$sz total)."
        ui_info "Auf andere Disk verschieben? Cross-Device-Copy, dauert je nach Disk."
        ui_warn "Loop muss VOR dem Verschieben gestoppt sein!"
        if ! ui_confirm "Auswahl jetzt anzeigen?"; then
            ui_info "Bleibt im Repo"
            return 0
        fi
    else
        ui_step "Storage-Ort fuer .venv (~15 GB) + models/ (~9 GB)"
    fi

    local choice
    choice=$(gum choose --header \
        "Wo sollen .venv + models/ liegen?" \
        "${opts[@]}")

    local chosen=""
    for i in "${!opts[@]}"; do
        [ "${opts[$i]}" = "$choice" ] && chosen="${paths[$i]}"
    done
    if [ -z "$chosen" ] || [ "$chosen" = "$ROOT" ]; then
        ui_ok "Bleibt im Repo"
        return 0
    fi

    local data_dir="$chosen/voiceagent-data"
    if [ -e "$data_dir" ] && [ "$(ls -A "$data_dir" 2>/dev/null)" ]; then
        ui_warn "$data_dir existiert bereits und ist nicht leer"
        if ! ui_confirm "Trotzdem benutzen? (vorhandener Inhalt bleibt)"; then
            ui_warn "Storage-Auswahl abgebrochen"
            return 0
        fi
    fi
    mkdir -p "$data_dir"

    if [ "$mode" = "fresh" ]; then
        # Leere Ziel-Verzeichnisse anlegen + Symlinks
        mkdir -p "$data_dir/venv" "$data_dir/models"
        if [ -d "$ROOT/models" ]; then
            [ -f "$ROOT/models/.gitkeep" ] && cp -n "$ROOT/models/.gitkeep" "$data_dir/models/"
            rm -f "$ROOT/models/.gitkeep"
            if ! rmdir "$ROOT/models" 2>/dev/null; then
                ui_err "models/ ist nicht leer — abbruch"
                return 1
            fi
        fi
        ln -s "$data_dir/venv"   "$ROOT/.venv"
        ln -s "$data_dir/models" "$ROOT/models"
    else
        # Vorhandene Daten verschieben (cross-device = copy + delete)
        if pgrep -f "python -m voiceagent" >/dev/null; then
            ui_err "Loop laeuft noch! Stoppe ihn erst:  pkill -f 'python -m voiceagent'"
            return 1
        fi
        if [ -d "$ROOT/.venv" ] && [ ! -L "$ROOT/.venv" ]; then
            ui_run "Verschiebe .venv -> $data_dir/venv (kann dauern)" -- \
                mv "$ROOT/.venv" "$data_dir/venv"
            ln -s "$data_dir/venv" "$ROOT/.venv"
        fi
        if [ -d "$ROOT/models" ] && [ ! -L "$ROOT/models" ]; then
            ui_run "Verschiebe models -> $data_dir/models (kann dauern)" -- \
                mv "$ROOT/models" "$data_dir/models"
            ln -s "$data_dir/models" "$ROOT/models"
        fi
    fi

    ui_ok "Symlinks angelegt:"
    ui_info "  $ROOT/.venv  ->  $data_dir/venv"
    ui_info "  $ROOT/models ->  $data_dir/models"
}

# 1. System-Pakete (immer noetig — keine Skip-Option)
install_system_packages() {
    ui_step "System-Pakete"
    case "$DISTRO" in
        debian)
            local pkgs=(ffmpeg python3 python3-venv python3-pip curl git pciutils)
            ui_info "Pakete: ${pkgs[*]}"
            sudo apt-get update -qq
            sudo apt-get install -y "${pkgs[@]}"
            ;;
        arch)
            local pkgs=(ffmpeg python python-pip curl git pciutils)
            ui_info "Pakete: ${pkgs[*]}"
            sudo pacman -S --needed --noconfirm "${pkgs[@]}"
            ;;
    esac
    ui_ok "System-Pakete installiert"
}

# 2a. AMD: nur ROCm-HIP-Runtime (das was PyTorch-ROCm wirklich braucht)
install_amd_rocm() {
    if command -v rocminfo >/dev/null 2>&1; then
        local rocm_ver
        rocm_ver=$(rocminfo 2>/dev/null | grep -i 'Runtime Version' | head -1 | awk '{print $NF}')
        ui_ok "ROCm Runtime bereits vorhanden ($rocm_ver)"
        return 0
    fi

    ui_warn "ROCm-Runtime nicht gefunden"
    ui_info "Minimaler Install: rocm-hip-runtime + Group-Membership (render/video)"
    ui_info "Reboot kann noetig sein wenn /dev/kfd neu erscheint."

    if ! ui_confirm "ROCm-Runtime jetzt installieren?"; then
        local manual
        case "$DISTRO" in
            debian) manual="
Debian/Ubuntu:
  curl -fsSL https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.2.60204-1_all.deb \\
       -o /tmp/amdgpu-install.deb
  sudo apt install -y /tmp/amdgpu-install.deb
  sudo amdgpu-install --usecase=hiplibsdk -y --no-dkms
  sudo usermod -aG render,video \$USER
  # ggf. reboot

Anschliessend install.sh erneut starten." ;;
            arch) manual="
Arch:
  sudo pacman -S --needed rocm-hip-runtime hip-runtime-amd
  sudo usermod -aG render,video \$USER
  # ggf. reboot

Anschliessend install.sh erneut starten." ;;
        esac
        skip_step "AMD ROCm-Runtime" "$manual"
        return 0
    fi

    case "$DISTRO" in
        debian)
            ui_info "Lade amdgpu-install Paket..."
            local tmpdeb=/tmp/amdgpu-install.deb
            curl -fsSL "https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.2.60204-1_all.deb" -o "$tmpdeb"
            sudo apt-get install -y "$tmpdeb"
            sudo amdgpu-install --usecase=hiplibsdk -y --no-dkms
            rm -f "$tmpdeb"
            ;;
        arch)
            sudo pacman -S --needed --noconfirm rocm-hip-runtime hip-runtime-amd
            ;;
    esac

    sudo usermod -aG render,video "$USER"
    ui_ok "ROCm Runtime installiert"
    ui_warn "Du musst dich evtl. neu einloggen oder rebooten (Group-Membership)"
}

# 2b. NVIDIA: nur Treiber (PyTorch-Wheel bringt CUDA mit, kein Toolkit noetig)
install_nvidia_driver() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local nv_ver
        nv_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        ui_ok "NVIDIA-Driver bereits installiert (v$nv_ver)"
        return 0
    fi

    ui_warn "NVIDIA-Driver nicht gefunden"
    ui_info "Reboot nach Treiber-Install ist normalerweise noetig."

    if ! ui_confirm "NVIDIA-Driver jetzt installieren?"; then
        local manual
        case "$DISTRO" in
            debian) manual="
Debian/Ubuntu:
  sudo apt install -y ubuntu-drivers-common
  sudo ubuntu-drivers autoinstall
  sudo reboot
Anschliessend install.sh erneut starten." ;;
            arch) manual="
Arch:
  sudo pacman -S nvidia nvidia-utils
  sudo reboot
Anschliessend install.sh erneut starten." ;;
        esac
        skip_step "NVIDIA-Driver" "$manual"
        return 0
    fi

    case "$DISTRO" in
        debian)
            sudo apt-get install -y ubuntu-drivers-common 2>/dev/null || true
            sudo ubuntu-drivers autoinstall
            ;;
        arch)
            sudo pacman -S --needed --noconfirm nvidia nvidia-utils
            ;;
    esac
    ui_ok "NVIDIA-Driver installiert"
    ui_warn "Reboot erforderlich, dann install.sh erneut starten"
}

install_gpu_stack() {
    ui_step "GPU-Stack (erkannt: $GPU_VENDOR)"
    case "$GPU_VENDOR" in
        amd)     install_amd_rocm ;;
        nvidia)  install_nvidia_driver ;;
        both)
            ui_info "AMD + NVIDIA gleichzeitig erkannt"
            local choice
            choice=$(gum choose --header "Welche GPU fuer VoiceAgent nutzen?" \
                "AMD (ROCm)" "NVIDIA (CUDA)")
            if [[ "$choice" == AMD* ]]; then
                GPU_VENDOR="amd"; install_amd_rocm
            else
                GPU_VENDOR="nvidia"; install_nvidia_driver
            fi
            ;;
        none)
            ui_err "Keine GPU erkannt"
            skip_step "GPU-Stack" "Keine kompatible GPU gefunden. VoiceAgent braucht eine
GPU fuer TTS+ASR. CPU-only ist zu langsam fuer Realtime."
            ;;
    esac
}

# 3. Ollama
install_ollama() {
    ui_step "Ollama LLM-Backend"
    if command -v ollama >/dev/null 2>&1; then
        ui_ok "Ollama vorhanden ($(ollama --version 2>/dev/null | head -1))"
    else
        if ! ui_confirm "Ollama jetzt installieren? (offizielles install.sh)"; then
            skip_step "Ollama" "
Manuell installieren:
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull gemma4:e4b
  ./scripts/make-modelfile.sh
Anschliessend install.sh erneut starten."
            return 0
        fi
        curl -fsSL https://ollama.com/install.sh | sh
        ui_ok "Ollama installiert"
    fi

    if ! ollama list 2>/dev/null | grep -q "^gemma4:e4b"; then
        if ! ui_confirm "gemma4:e4b (~6 GB) jetzt pullen?"; then
            skip_step "gemma4:e4b Modell" "
Manuell:
  ollama pull gemma4:e4b
  ./scripts/make-modelfile.sh"
            return 0
        fi
        ollama pull gemma4:e4b
        ui_ok "gemma4:e4b vorhanden"
    else
        ui_ok "gemma4:e4b bereits gepullt"
    fi

    if ! ollama list 2>/dev/null | grep -q "^gemma4-data"; then
        ui_info "Custom-Modelfile gemma4-data bauen..."
        if "$ROOT/scripts/make-modelfile.sh"; then
            ui_ok "gemma4-data Custom-Model erstellt"
        else
            skip_step "gemma4-data Modelfile" "
Manuell:
  ./scripts/make-modelfile.sh"
        fi
    else
        ui_ok "gemma4-data Custom-Model vorhanden"
    fi
}

# 4. Python venv
setup_venv() {
    ui_step "Python venv"
    if [ ! -d "$ROOT/.venv" ]; then
        ui_info "Erzeuge .venv mit $(python3 --version)"
        python3 -m venv "$ROOT/.venv"
    else
        ui_ok ".venv vorhanden"
    fi
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
    ui_run "pip + setuptools + wheel updaten" -- \
        pip install -q --upgrade pip wheel setuptools
    ui_ok "venv ready"
}

# 5. PyTorch (GPU-Branch). Auch installierbar wenn GPU-Stack uebersprungen.
install_pytorch() {
    ui_step "PyTorch ($GPU_VENDOR)"
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"

    local need=1 idx="" check=""
    case "$GPU_VENDOR" in
        amd)
            check="import torch; assert 'rocm' in torch.__version__"
            idx="https://download.pytorch.org/whl/rocm7.1"
            ;;
        nvidia)
            check="import torch; assert torch.version.cuda"
            idx="https://download.pytorch.org/whl/cu124"
            ;;
        none)
            ui_warn "Keine GPU — PyTorch wird nicht installiert"
            return 0
            ;;
    esac

    if python -c "$check" 2>/dev/null; then
        ui_ok "PyTorch $(python -c 'import torch; print(torch.__version__)') passend"
        return 0
    fi

    if ! ui_confirm "PyTorch fuer $GPU_VENDOR (~5 GB Download) jetzt installieren?"; then
        skip_step "PyTorch ($GPU_VENDOR)" "
Manuell:
  source .venv/bin/activate
  pip install torch torchaudio --index-url $idx"
        return 0
    fi

    pip install torch torchaudio --index-url "$idx"
    python -c "import torch; print(f'  torch={torch.__version__} hip={getattr(torch.version, \"hip\", None)} cuda={torch.version.cuda}')"
    ui_ok "PyTorch installiert"
}

# 6. Python deps (immer noetig — keine Skip-Option, klein)
install_python_deps() {
    ui_step "Python-Dependencies"
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
    ui_run "Audio + VAD + utils" -- pip install -q \
        numpy soundfile resampy librosa \
        pyyaml httpx silero-vad sounddevice num2words
    ui_run "Qwen3-TTS" -- pip install -q qwen-tts
    ui_run "Transformers (Qwen3-ASR)" -- pip install -q \
        transformers safetensors tokenizers
    ui_ok "Python-Dependencies installiert"
}

# 7. Modelle vorab cachen
# Check ob Modell schon im HF-Cache, project-models/, oder als legacy-checkout
# vorliegt — vermeidet unnoetiges Fragen wenn alles da ist.
_check_model_present() {
    # _check_model_present <hf_id>  -> echo "yes"|"no"
    local hf_id="$1"
    python - <<PY 2>/dev/null
from pathlib import Path
import sys

hf_id = "$hf_id"
project_root = Path("$ROOT")

# 1. Project-local checkouts (was tts.py / stt.py auch sucht)
local_candidates = [
    project_root / "models" / "qwen3-asr",
    project_root / "models" / "qwen3-tts-base",
]
for p in local_candidates:
    if p.exists() and (p / "config.json").exists():
        # heuristisch: ASR-id matched asr-dir, TTS-id matched tts-dir
        name = p.name.lower()
        if ("asr" in hf_id.lower() and "asr" in name) or \
           ("tts" in hf_id.lower() and ("tts" in name or "base" in name)):
            print("yes"); sys.exit(0)

# 2. HF Cache via try_to_load_from_cache (default cache + project models/hf)
from huggingface_hub import try_to_load_from_cache
import os
caches = [None,  # default ~/.cache/huggingface/hub
          str(project_root / "models" / "hf")]
for cache in caches:
    try:
        p = try_to_load_from_cache(hf_id, "config.json", cache_dir=cache)
        if p is not None and p != "_CACHED_NO_EXIST":
            print("yes"); sys.exit(0)
    except Exception:
        pass
print("no")
PY
}

predownload_models() {
    ui_step "Modelle vorab cachen"
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
    [ -f "$ROOT/scripts/env.sh" ] && source "$ROOT/scripts/env.sh"

    local asr_present tts_present
    asr_present=$(_check_model_present "Qwen/Qwen3-ASR-1.7B")
    tts_present=$(_check_model_present "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    if [ "$asr_present" = "yes" ] && [ "$tts_present" = "yes" ]; then
        ui_ok "Beide Modelle bereits vorhanden (Qwen3-ASR + Qwen3-TTS-Base)"
        return 0
    fi

    local missing=()
    [ "$asr_present" != "yes" ] && missing+=("Qwen3-ASR-1.7B (~1.5 GB)")
    [ "$tts_present" != "yes" ] && missing+=("Qwen3-TTS-Base (~3 GB)")
    ui_warn "Fehlt: ${missing[*]}"

    if ! ui_confirm "Jetzt vorab laden?"; then
        local manual="
Werden beim ersten ./scripts/start.sh automatisch gepullt.
Manuell:
  source .venv/bin/activate
  python -c \"from huggingface_hub import snapshot_download; \\
    snapshot_download('Qwen/Qwen3-ASR-1.7B'); \\
    snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base')\""
        skip_step "Modell-Pre-Download" "$manual"
        return 0
    fi

    # local_dir= statt cache_dir= — flaches Layout (config.json + safetensors
    # direkt im Verzeichnis), genau wie tts.py / stt.py es suchen.
    if [ "$tts_present" != "yes" ]; then
        ui_run "Qwen3-TTS-Base ziehen (-> models/qwen3-tts-base/)" -- python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base',
                  local_dir='$ROOT/models/qwen3-tts-base')
"
    fi
    if [ "$asr_present" != "yes" ]; then
        ui_run "Qwen3-ASR-1.7B ziehen (-> models/qwen3-asr/)" -- python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-ASR-1.7B',
                  local_dir='$ROOT/models/qwen3-asr')
"
    fi
    ui_ok "Modelle in models/qwen3-tts-base/ + models/qwen3-asr/ verfuegbar"
}

# 8. Verify
verify_install() {
    ui_step "Verifikation"
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
    [ -f "$ROOT/scripts/env.sh" ] && source "$ROOT/scripts/env.sh"

    python - <<'PY'
import sys
results = []
def chk(label, fn):
    try:
        v = fn(); results.append(("OK", label, str(v)))
    except Exception as e:
        results.append(("FAIL", label, repr(e)[:80]))

try:
    import torch
    chk("torch", lambda: torch.__version__)
    chk("GPU detect", lambda: torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
except Exception as e:
    results.append(("FAIL", "torch", repr(e)[:80]))
chk("qwen-tts", lambda: __import__("qwen_tts").__file__)
chk("transformers", lambda: __import__("transformers").__version__)
chk("silero-vad", lambda: __import__("silero_vad").__version__)
chk("sounddevice", lambda: __import__("sounddevice").__version__)
chk("num2words", lambda: __import__("num2words").__version__)

for status, label, val in results:
    sym = "✓" if status == "OK" else "✗"
    print(f"  {sym} {label:20s}  {val[:60]}")
PY
}

# =============================================================================
# Wait-on-exit (damit Doppelklick-Terminals nicht zu schnell zumachen)
# =============================================================================
wait_for_user() {
    echo
    gum style --foreground 244 "Druecke ENTER zum Beenden..."
    # /dev/tty fallback wenn stdin redirected ist; 5 min timeout als Sicherung
    if read -r -t 300 _ </dev/tty 2>/dev/null; then :; fi
}

# =============================================================================
# Final Skip Report
# =============================================================================
print_skipped_report() {
    if [ ${#SKIPPED[@]} -eq 0 ]; then
        return 0
    fi

    ui_header "Uebersprungene Schritte — manuelle Aktionen erforderlich"
    local i=1
    for item in "${SKIPPED[@]}"; do
        local title="${item%%|||*}"
        local instr="${item#*|||}"
        gum style --foreground 220 --bold "  $i) $title"
        echo "$instr" | sed 's/^/     /'
        echo
        i=$((i+1))
    done
    ui_warn "Nach Erledigung: ./install.sh erneut starten."
}

# =============================================================================
# Main
# =============================================================================
main() {
    if [ "$DISTRO" = "unknown" ]; then
        echo "ERROR: Unsupported distro. Supported: Ubuntu, Debian, Arch, Manjaro."
        exit 1
    fi

    ensure_gum

    ui_header "VoiceAgent Installer"
    ui_info "Distro: $DISTRO  |  GPU: $GPU_VENDOR  |  Root: $ROOT"
    echo

    install_system_packages
    install_gpu_stack
    install_ollama
    choose_storage_location
    setup_venv
    install_pytorch
    install_python_deps
    predownload_models
    verify_install

    if [ ${#SKIPPED[@]} -eq 0 ]; then
        ui_header "Installation komplett"
        gum format <<EOF
**Start:**

\`\`\`
./scripts/start.sh
\`\`\`

**Wake-Phrasen** (default Wake-Word: \`data\`):

| Sage | Persona | Voice |
|---|---|---|
| \`hey data\` | data | data |
| \`hey sam\` | samwell-emo | samwell |
| \`hey sora\` | sora | sora |

**Eigene Stimme:** WAV nach \`reference_audio/<name>.wav\`, dann
\`./scripts/start.sh --voice <name>\`

**Audio-Devices listen:** \`./scripts/start.sh --list-devices\`
EOF
    else
        ui_header "Installation teilweise abgeschlossen"
        ui_info "Was lief: System-Pakete, gum, was du bestaetigt hast."
        print_skipped_report
    fi

    wait_for_user
}

main "$@"
