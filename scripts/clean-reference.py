#!/usr/bin/env python3
"""Pipeline: dirty source audio (mit Musik / Katze / mehreren Sprechern)
        ->  clean Voice-Cloning-Reference fuer Qwen3-TTS.

Steps:
  1. ffmpeg-extract       (Video -> mono 44.1 kHz WAV)
  2. Demucs separation    (Musik raus, nur Vocals)
  3. VAD-segment + cluster (Silero-VAD + Resemblyzer + KMeans)
  4. INTERACTIVE          (User waehlt: welcher Cluster ist die Zielstimme?)
  5. Concat               (laengste / cleansten Segmente bis target duration)

Output-Konvention (Subfolder pro Voice):
    --voice Anna  -> voices/Anna/Anna.wav

Usage:
    # Empfohlen — legt Anna/Anna.wav automatisch an:
    python scripts/clean-reference.py /pfad/zu/source.mp4 --voice Anna

    # Oder voller Output-Pfad explizit:
    python scripts/clean-reference.py /pfad/zu/source.mp4 \\
        --out voices/Anna/Anna.wav --target-duration 25

Work-Dir (Demucs/Cluster-Cache, ~5 GB pro Run) landet per default unter
/tmp/voiceagent-clean/<stem>/ — NICHT mehr im voices-Ordner.
Mit --keep-work bleibt es nach dem Run liegen.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def fail(msg: str, code: int = 1):
    print(f"\n  FAIL: {msg}", file=sys.stderr)
    sys.exit(code)


def step(n: int, title: str):
    bar = "=" * 64
    print(f"\n{bar}\n  Step {n}: {title}\n{bar}")


def fmt_t(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:05.2f}"


# ---------- step 1: ffmpeg extract -----------------------------------------

def extract_audio(src: Path, dst: Path, sr: int = 44100):
    if not shutil.which("ffmpeg"):
        fail("ffmpeg not on PATH")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-ac", "1", "-ar", str(sr), "-vn", str(dst)]
    subprocess.check_call(cmd)
    info = sf.info(str(dst))
    print(f"  extracted: {dst} ({info.duration:.1f} s @ {info.samplerate} Hz, mono)")


# ---------- step 2: demucs vocals separation -------------------------------

def demucs_vocals(src: Path, work: Path, device: str) -> Path:
    print(f"  running demucs (htdemucs) on {device} ...")
    cmd = ["python", "-m", "demucs.separate",
           "--two-stems=vocals", "-n", "htdemucs",
           "-d", device, "-o", str(work), str(src)]
    subprocess.check_call(cmd)
    out = work / "htdemucs" / src.stem / "vocals.wav"
    if not out.exists():
        fail(f"Demucs output not found at {out}")
    print(f"  vocals: {out}")
    return out


# ---------- step 3: VAD-segment + speaker-embed + cluster -----------------

def segment_and_cluster(vocals: Path, num_speakers: int | None,
                        device: str, min_seg_s: float = 1.5):
    """Split vocals into VAD-derived segments, embed each with Resemblyzer,
    then KMeans-cluster. Returns: (segments, labels, audio16, sr=16000).
    `segments` is a list of (start_s, end_s) in seconds.
    """
    from silero_vad import load_silero_vad, get_speech_timestamps
    from resemblyzer import VoiceEncoder, preprocess_wav
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import resampy

    audio, sr = sf.read(str(vocals), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio16 = audio if sr == 16000 else resampy.resample(audio, sr, 16000).astype("float32")

    print("  running Silero VAD ...")
    vad_model = load_silero_vad()
    ts = get_speech_timestamps(
        torch.from_numpy(audio16), vad_model,
        sampling_rate=16000, threshold=0.5,
        min_speech_duration_ms=int(min_seg_s * 1000),
        min_silence_duration_ms=300,
    )
    if not ts:
        fail("Silero VAD found no speech.")
    segments_s = [(t["start"] / 16000.0, t["end"] / 16000.0) for t in ts]
    print(f"  VAD: {len(segments_s)} speech segments "
          f"(total {sum(e-s for s,e in segments_s):.1f} s)")

    print("  embedding segments with Resemblyzer ...")
    enc = VoiceEncoder(device)
    embeds = []
    keep_segments = []
    for s, e in segments_s:
        seg = audio16[int(s*16000):int(e*16000)]
        # Resemblyzer expects float32 mono in -1..1 at any sr; preprocess_wav resamples
        wav = preprocess_wav(seg, source_sr=16000)
        if wav.size < 16000:  # < 1 s after preprocess_wav trim
            continue
        embeds.append(enc.embed_utterance(wav))
        keep_segments.append((s, e))
    if not embeds:
        fail("No segments survived embedding (all too short).")
    X = np.stack(embeds, axis=0)
    print(f"  embeddings: {X.shape}")

    if num_speakers is None:
        # auto-pick best k via silhouette over [2..6]
        best_k, best_score = 2, -1.0
        for k in range(2, min(7, len(X))):
            km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
            if len(set(km.labels_)) < 2:
                continue
            sc = silhouette_score(X, km.labels_)
            if sc > best_score:
                best_score = sc
                best_k = k
        print(f"  auto-K via silhouette: k={best_k} (score {best_score:.3f})")
        num_speakers = best_k

    print(f"  KMeans clustering (k={num_speakers}) ...")
    km = KMeans(n_clusters=num_speakers, n_init="auto", random_state=0).fit(X)
    labels = km.labels_

    # Group segments per label
    by_label: dict[int, list[tuple[float, float]]] = {}
    for (s, e), lab in zip(keep_segments, labels):
        by_label.setdefault(int(lab), []).append((s, e))

    return by_label, audio16, 16000


# ---------- step 4: interactive cluster picker ----------------------------

def preview_clusters(by_label, audio16, sr, preview_dir: Path,
                     n_snippets: int = 3) -> int:
    preview_dir.mkdir(parents=True, exist_ok=True)
    print("\n  Per-cluster stats and previews:")
    print()
    sorted_labels = sorted(by_label.keys(),
                           key=lambda k: -sum(e-s for s,e in by_label[k]))
    for lab in sorted_labels:
        segs = by_label[lab]
        total = sum(e-s for s,e in segs)
        longest = sorted(segs, key=lambda se: se[1]-se[0], reverse=True)[:n_snippets]
        chunks = []
        budget = 12.0
        for s, e in longest:
            dur = min(e-s, 4.0, budget)
            chunks.append(audio16[int(s*sr):int((s+dur)*sr)])
            budget -= dur
            if budget <= 0:
                break
        preview = np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")
        out = preview_dir / f"cluster_{lab}.wav"
        sf.write(str(out), preview, sr)
        print(f"    cluster {lab}: {len(segs):3d} segments, total {total:6.1f} s, "
              f"longest {longest[0][1]-longest[0][0]:5.1f} s  ->  {out.name}")

    print()
    print(f"  Listen to each preview file:")
    print(f"    for f in {preview_dir}/cluster_*.wav; do echo $f; paplay $f; done")
    print()
    while True:
        ans = input(f"  Welcher Cluster ist die Zielstimme? "
                    f"({sorted(by_label.keys())}): ").strip()
        try:
            n = int(ans)
            if n in by_label:
                return n
        except ValueError:
            pass
        print(f"    '{ans}' nicht in {sorted(by_label.keys())} — nochmal.")


# ---------- step 5: VAD-trim + concat ------------------------------------

def concat_segments(audio16: np.ndarray, segments: list[tuple[float, float]],
                    target_s: float, out: Path):
    """Concat the longest segments first, with short cross-fades, until target."""
    sr = 16000
    # sort by length, longest first
    segments = sorted(segments, key=lambda se: -(se[1]-se[0]))
    picked: list[np.ndarray] = []
    total = 0
    target_samples = int(target_s * sr)
    for s, e in segments:
        if total >= target_samples:
            break
        clip = audio16[int(s*sr):int(e*sr)]
        picked.append(clip)
        total += clip.size
    if not picked:
        fail("No segments picked (target_duration too small?)")

    xfade = int(0.05 * sr)
    out_buf: list[np.ndarray] = []
    for i, clip in enumerate(picked):
        if i == 0:
            out_buf.append(clip)
        else:
            prev = out_buf[-1]
            if prev.size > xfade and clip.size > xfade:
                fade_out = np.linspace(1.0, 0.0, xfade, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, xfade, dtype=np.float32)
                prev[-xfade:] = prev[-xfade:] * fade_out + clip[:xfade] * fade_in
                out_buf.append(clip[xfade:])
            else:
                out_buf.append(clip)
    final = np.concatenate(out_buf)

    import resampy
    final_22 = resampy.resample(final, 16000, 22050).astype("float32")
    peak = float(np.max(np.abs(final_22))) or 1.0
    final_22 = (final_22 / peak) * 0.891  # ~ -1 dBFS

    sf.write(str(out), final_22, 22050)
    print(f"  wrote {out} ({len(final_22)/22050:.2f} s, {len(picked)} clips combined)")


# ---------- main ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="Source file (audio or video)")
    ap.add_argument("--voice", default=None,
                    help="Voice-ID -> Output landet automatisch in "
                         "voices/<voice>/<voice>.wav")
    ap.add_argument("--out", default=None,
                    help="Expliziter Output-WAV-Pfad (alternativ zu --voice)")
    ap.add_argument("--target-duration", type=float, default=25.0,
                    help="Target total clean speech length (default: 25 s)")
    ap.add_argument("--min-segment", type=float, default=1.5,
                    help="Min length per kept VAD segment (default: 1.5 s)")
    ap.add_argument("--num-speakers", type=int, default=None,
                    help="Force fixed number of speaker clusters "
                         "(default: auto-detect via silhouette)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--work-dir", default=None,
                    help="Work-Verzeichnis fuer Demucs/Cluster-Cache "
                         "(default: /tmp/voiceagent-clean/<stem>/)")
    ap.add_argument("--keep-work", action="store_true",
                    help="Work-Dir nach dem Run NICHT loeschen")
    ap.add_argument("--cluster", type=int, default=None,
                    help="Phase 2: skip prep + use this cluster id (must "
                         "have been generated by a previous Phase 1 run).")
    args = ap.parse_args()

    src = Path(args.source).expanduser().resolve()
    if not src.exists():
        fail(f"Source not found: {src}")

    # Output-Pfad: --voice (Convention) ODER --out (explizit). Mindestens eins.
    if args.voice and args.out:
        fail("Nutze entweder --voice ODER --out, nicht beides")
    if not args.voice and not args.out:
        fail("Brauche --voice <name> ODER --out <pfad>")
    if args.voice:
        project_root = Path(__file__).resolve().parent.parent
        out = project_root / "voices" / args.voice / f"{args.voice}.wav"
    else:
        out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Work-Dir: per default in /tmp damit reference_audio sauber bleibt
    if args.work_dir:
        work = Path(args.work_dir).expanduser().resolve()
    else:
        work = Path("/tmp/voiceagent-clean") / src.stem
    work.mkdir(parents=True, exist_ok=True)
    state_pkl = work / "state.npz"
    preview_dir = work / "previews"

    print(f"  source : {src}")
    print(f"  out    : {out}")
    print(f"  device : {args.device}")
    print(f"  target : {args.target_duration} s")

    if args.cluster is None:
        # ---- Phase 1: extract, demucs, cluster, write previews ----
        step(1, "Extract audio (ffmpeg, mono 44.1 kHz)")
        extracted = work / "extracted.wav"
        extract_audio(src, extracted, sr=44100)

        step(2, "Separate vocals (Demucs htdemucs)")
        vocals = demucs_vocals(extracted, work, device=args.device)

        step(3, "Segment + speaker-embed + cluster (Silero-VAD + Resemblyzer + KMeans)")
        by_label, audio16, sr = segment_and_cluster(
            vocals, args.num_speakers, args.device, args.min_segment,
        )

        step(4, "Write cluster previews + cache state for Phase 2")
        preview_dir.mkdir(parents=True, exist_ok=True)
        sorted_labels = sorted(by_label.keys(),
                               key=lambda k: -sum(e-s for s,e in by_label[k]))
        for lab in sorted_labels:
            segs = by_label[lab]
            total = sum(e-s for s,e in segs)
            longest = sorted(segs, key=lambda se: se[1]-se[0], reverse=True)[:3]
            chunks = []
            budget = 12.0
            for s, e in longest:
                dur = min(e-s, 4.0, budget)
                chunks.append(audio16[int(s*sr):int((s+dur)*sr)])
                budget -= dur
                if budget <= 0:
                    break
            preview = np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")
            sf.write(str(preview_dir / f"cluster_{lab}.wav"), preview, sr)
            print(f"    cluster {lab}: {len(segs):3d} segments, total {total:6.1f} s, "
                  f"longest {longest[0][1]-longest[0][0]:5.1f} s")

        # Persist state for Phase 2
        labels = np.array(sorted(by_label.keys()), dtype=np.int32)
        flat_segs = np.array(
            [(lab, s, e) for lab in by_label for (s, e) in by_label[lab]],
            dtype=np.float64,
        )
        np.savez(str(state_pkl), audio16=audio16, sr=np.int32(sr),
                 labels=labels, flat_segs=flat_segs)

        print()
        print("  PHASE 1 DONE. Listen to previews:")
        for lab in sorted_labels:
            print(f"     paplay {preview_dir / f'cluster_{lab}.wav'}")
        print()
        print("  Then run Phase 2 with the chosen cluster id, e.g.:")
        if args.voice:
            cmd = (f"  python scripts/clean-reference.py '{src}' "
                   f"--voice {args.voice} "
                   f"--target-duration {args.target_duration} --cluster <ID>")
        else:
            cmd = (f"  python scripts/clean-reference.py '{src}' "
                   f"--out '{out}' --target-duration {args.target_duration} "
                   f"--cluster <ID>")
        print(cmd)
        return

    # ---- Phase 2: pick chosen cluster, concat, finalize ----
    if not state_pkl.exists():
        fail(f"State file missing: {state_pkl}\n  Run Phase 1 first (no --cluster).")

    z = np.load(str(state_pkl))
    audio16 = z["audio16"]
    sr = int(z["sr"])
    flat = z["flat_segs"]
    target_segs = [(float(s), float(e)) for (lab, s, e) in flat
                   if int(lab) == args.cluster]
    if not target_segs:
        fail(f"No segments for cluster {args.cluster}. "
             f"Available: {sorted(set(int(x) for x in flat[:,0]))}")

    total_s = sum(e-s for s,e in target_segs)
    print(f"\n  selected cluster {args.cluster}: {len(target_segs)} segments, "
          f"total {total_s:.1f} s")

    step(5, f"Concat to {args.target_duration} s with cross-fades")
    concat_segments(audio16, target_segs, args.target_duration, out)

    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
        print(f"\n  cleaned: {work}")
    else:
        print(f"\n  work kept at: {work}")

    print(f"\n  DONE. Reference written to: {out}")
    print(f"        Use it: in config.yaml set tts.voice = {out.stem}")


if __name__ == "__main__":
    main()
