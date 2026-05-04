"""Entry point: python -m voiceagent [--voice NAME] [--config PATH]"""
from __future__ import annotations

import argparse
import logging
import sys

from . import audio
from .config import load, load_persona, list_voices


def main(argv=None):
    p = argparse.ArgumentParser(prog="voiceagent")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--voice", default=None,
                   help="Override tts.voice (subfolder name in voices/)")
    p.add_argument("--list-devices", action="store_true",
                   help="Print PortAudio devices and exit")
    p.add_argument("--list-voices", action="store_true",
                   help="Print available voices and exit")
    args = p.parse_args(argv)

    if args.list_devices:
        print(audio.list_devices())
        return 0

    cfg = load(args.config)

    if args.list_voices:
        for v in list_voices(cfg.root, cfg.tts.voices_dir):
            print(v)
        return 0

    if args.voice is not None:
        cfg.tts.voice = args.voice
        # Re-resolve persona aus dem neuen Voice-Ordner
        cfg.llm.system_prompt = load_persona(cfg.root, cfg.tts.voices_dir, cfg.tts.voice)

    logging.basicConfig(
        level=getattr(logging, cfg.system.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from .loop import VoiceLoop
    loop = VoiceLoop(cfg)
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nbye.")
        return 0
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
