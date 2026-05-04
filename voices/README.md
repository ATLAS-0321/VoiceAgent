# Voices

A voice = one subfolder containing everything that belongs together:
audio sample, persona (LLM system prompt), and optionally an ICL transcript.

## Layout

```
voices/
├── Anna/
│   ├── Anna.wav        # voice reference (mono, 10–30 s, clean)
│   ├── persona.txt     # LLM system prompt (character + emotion tags + rules)
│   └── Anna.txt        # optional: ICL transcript of the WAV
├── Peter/
│   ├── Peter.mp3
│   └── persona.txt
└── ...
```

Voice ID = folder name. In `config.yaml`: `tts.voice: Anna`.

## Sample requirements

- Format: **WAV** (16-bit PCM preferred) or **MP3**
- Duration: **10–30 seconds**
- Content: only **one** speaker
- Quality: clean — no music, no background noise, no reverb
- Sample rate: any (Qwen3-TTS resamples internally)

## persona.txt

Plain-text system prompt. Should include:

- Character description (who am I? how do I speak?)
- Address rules (Sir / Mylord / Captain / by name)
- World/context (what does the persona know, what not?)
- Tag block listing the allowed `<style:NAME>` emotion tags
- Standard rules (max. 2–3 sentences, no markdown, no special chars)

Examples in `voices/Data_Sample-Set/persona.txt` and the others.

## Creating a new voice with clean-reference.py

Turn a dirty source (video, MP3, with music or multiple speakers) into
a clean reference — automatically lands in the right place:

```bash
python scripts/clean-reference.py /path/to/source.mp4 --voice Anna
```

Creates `voices/Anna/Anna.wav`. The Demucs workspace lives in
`/tmp/voiceagent-clean/<stem>/`, NOT inside `voices/`.

Afterwards create `voices/Anna/persona.txt` manually (use one of the
existing personas as a template).

## Tips

- One clean recording beats several mediocre ones.
- A recording from the **same microphone** you'll use for output gives
  more consistent results.
- Reading text with normal sentence melody works better than monotone
  enumeration.

## License notice

Only clone voices of other people with their permission.
