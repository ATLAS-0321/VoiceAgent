# Voices

Eine Voice = ein Subfolder mit allem was dazugehoert: Audio-Sample,
Persona (LLM System-Prompt) und optional ICL-Transkript.

## Struktur

```
voices/
├── Anna/
│   ├── Anna.wav        # Voice-Reference (mono, 10-30 s, sauber)
│   ├── persona.txt     # LLM System-Prompt (Charakter, Emotion-Tags, Regeln)
│   └── Anna.txt        # optional: ICL-Transkript der WAV
├── Peter/
│   ├── Peter.mp3
│   └── persona.txt
└── ...
```

Voice-ID = Ordnername. In `config.yaml`: `tts.voice: Anna`.

## Anforderungen pro Sample

- Format: **WAV** (16-bit PCM bevorzugt) oder **MP3**
- Dauer: **10-30 Sekunden**
- Inhalt: nur **eine** Person spricht
- Qualitaet: clean, keine Musik, keine Hintergrundgeraeusche, kein Hall
- Sample-Rate: egal (Qwen3-TTS resampled selbst)

## persona.txt

Plain-Text System-Prompt. Pflicht:

- Charakter-Beschreibung (Wer bin ich? Wie spreche ich?)
- Anrede-Regeln
- Welt/Kontext (was kennt die Persona, was nicht?)
- Tag-Block mit erlaubten `<style:NAME>`-Emotions-Tags
- Standard-Regeln (max. 2-3 Saetze, keine Markdown, keine Sonderzeichen)

Beispiele in `voices/Data_Sample-Set/persona.txt` u.a.

## Neue Voice mit clean-reference.py

Aus dirty Source (Video, MP3, mit Musik / mehreren Sprechern) eine
saubere Reference erzeugen — landet automatisch hier:

```
python scripts/clean-reference.py /pfad/zu/quelle.mp4 --voice Anna
```

Erstellt `voices/Anna/Anna.wav`. Demucs-Workspace landet in
`/tmp/voiceagent-clean/<stem>/`, NICHT in `voices/`.

Anschliessend `voices/Anna/persona.txt` von Hand anlegen (Vorlage:
ein bestehender persona.txt von Data/Sam/Sora kopieren und anpassen).

## Tipps

- Lieber **eine** saubere Aufnahme als mehrere durchschnittliche.
- Aufnahme aus dem **gleichen Mikro** wie der spaetere Output bringt
  konsistentere Ergebnisse.
- Vorlesetext mit normaler Sprechmelodie funktioniert besser als
  monotone Aufzaehlung.

## Lizenzhinweis

Stimmen anderer Personen nur mit deren Einverstaendnis klonen.
