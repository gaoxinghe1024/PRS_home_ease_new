# Post-hoc diagnostic (not part of the run)

Produced after the run finished and after the test split was scored, by: python -m homeease.experiments.audio_intent rescore <run_dir>. It selects nothing.

### Rule v2 without the cues that only one split's wording triggers (post-selection diagnostic)

| split | input | cues switched off | macro_f1 full | macro_f1 without | self HELP missed full | self HELP missed without |
|---|---|---|---|---|---|---|
| dev | reference text (no ASR) | 9 | 1.000 | 0.948 | 0/38 | 4/38 |
| dev | ASR transcript (tiny|none|auto) | 9 | 0.940 | 0.898 | 7/38 | 10/38 |
| test | reference text (no ASR) | 13 | 0.990 | 0.914 | 1/38 | 9/38 |
| test | ASR transcript (tiny|none|auto) | 13 | 0.887 | 0.829 | 5/38 | 13/38 |

Read as a range: the true generalisation of a lexicon written without sight of the data lies between 'without' (too pessimistic: ordinary words such as pain / slipped are removed) and 'full' (optimistic).

