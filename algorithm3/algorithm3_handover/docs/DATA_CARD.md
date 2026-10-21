# Data card: data and splits used by Algorithm 3

The package does not include the training and evaluation datasets, only the 16 short recordings in `samples/`. This card says where the data came from, how it was used and where its limits are.

## Sound events (training and evaluating the sound model)

| Data | Content | Use | Licence |
|---|---|---|---|
| ESC-50 (Piczak 2015) | 2,000 clips of 5 s, 50 classes, 5 official folds | IMPACT (door knock, as a stand-in for an impact), ABNORMAL (glass breaking, crying baby, siren, alarm clock), DAILY (19 household classes) | CC BY-NC 3.0 |
| VIVAE (Holz et al. 2022) | 1,085 acted non-speech vocalisations, 11 speakers, 6 emotions, 4 intensities | HELP: fear and pain at medium, strong and peak intensity (277 clips); the 84 low-intensity clips are neither trained nor scored; anger goes to ABNORMAL (174), achievement, pleasure and surprise to DAILY (550) | CC BY-NC 4.0 |
| FSD50K subset | 265 clips, 171 uploaders, downloaded and checked file by file | HELP: label Screaming (130 clips); negatives from the same corpus: laughter, cough, sneeze as DAILY (65), heavy fall and knock as IMPACT (45), breaking as ABNORMAL (25) | CC0 / CC BY / CC BY-NC per clip |
| Mini LibriSpeech dev-clean-2 | 1,089 sentences, 26 speakers | the DAILY "speech" class (156 windows), television speech, evaluation of the speech gate and of speaker verification | CC BY 4.0 |

- Windows are 2 s long, RMS-normalised to 0.1; each window comes from one source clip. There are 2,382 windows (HELP 407, IMPACT 85, ABNORMAL 359, DAILY 1,531).
- Splits are by group: the original recording for ESC-50, the speaker for VIVAE, the uploader for FSD50K, the speaker for LibriSpeech; 5 folds. The shipped model is trained on folds 1 to 4 and tested on fold 5 (466 windows, 212 groups).
- The label mapping was fixed before training. Negatives from the same corpus and a leave-one-corpus-out check test whether the model learned the corpus instead of the sound.
- Clips shorter than 2 s are mostly padding inside a window, and the share of silent frames alone separates HELP from the longer clips (AUC 0.785). This is a known uncontrolled cue.
- 14 Freesound recordings appear in both ESC-50 and FSD50K (none is HELP); 11 of them fall into different folds. Removing them changes macro-F1 by 0.001.

## Reply recognition

- A phrase bank written for the project: 312 Chinese and English sentences, each labelled with the intent (help, reassurance, other), the speaker role (the person, a visitor, television), a paraphrase family and the split. Split by family: 78 families for development and 78 for test.
- Rendered with system speech synthesis (Zira, David, Huihui; the Mandarin visitor and television voices are Huihui resampled), partly mixed with 5 dB household noise. Real LibriSpeech sentences serve as television. Each split has 208 items, 196 scored and 12 appliance-noise items that test the recogniser for hallucination.
- 20 more calls for help and 15 reassurances were written by one reviewer after the rules were done and serve only as a text probe, not for tuning the lexicon.
- The lexicon was written after both splits had been seen: 13 cues fire only in test wording, so 0.894 is an upper bound.

## Speaker verification and false cancels (A4, A6)

- A4: LibriSpeech speakers are split by person into a development and an evaluation half; the evaluation has 13 speakers, 268 genuine trials and 3,216 impostor trials. 3,216 impostor trials are not enough to estimate a false-accept rate of 0.1% (at least 10,000 are needed).
- A6: 150 development and 150 evaluation speakers (disjoint) drawn at random from the 904 synthetic speakers of Piper libritts_r. Each enrols with the same three neutral sentences and then says the phrases of the phrase bank; with 5 dB household noise, loudspeaker and room simulation (television, replay), household sounds and silence, this gives 2,370 evaluation items. The threshold is set on the development speakers only.
- Everything is synthetic English, clean, and from one family of synthesis engines.

## Samples (`samples/`)

- 6 scenes (20 to 24 s): washing-machine background (-48 dBFS, ESC-50) with an impact, a scream, breaking glass, television speech and replies. Every source clip comes from fold 5 (the fold the shipped model has not seen); `scene_*.json` records the source, fold, licence and hash of each clip.
- 10 short replies made with system speech synthesis, to show a call for help, reassurance, an unrelated remark, thanks, a long television sentence and silence.
- The scenes mix CC BY-NC and CC BY clips and are therefore for non-commercial course research demonstrations only.

## What these data cannot show

There are no older adults, dialects and accents, real television and visitors, real room acoustics, microphone distances or continuous home recordings (so no false alarms per hour). All numbers describe the methods on these data, not the product in a home.
