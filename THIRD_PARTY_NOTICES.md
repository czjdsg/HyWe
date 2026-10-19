# Third-party notices

Original release code is Apache-2.0. The following portions retain upstream
licenses and copyright notices.

| Source | Included/adapted material | License |
| --- | --- | --- |
| [Online-Mind2Web](https://github.com/OSU-NLP-Group/Online-Mind2Web) | `evaluation/online_mind2web/` WebJudge prompts, methods, and runner | MIT, `LICENSES/Online-Mind2Web-MIT.txt` |
| [WebVoyager](https://github.com/MinorJerry/WebVoyager) | WebVoyager task set and evaluator prompt adapted for hybrid evidence | Apache-2.0, `LICENSES/WebVoyager-Apache-2.0.txt` |
| [MobileAgent](https://github.com/X-PLUG/MobileAgent) | `vendor/mobile_agent_browser_use/`, browser/SoM helpers and adapted runtime | MIT, copyright mPLUG, `LICENSES/MobileAgent-MIT.txt` |
| [InSTA-150k-v3](https://huggingface.co/datasets/data-for-agents/insta-150k-v3) | Screened site lists and derived task-design metadata in `collector/data/` | MIT as declared in its dataset card |

Changes to the operation evaluators add hybrid text observations, portable
image transport, robust response parsing, blocking annotations, configurable
endpoints, and small-batch handling. Their adapted prompts are not presented as
unmodified upstream evaluations.

`vendor/mobile_agent_browser_use/browser/playwright/Arial.ttf` is a legacy
third-party font bundled with the upstream browser code. It is omitted from
this release; the SoM helper uses a system font or Pillow's fallback.

Third-party benchmarks downloaded by `benchmarks/prepare_benchmarks.py` retain
their original access conditions and canaries. The converter does not grant
additional redistribution rights. Model weights retain their model-card
license. Training datasets keep their source attribution in their HF cards.
