# Changelog

All notable changes to this project will be documented in this file.

## [0.17.0] - 2026-08-06

### Added
- **UseapiMurekaAdvanced** — custom lyrics + style/vocal/ref
- **UseapiFaceswapInswapper** — INSwapper via Discord message id

### Note
Creative generation surface for Meta/media workflows is complete.
Account CRUD UIs intentionally out of scope for Comfy node packs.

## [0.16.0] - 2026-08-06

### Added
- **UseapiMurekaRegenerate** — regenerate from start offset
- **UseapiFaceswapChangeBG** — background replace via text prompt

## [0.15.0] - 2026-08-06

### Added
- **UseapiMurekaInstrumental** — instrumental create with async poll
- **UseapiMurekaExtend** — extend song with new lyrics
- Shared `_mureka_create_and_poll` helper

## [0.14.0] - 2026-08-06

### Added
- **UseapiMurekaCreate** — Mureka song create (async poll, dual tracks)
- **UseapiFaceswapPicsi** — InsightFaceSwap /picsi face morph

## [0.13.0] - 2026-08-06

### Added
- **UseapiPixverseUploadFile** — upload image/video/audio for PixVerse ops
- **UseapiPixverseLipsync** — lip-sync with audio or TTS prompt
- **UseapiPixverseMotionControl** — character image + motion video
- **UseapiPixverseExtend** — extend v6 / grok-imagine videos

## [0.12.0] - 2026-08-06

### Added
- **UseapiKlingTTS** — free Kling text-to-speech (audio URL + path + AUDIO)
- **UseapiKlingAvatarVideo** — Avatars 2.0 talking-head (image/avatar + audio or TTS)

## [0.11.0] - 2026-08-06

### Added
- **UseapiKlingUploadAsset** — upload image/video/audio to Kling assets
- **UseapiKlingImage2Video** — image2video-frames (v3/turbo/2.x)
- **UseapiKlingLipsync** — video+audio lipsync
- **UseapiKlingMotionCreate** — motion-control (v3.0 / v2.6)

## [0.10.0] - 2026-08-06

### Added
- **UseapiKlingText2Video** — Kling v1 text2video (v3 / turbo / 2.6 / 2.5 / master / 1.x)
- **UseapiPixverseGenerateVideo** — PixVerse v2 video create (v6, Seedance, Kling, Veo 3.1, Sora 2, HappyHorse, Grok)

## [0.9.0] - 2026-08-06

### Added
- **`UseapiPixverseGenerateImage`** — PixVerse v2 images (Seedream 5, Nano Banana 2 Lite/Pro, Kling image, GPT Image 2.0, Qwen, …)
- **`UseapiMinimaxUploadFile`** — upload IMAGE tensors to MiniMax for `fileID` use in H3/Seedance video nodes

## [0.8.0] - 2026-08-06

### Added
- **`UseapiMinimaxGenerate`** node (`Useapi.net/MiniMax`): MiniMax API v1 video generation with **Hailuo-3.0 (MiniMax H3)**, Seedance 2.0 / Fast / Mini, plus legacy 02/2.3/Sora/Veo model IDs. Supports aspect ratio, resolution, duration (4–15s), start/end frame file IDs, and per-node timeout.
- Google Flow image models aligned to July 2026 API: **`nano-banana-2-lite`** (default), `nano-banana-2`, `nano-banana-pro` (deprecated aliases `imagen-4` / `nano-banana` still listed).
- Google Flow aspect ratios: `16:9`, `4:3`, `1:1`, `3:4`, `9:16`, `auto` (+ legacy landscape/portrait).
- Up to **10 reference images** on `UseapiGoogleFlowGenerateImage` (API max).
- **`encodedImage` fallback** when Google omits `fifeUrl` (July 27 2026 CDN rate-limit behavior).
- Runway Images model list includes `nano-banana-2-lite`.

### Fixed
- Veo media responses: accept `videoUrl` / `fifeUrl` / `servingBaseUri`; clearer error when Google omits download links temporarily.
- Default image model no longer points at removed Imagen-4 as primary.

### Docs / config
- `nodes_config.json` defaults updated for nano-banana-2-lite and MiniMax H3.
- Package version bumped to **0.8.0**.

## [0.7.0] - 2026-03-03

### Added
- **Per-node timeout override** (`#50`): Optional `timeout` INT input (default `0` = use global config) on `UseapiVeoGenerate`, `UseapiVeoUpscale`, `UseapiVeoExtend`, `UseapiVeoConcatenate`, `UseapiRunwayGenerate`, `UseapiRunwayVideoToVideo`, and `UseapiRunwayFramesGenerate`. Set to any value >0 to override the global default for that specific node — useful for Veo 4K upscaling or other unusually long operations.
- **`nodes_config.json` schema validation** (`#45`): `_load_config()` now validates global keys (`default_timeout`, `default_aspect_ratio`) for correct types, checks that node-specific keys are dicts, and uses `difflib` to suggest corrections for typos (e.g. `defualt_timeout` → `default_timeout`). Validation is non-fatal — warnings are logged and the plugin continues to load.

### Fixed
- **`UseapiVeoGenerate` model list**: Removed unsupported `veo-3` and `veo-2` from the model dropdown. The `/google-flow/videos` endpoint only supports `veo-3.1-fast`, `veo-3.1-quality`, and `veo-3.1-fast-relaxed`.
- **`UseapiVeoGenerate` silent "All operations failed" with wrong image type**: Passing a raw image URL (e.g. a fife URL from `UseapiGoogleFlowGenerateImage`'s `image_url` output) to `start_image` or `end_image` caused a persistent 400 error. The API requires a `mediaGenerationId`, not a URL. These fields now raise a clear `ValueError` immediately when a URL is detected, with instructions to use `media_generation_id` or `start_image_tensor` instead.
- **Improved "All operations failed" error hint**: Now explicitly mentions Google AI Ultra subscription requirement and multi-account email routing.

### Tests
- Integration tests updated for `UseapiRunwayFramesGenerate` and `UseapiVeoGenerate` timeout parameter.
- Config validation tests added in `tests/test_validation.py`.

## [0.6.0] - 2026-03-03

### Added
- **`UseapiRunwayUploadAudio` node** (`Useapi.net/Runway`): Upload MP3 or WAV audio files for use with Runway audio-to-video workflows. Returns `audio_asset_id`.
- **IMAGE tensor inputs to `UseapiVeoGenerate`**: Optional `start_image` and `end_image` IMAGE tensor inputs with automatic upload via new `_google_flow_upload_image()` helper. Enables image-conditioned video generation.
- **`audio_url` output on `UseapiRunwayGenerate`**: 4th return value `audio_url` (STRING) for workflows that need access to the generated audio URL. Falls back to empty string if not present.
- **Automatic GitHub issue error reporting** (`error_reporter.py`): Unhandled exceptions are auto-reported as GitHub issues on the repository, with full traceback, node context, and git commit hash.

### Tests
- Added integration tests for `UseapiVideoToFrames` covering frame extraction and output tensor shape.

## [0.5.3] - 2026-03-03

### Fixed
- **`UseapiVeoExtend` persistent 400 "All operations failed"**: Added optional `email` field to `UseapiVeoExtend`. The Useapi.net multi-tenant API routes requests to the correct Google account via `email`; without it, extend requests were routed to a different account than the one that generated the source video, causing the error. Pass the same email used in `UseapiVeoGenerate`.
- **Improved error message for "All operations failed" (400)**: `_check_status` now detects this specific response and raises a targeted, context-aware message. Veo Extend calls get guidance about passing `email` and trying a different prompt; all other endpoints get a generic account-routing hint.

## [0.5.1] - 2026-03-03

### Added
- **`UseapiVideoToFrames` node** (`Useapi.net/Utils`): Decodes any UseAPI video output (`video_path`) into a ComfyUI `IMAGE` tensor batch compatible with `VHS_VideoCombine` and native `SaveVideo` nodes. Also shows an in-node video preview. Outputs: `frames` (IMAGE), `frame_count` (INT), `fps` (FLOAT). Requires `opencv-python`.

## [0.5.0] - 2026-03-03

### Added
- **3 New Runway Nodes**:
  - `UseapiRunwayAleph`: Video-to-video transformation using Gen4 Aleph with optional image conditioning.
  - `UseapiRunwayGen3TurboExpand`: Expand (outpaint) Gen3 Turbo videos to landscape or portrait.
  - `UseapiRunwayGen3TurboActOne`: Motion transfer from a driving video to a character using Gen3 Turbo Act One.
- Updated tests to cover the 3 new nodes (structure, contract, and category validation).
- Added `pyproject.toml` — ComfyUI now displays the pack as **UseAPI.net** instead of the folder name.

## [0.2.0] - 2026-02-25

### Added
- **10 New Nodes**:
  - `UseapiVeoVideoToGif`: Convert Veo videos to GIF.
  - `UseapiVeoConcatenate`: Concatenate multiple Veo videos with trim options.
  - `UseapiRunwayImages`: Generate images with Runway (nano-banana, gen4, gen4-turbo).
  - `UseapiRunwayGen4Upscale`: Upscale Runway Gen4 videos.
  - `UseapiRunwayActTwo`: Motion transfer from driving video to character.
  - `UseapiRunwayActTwoVoice`: Add voice to Act Two videos.
  - `UseapiRunwayLipsync`: Create lipsync videos.
  - `UseapiRunwaySuperSlowMotion`: Apply super slow-motion.
  - `UseapiRunwayTranscribe`: Transcribe video/audio assets.
  - `UseapiRunwayGen3TurboExtend`: Extend Gen3 Turbo videos.
- **Documentation**:
  - Updated `README.md` with installation steps for Windows/Linux/Mac.
  - Added full input/output reference for all 24 nodes.
  - Added `examples/` directory with simulated workflow JSON files (`google_flow_workflows.json`, `runway_workflows.json`).

## [0.1.0] - 2026-02-25

### Added
- Added CI workflow via GitHub Actions (`.github/workflows/ci.yml`) to run structure tests.
- Added support for `nodes_config.json` to allow users to customize default parameter values (e.g., model, aspect ratio, timeout).
- Added CI status badge to `README.md`.
