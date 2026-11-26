# The `best` branch

This branch is upstream VIctory `main` plus open pull requests that have been measured to help on real hardware but are not merged yet. Each one was security-reviewed, built and tested before it was added. All commits keep their original authors, and every fix also stays open as a pull request on [Marvisak/VIctory](https://github.com/Marvisak/VIctory), which remains the main project.

No game files, shaders, shader names or caches are in this repository. You need your own dump.

## What it adds to `main` `63bcca4`

| Pull request | Author | What it does |
|---|---|---|
| #1924 | slatefield (rebase of #1399) | Merged geometry stages: export programs as `vs_6_6`, geometry programs as `gs_6_6` |
| #2978 | Anzymanz | Whole-wave sections follow binary64 sums; `v_cvt_f32_f64` and 64-bit float sources |
| #385 | nathanbzrr | Track every incoming edge of a lane-mask phi |
| #2999 | wwwwwwww3333333 | Implicit-level samples in compute follow the console's lane quads |
| #1737 | jaschinski97-ops | Permlane EXEC and FI/BC flags (with `v_permlane16_b32` / `v_permlanex16_b32` from #1014) |
| #3016 | Anzymanz | The SCC a waterfall loop's `s_andn2_wrexec` leaves is false after the loop |
| #2363 | Calvix2k | Descriptor dwords read as data: the record count and the format |
| #1738 | jaschinski97-ops | Preserve live interpolation values, with computed barycentrics |
| #3047, #3048 | zed6000 | Rasteriser and depth-stencil states named from the pack's register images |
| #2703 | Calvix2k | A root constant's dword 4 is a value, not a view offset |

## Measured

RX 7900 XTX, driver 32.0.32029.2010, Series X profile, intro cutscene, pipelines cached:

| | `main` | `best` |
|---|---:|---:|
| Shader programs that validate (of the archive's 26,084) | 24,850 | 26,013 |
| Shaders that fail to translate in the intro | 8 | 0 |
| Draws dropped per frame (no host pipeline) | 206 | 0 |
| Frame rate (median) | 1.9 fps | 2.4 fps |

Visible result: characters' faces draw properly. Skin still looks too dark, and some scenes still have missing or blocky parts.

## Build and run

Follow the main [README](README.md) for the toolchain and setup, then:

```powershell
cargo build --release --locked -p victory
.\target\release\victory.exe run "<path to your game executable>" --execute
```

For drop reasons and scripted keys (`VICTORY_KEYS`), use the diagnostics build: `cargo build --profile diag --locked -p victory --features diagnostics`.

The first run with a new build is slow for a few minutes while the GPU driver compiles pipelines. Later runs are faster.
