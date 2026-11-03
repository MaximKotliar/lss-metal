# LSSMetal

Lossless Scaling's **spatial upscalers** on Metal, for developers: LS1 (Lossless Scaling's own trained upscaler),
AMD FSR 1, BCAS (bicubic + AMD CAS), bicubic, and nearest neighbour, as a Swift library, a C API and a command-line
tool. They upscale a finished frame on its own (no depth, motion or history), so they work for any game or video.

Only for people who own [Lossless Scaling on Steam](https://store.steampowered.com/app/993090/). The shaders, including
LS1's trained weights, are read from the user's own `Lossless.dll` and translated on their Mac. This repository
contains no DLL, no shaders and nothing translated from them. Nearest neighbour is our own small kernel.

Part of a set: [lsfg-metal](https://github.com/MaximKotliar/lsfg-metal) (Lossless Scaling's frame generation; this
package uses its DLL locator and shader translation), [nss-metal](https://github.com/MaximKotliar/nss-metal) (Arm's
neural temporal upscaler) and the app that uses all three, [Lightscale](https://github.com/MaximKotliar/lightscale).

## The upscalers

GPU time, 1280×720 → 2560×1440, M4 Max (warm GPU, median of 60 runs; it varies with GPU clocks):

| Mode | What it is | GPU time |
|---|---|---|
| `ls1` | Lossless Scaling's LS1: a small network upscales brightness 2×, its detail is added to a bilinear enlargement; sharpness picks one of five trained weight sets (softest to sharpest) | 0.5–0.9 ms |
| `ls1` performance | LS1's single-pass variant | 0.44 ms |
| `fsr` | AMD FSR 1: EASU (edge-adaptive upscale), then RCAS sharpening | 0.21 ms |
| `bcas` | Catmull-Rom bicubic, then AMD CAS sharpening | 0.2 ms |
| `bicubic` | Catmull-Rom bicubic | 0.1–0.3 ms |
| `nearest` | nearest neighbour; integer scaling at whole ratios | 0.08 ms |

LS1's output is any size: the network works at 2×, and its detail and the colour are resampled to the output.

## Setup

LSSMetal uses lsfg-metal's shader translation, which needs `spirv-cross`, `spirv-tools` and, for these shaders (they
are DXBC, unlike LSFG 3.1's SPIR-V), `dxbc_compiler` from [dxbc-spirv](https://github.com/doitsujin/dxbc-spirv):

```bash
brew install spirv-cross spirv-tools meson ninja
git clone --recurse-submodules https://github.com/doitsujin/dxbc-spirv && cd dxbc-spirv
meson setup build -Denable_tools=true --buildtype=release && ninja -C build
```

Then, in a checkout of lsfg-metal, with Lossless Scaling installed through Steam in a CrossOver bottle:

```bash
DXBC_COMPILER=/path/to/dxbc-spirv/build/tools/dxbc_compiler scripts/translate-shaders.sh
```

This writes `~/Library/Caches/lsfg-metal/<dll hash>/lsfg.metallib` with every shader in the DLL (`LSFG_CACHE`
overrides the folder). LSSMetal loads the ones it needs from there.

## Use it

```swift
.package(url: "https://github.com/MaximKotliar/lss-metal", branch: "main")
// target dependency: .product(name: "LSSMetal", package: "LSSMetal")
```

```swift
import LSSMetal

let kernels = try LSSKernels(dll: dllURL, device: device)          // the user's Lossless.dll
let upscaler = try LSSUpscaler(kernels: kernels, mode: .ls1(sharpness: 3, performance: false),
                               inputSize: MTLSize(width: 1280, height: 720, depth: 1),
                               outputSize: MTLSize(width: 2560, height: 1440, depth: 1))
upscaler.encode(commandBuffer: commandBuffer, input: frame, output: target)   // target: shader-writable
```

`inputOrigin` / `outputOrigin` select a region of each texture. For hosts that aren't Swift (for example a game
hook under Rosetta), `LSSBridge` is the same as a C API (`lss_bridge_create`, `_encode`, `_destroy`, documented in
`Sources/LSSBridge/Bridge.swift`); it builds for x86_64 too and copies into textures that aren't shader-writable
(drawables).

## Try it

```bash
swift run -c release lss-metal upscale frame.png -o big.png --mode ls1 --sharpness 3 --scale 2
swift run -c release lss-metal eval frame.png --out results/   # shrink 2x, upscale back with every mode, score
```

## How the shaders were mapped

The DLL holds 202 DXBC compute shaders besides LSFG 3.1's SPIR-V. Their resource and constant names identify the
families: LS1 (IDs 141–161: five three-pass weight sets, 147–161, five single-pass ones, 141–145, and the colour
combine, 146), FSR 1 (EASU 137–140, RCAS 247–248), CAS (134), bicubic (133, 135), NVIDIA NIS (243–246), Anime4K-style
CNNs (101–132) and older LSFG versions. The shared 48-byte constant block is (input region size, input texture size),
(input origin, HDR flag), (output size, output origin). NIS and the Anime4K-style modes aren't wired up yet.

## License

MIT (see [LICENSE](LICENSE)). Lossless Scaling, its DLL and its shaders are not covered: they're the user's own
licensed copy, read at run time.
