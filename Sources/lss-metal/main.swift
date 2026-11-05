import Foundation
import LSFGCore
import LSSMetal
import Metal

let usage = """
usage: lss-metal <command>
  upscale IN.png -o OUT.png [--mode ls1|ls1-perf|fsr|bcas|bicubic|nearest] [--sharpness N] [--scale F | --size WxH] [--runs N]
  eval IN.png [--scale F] [--out DIR]   shrink IN by F (default 2), upscale it back with every mode, score against IN
  check-baked DIR IN.png   every mode with the baked kernels in DIR vs the translated ones: identical output?

The shaders come from your own Lossless.dll (Steam install inside a CrossOver bottle, or LSFG_DLL), translated by
lsfg-metal's scripts/translate-shaders.sh with DXBC_COMPILER set.
"""

func fail(_ s: String) -> Never { FileHandle.standardError.write(Data((s + "\n").utf8)); exit(1) }

var args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else { fail(usage) }
args.removeFirst()
func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { fail("no Metal device") }
let dll = ProcessInfo.processInfo.environment["LSFG_DLL"].map { URL(fileURLWithPath: $0) }
    ?? LosslessLocator.findAll().first?.dll
guard let dll else { fail("Lossless.dll not found") }
let kernels: LSSKernels
do { kernels = try LSSKernels(dll: dll, device: device) } catch { fail("\(error)") }

func parseMode(_ s: String, sharpness: Double) -> LSSMode {
    switch s {
    case "ls1": .ls1(sharpness: Int(sharpness), performance: false)
    case "ls1-perf": .ls1(sharpness: Int(sharpness), performance: true)
    case "fsr": .fsr(sharpness: Float(sharpness))
    case "bicubic": .bicubic
    case "bcas": .bcas(sharpness: Float(sharpness))
    case "nearest": .nearest
    default: fail("unknown mode \(s)")
    }
}

func texture(_ w: Int, _ h: Int, write: Bool) -> MTLTexture {
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
    d.usage = write ? [.shaderRead, .shaderWrite] : .shaderRead
    return device.makeTexture(descriptor: d)!
}

/// Upscales `img` to w x h; returns the result and the median GPU time of `runs` runs.
func upscale(_ img: FrameIO.Image, mode: LSSMode, w: Int, h: Int, runs: Int = 1, with kernels: LSSKernels = kernels) throws -> (FrameIO.Image, Double) {
    let src = texture(img.width, img.height, write: false)
    FrameIO.upload(img, to: src)
    let dst = texture(w, h, write: true)
    let up = try LSSUpscaler(kernels: kernels, mode: mode, inputSize: MTLSize(width: img.width, height: img.height, depth: 1),
                             outputSize: MTLSize(width: w, height: h, depth: 1))
    var times: [Double] = []
    for _ in 0..<max(1, runs) {
        let cb = queue.makeCommandBuffer()!
        up.encode(commandBuffer: cb, input: src, output: dst)
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw e }
        times.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3)
    }
    return (FrameIO.download(dst), times.sorted()[times.count / 2])
}

func psnr(_ a: FrameIO.Image, _ b: FrameIO.Image) -> Double {
    var se = 0.0, n = 0.0
    for i in 0..<a.rgba.count where i % 4 != 3 {
        let d = Double(a.rgba[i]) - Double(b.rgba[i])
        se += d * d; n += 1
    }
    return se == 0 ? 99 : 10 * log10(255 * 255 / (se / n))
}

do {
    switch command {
    case "upscale":
        guard args.count >= 1, let out = option("-o") else { fail(usage) }
        let img = try FrameIO.load(URL(fileURLWithPath: args[0]))
        let sharp = Double(option("--sharpness") ?? "") ?? 3
        let mode = parseMode(option("--mode") ?? "ls1", sharpness: sharp)
        var w = img.width * 2, h = img.height * 2
        if let s = option("--scale"), let f = Double(s) { w = Int((Double(img.width) * f).rounded()); h = Int((Double(img.height) * f).rounded()) }
        if let s = option("--size") { let p = s.split(separator: "x").compactMap { Int($0) }; if p.count == 2 { w = p[0]; h = p[1] } }
        let (res, ms) = try upscale(img, mode: mode, w: w, h: h, runs: Int(option("--runs") ?? "") ?? 1)
        try FrameIO.save(res, to: URL(fileURLWithPath: out))
        print("\(mode) \(img.width)x\(img.height) -> \(w)x\(h): GPU \(String(format: "%.3f", ms)) ms, wrote \(out)")
    case "check-baked":
        guard args.count >= 2 else { fail(usage) }
        let baked = try LSSKernels(bakedDirectory: URL(fileURLWithPath: args[0]), dll: dll, device: device)
        let img = try FrameIO.load(URL(fileURLWithPath: args[1]))
        var modes: [(String, LSSMode)] = [("nearest", .nearest), ("bicubic", .bicubic), ("bcas", .bcas(sharpness: 0.5)),
                                          ("fsr", .fsr(sharpness: 0.8))]
        for s in 1...5 { modes.append(("ls1-\(s)", .ls1(sharpness: s, performance: false))) }
        for s in 1...5 { modes.append(("ls1-perf-\(s)", .ls1(sharpness: s, performance: true))) }
        var same = true
        for (name, mode) in modes {
            let (a, ta) = try upscale(img, mode: mode, w: img.width * 2, h: img.height * 2, runs: 30)
            let (b, tb) = try upscale(img, mode: mode, w: img.width * 2, h: img.height * 2, runs: 30, with: baked)
            let diff = zip(a.rgba, b.rgba).count { $0 != $1 }
            let worst = zip(a.rgba, b.rgba).map { abs(Int($0) - Int($1)) }.max() ?? 0
            same = same && diff == 0
            print(String(format: "  %-12@ differing bytes %d (at most %d/255)   translated %.3f ms   baked %.3f ms", name as NSString, diff, worst, ta, tb))
        }
        print(same ? "PASS: baked kernels give identical output" : "FAIL")
        if !same { exit(1) }
    case "eval":
        guard args.count >= 1 else { fail(usage) }
        let truth = try FrameIO.load(URL(fileURLWithPath: args[0]))
        let f = Double(option("--scale") ?? "") ?? 2
        let small = FrameIO.resized(truth, width: Int(Double(truth.width) / f), height: Int(Double(truth.height) / f))
        let outDir = option("--out").map { URL(fileURLWithPath: $0) }
        if let outDir { try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true) }
        var modes: [(String, LSSMode)] = [("nearest", .nearest), ("bicubic", .bicubic), ("bcas", .bcas(sharpness: 0.5)),
                                          ("fsr-soft", .fsr(sharpness: 0.2)), ("fsr", .fsr(sharpness: 0.8))]
        for s in 1...5 { modes.append(("ls1-\(s)", .ls1(sharpness: s, performance: false))) }
        for s in 1...5 { modes.append(("ls1-perf-\(s)", .ls1(sharpness: s, performance: true))) }
        print("\(truth.width)x\(truth.height), shrunk \(f)x to \(small.width)x\(small.height) and upscaled back:")
        for (name, mode) in modes {
            let (res, ms) = try upscale(small, mode: mode, w: truth.width, h: truth.height, runs: 5)
            print(String(format: "  %-12@ PSNR %6.2f dB   GPU %.3f ms", name as NSString, psnr(res, truth), ms))
            if let outDir { try FrameIO.save(res, to: outDir.appending(path: "\(name).png")) }
        }
    default:
        fail(usage)
    }
} catch {
    fail("\(error)")
}
