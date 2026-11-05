import Foundation
import LSFGCore
import LSSMetal
import Metal

let usage = """
usage: lss-metal <command>
  upscale IN.png -o OUT.png [--mode ls1|ls1-perf|fsr|bcas|bicubic|nearest|nis] [--sharpness N] [--scale F | --size WxH] [--runs N]
  eval IN.png [--scale F] [--out DIR]   shrink IN by F (default 2), upscale it back with every mode, score against IN
  check-baked DIR IN.png   every mode with the baked kernels in DIR vs the translated ones: identical output?
  nis-check DIR IN.png [--scale F]   our NIS vs NVIDIA's compiled NIS translated to Metal (DIR/243.metal: SDR,
                         244.metal: linear HDR, from Lossless.dll via dxbc-spirv and spirv-cross): how different?
  nis-bench IN.png [--scale F]   NIS's compile-time variants (block size, half or float maths): time, and PSNR vs float

nis and nearest need no Lossless.dll.

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
// Lossless Scaling's kernels, when there's a Lossless.dll (nis and nearest work without)
let kernels: LSSKernels? = dll.flatMap { try? LSSKernels(dll: $0, device: device) }

func parseMode(_ s: String, sharpness: Double) -> LSSMode {
    switch s {
    case "ls1": .ls1(sharpness: Int(sharpness), performance: false)
    case "ls1-perf": .ls1(sharpness: Int(sharpness), performance: true)
    case "fsr": .fsr(sharpness: Float(sharpness))
    case "bicubic": .bicubic
    case "bcas": .bcas(sharpness: Float(sharpness))
    case "nearest": .nearest
    case "nis": .nis(sharpness: Float(sharpness))
    default: fail("unknown mode \(s)")
    }
}

func texture(_ w: Int, _ h: Int, write: Bool) -> MTLTexture {
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
    d.usage = write ? [.shaderRead, .shaderWrite] : .shaderRead
    return device.makeTexture(descriptor: d)!
}

/// Upscales `img` to w x h; returns the result and the median GPU time of `runs` runs.
func upscale(_ img: FrameIO.Image, mode: LSSMode, w: Int, h: Int, runs: Int = 1, with kernels: LSSKernels? = kernels) throws -> (FrameIO.Image, Double) {
    let src = texture(img.width, img.height, write: false)
    FrameIO.upload(img, to: src)
    let dst = texture(w, h, write: true)
    let up = try LSSUpscaler(kernels: kernels, device: device, mode: mode, inputSize: MTLSize(width: img.width, height: img.height, depth: 1),
                             outputSize: MTLSize(width: w, height: h, depth: 1))
    // runs > 1: steady clocks (a few command buffers to warm up, then `runs` buffers of 10 upscales each)
    var times: [Double] = []
    let batch = runs > 1 ? 10 : 1, warm = runs > 1 ? 5 : 0
    for i in 0..<(max(1, runs) + warm) {
        let cb = queue.makeCommandBuffer()!
        for _ in 0..<batch { up.encode(commandBuffer: cb, input: src, output: dst) }
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw e }
        if i >= warm { times.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3 / Double(batch)) }
    }
    return (FrameIO.download(dst), times.sorted()[times.count / 2])
}

/// fp16 bits to Float (Float16 doesn't exist on x86_64).
func halfBitsToFloat(_ h: UInt16) -> Float {
    let sign: Float = h & 0x8000 != 0 ? -1 : 1
    let e = Int((h >> 10) & 0x1F), m = Float(h & 0x3FF)
    if e == 0 { return sign * m * pow(2, -24) }
    if e == 31 { return m == 0 ? sign * .infinity : .nan }
    return sign * (1 + m / 1024) * pow(2, Float(e - 15))
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
        guard let dll else { fail("Lossless.dll not found") }
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
        var modes: [(String, LSSMode)] = [("nearest", .nearest)]
        if kernels != nil {
            modes += [("bicubic", .bicubic), ("bcas", .bcas(sharpness: 0.5)), ("fsr-soft", .fsr(sharpness: 0.2)), ("fsr", .fsr(sharpness: 0.8))]
            for s in 1...5 { modes.append(("ls1-\(s)", .ls1(sharpness: s, performance: false))) }
            for s in 1...5 { modes.append(("ls1-perf-\(s)", .ls1(sharpness: s, performance: true))) }
        }
        modes += [("nis-0", .nis(sharpness: 0)), ("nis-0.25", .nis(sharpness: 0.25)), ("nis-0.5", .nis(sharpness: 0.5)), ("nis-1", .nis(sharpness: 1))]
        print("\(truth.width)x\(truth.height), shrunk \(f)x to \(small.width)x\(small.height) and upscaled back:")
        for (name, mode) in modes {
            let (res, ms) = try upscale(small, mode: mode, w: truth.width, h: truth.height, runs: 15)
            print(String(format: "  %-12@ PSNR %6.2f dB   GPU %.3f ms", name as NSString, psnr(res, truth), ms))
            if let outDir { try FrameIO.save(res, to: outDir.appending(path: "\(name).png")) }
        }
    case "nis-check":
        guard args.count >= 2 else { fail(usage) }
        let dir = URL(fileURLWithPath: args[0])
        let truth = try FrameIO.load(URL(fileURLWithPath: args[1]))
        let f = Double(option("--scale") ?? "") ?? 2
        let small = FrameIO.resized(truth, width: Int(Double(truth.width) / f), height: Int(Double(truth.height) / f))
        let (iw, ih, ow, oh) = (small.width, small.height, truth.width, truth.height)
        // coefficient textures as NVIDIA makes them: RGBA32F, 2 x 64 (8 taps per phase)
        func coefTexture(_ t: [[Float]]) -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 2, height: 64, mipmapped: false)
            d.usage = .shaderRead
            let tex = device.makeTexture(descriptor: d)!
            var v: [Float] = []
            for row in t { v += row + [0, 0] }
            tex.replace(region: MTLRegionMake2D(0, 0, 2, 64), mipmapLevel: 0, withBytes: v, bytesPerRow: 2 * 16)
            return tex
        }
        let coefS = coefTexture(NISUpscaler.coefScale), coefU = coefTexture(NISUpscaler.coefUSM)
        let sd = MTLSamplerDescriptor(); sd.minFilter = .linear; sd.magFilter = .linear
        sd.sAddressMode = .clampToEdge; sd.tAddressMode = .clampToEdge
        let sampler = device.makeSamplerState(descriptor: sd)!
        // in a float format, so the comparison sees more than 8 bits
        func tex(_ w: Int, _ h: Int) -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: w, height: h, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            return device.makeTexture(descriptor: d)!
        }
        let src8 = texture(iw, ih, write: false)
        FrameIO.upload(small, to: src8)
        let src = tex(iw, ih)
        do { let cb = queue.makeCommandBuffer()!; let b = cb.makeBlitCommandEncoder()!
             // rgba8 -> rgba16F through a shader-less path isn't a blit: draw it with nearest at 1:1
             b.endEncoding()
             let up = try LSSUpscaler(kernels: nil, device: device, mode: .nearest, inputSize: MTLSize(width: iw, height: ih, depth: 1), outputSize: MTLSize(width: iw, height: ih, depth: 1))
             up.encode(commandBuffer: cb, input: src8, output: src); cb.commit(); cb.waitUntilCompleted() }
        func pixels(_ t: MTLTexture) -> [Float] {
            var h = [UInt16](repeating: 0, count: t.width * t.height * 4)
            let cb = queue.makeCommandBuffer()!
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: t.pixelFormat, width: t.width, height: t.height, mipmapped: false)
            d.storageMode = .shared; d.usage = .shaderRead
            let copy = device.makeTexture(descriptor: d)!
            let b = cb.makeBlitCommandEncoder()!; b.copy(from: t, to: copy); b.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            copy.getBytes(&h, bytesPerRow: t.width * 8, from: MTLRegionMake2D(0, 0, t.width, t.height), mipmapLevel: 0)
            return h.map(halfBitsToFloat)
        }
        // NVIDIA's bounds check converts the first column's source x (-0.25) to an unsigned integer: 0 in Direct3D,
        // undefined after translation to Metal, so the reference is unreliable within a few pixels of the border.
        // The interior is what's compared.
        func compare(_ a: [Float], _ b: [Float]) -> String {
            var worst: Float = 0, se: Double = 0, n = 0, over = 0
            for i in 0..<a.count where i % 4 != 3 {
                let x = (i / 4) % ow, y = (i / 4) / ow
                if x < 4 || y < 4 || x >= ow - 4 || y >= oh - 4 { continue }
                let d = abs(a[i] - b[i]); worst = max(worst, d); se += Double(d * d); n += 1
                if d > 1.0 / 255 { over += 1 }
            }
            let p = se == 0 ? 99 : 10 * log10(1 / (se / Double(n)))
            return String(format: "interior PSNR %6.2f dB, largest difference %.4f (%.1f/255), %.3f%% of values off by more than 1/255", p, worst, worst * 255, 100 * Double(over) / Double(n))
        }
        for (id, hdr) in [(243, NISUpscaler.HDR.none), (244, .linear)] {
            let file = dir.appending(path: "\(id).metal")
            guard var code = try? String(contentsOf: file, encoding: .utf8) else { print("  \(id): no \(file.path)"); continue }
            // dxbc-spirv gives the group size as LocalSizeId, which spirv-cross leaves undefined
            code = code.replacingOccurrences(of: "kernel void main0", with: "constant uint3 gl_WorkGroupSize = uint3(256u, 1u, 1u);\nkernel void main0")
            let lib = try device.makeLibrary(source: code, options: nil)
            var refl: MTLComputePipelineReflection?
            let ref = try device.makeComputePipelineState(function: lib.makeFunction(name: "main0")!, options: [.bindingInfo], reflection: &refl)
            var idx: [String: Int] = [:]
            for b in refl?.bindings ?? [] { idx[b.name] = b.index }
            let refOut = tex(ow, oh)
            var cfg = NISUpscaler.configBytes(sharpness: 0.5, hdr: hdr, input: (iw, ih), output: (ow, oh))
            var cb = queue.makeCommandBuffer()!
            var enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(ref)
            enc.setBytes(&cfg, length: cfg.count, index: idx["cb0"] ?? 0)
            enc.setTexture(src, index: idx["t0"]!); enc.setTexture(coefS, index: idx["t1"]!); enc.setTexture(coefU, index: idx["t2"]!)
            enc.setTexture(refOut, index: idx["u0"]!); enc.setSamplerState(sampler, index: idx["s0"]!)
            enc.dispatchThreadgroups(MTLSize(width: (ow + 31) / 32, height: (oh + 23) / 24, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            if let e = cb.error { throw e }
            let refPixels = pixels(refOut)
            print("\(id) (\(hdr)), \(iw)x\(ih) -> \(ow)x\(oh), sharpness 0.5, against NVIDIA's compiled shader:")
            for (storage, half) in [(false, false), (true, false), (true, true)] {
                var o = NISUpscaler.Options(); o.halfMath = half; o.halfStorage = storage
                let ours = tex(ow, oh)
                let up = try NISUpscaler(device: device, inputSize: MTLSize(width: iw, height: ih, depth: 1),
                                         outputSize: MTLSize(width: ow, height: oh, depth: 1), sharpness: 0.5, hdr: hdr, options: o)
                cb = queue.makeCommandBuffer()!
                enc = cb.makeComputeCommandEncoder()!
                up.encode(encoder: enc, input: src, output: ours)
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                if let e = cb.error { throw e }
                let mine = pixels(ours)
                print("  ours, \(storage ? "half " : "float") storage, \(half ? "half " : "float") maths: " + compare(mine, refPixels))
                if ProcessInfo.processInfo.environment["NIS_WORST"] != nil, !storage {
                    // where the differences are: block-relative position histogram and worst pixels
                    var worst: [(Float, Int, Int)] = []
                    var bx = [Int](repeating: 0, count: 32), by = [Int](repeating: 0, count: 24), edgeCount = 0
                    for y in 0..<oh { for x in 0..<ow {
                        var d: Float = 0
                        for ch in 0..<3 { d = max(d, abs(mine[(y * ow + x) * 4 + ch] - refPixels[(y * ow + x) * 4 + ch])) }
                        if d > 2.0 / 255 { worst.append((d, x, y)); bx[x % 32] += 1; by[y % 24] += 1
                            if x < 4 || y < 4 || x >= ow - 4 || y >= oh - 4 { edgeCount += 1 } }
                    } }
                    worst.sort { $0.0 > $1.0 }
                    print("    > 2/255: \(worst.count) pixels, \(edgeCount) within 4 of the image border")
                    print("    by x in block: \(bx)")
                    print("    by y in block: \(by)")
                    print("    worst: " + worst.prefix(8).map { String(format: "(%d,%d) %.3f", $0.1, $0.2, $0.0) }.joined(separator: " "))
                }
            }
        }
    case "nis-bench":
        guard args.count >= 1 else { fail(usage) }
        let truth = try FrameIO.load(URL(fileURLWithPath: args[0]))
        let f = Double(option("--scale") ?? "") ?? 2
        let small = FrameIO.resized(truth, width: Int(Double(truth.width) / f), height: Int(Double(truth.height) / f))
        let src = texture(small.width, small.height, write: false)
        FrameIO.upload(small, to: src)
        let dst = texture(truth.width, truth.height, write: true)
        func run(_ o: NISUpscaler.Options) throws -> (FrameIO.Image, Double) {
            let up = try NISUpscaler(device: device, inputSize: MTLSize(width: small.width, height: small.height, depth: 1),
                                     outputSize: MTLSize(width: truth.width, height: truth.height, depth: 1), sharpness: 0.5, options: o)
            // steady clocks: warm up, then 20 upscales per command buffer; median per upscale over 15 buffers
            var times: [Double] = []
            for i in 0..<25 {
                let cb = queue.makeCommandBuffer()!
                for _ in 0..<20 { up.encode(commandBuffer: cb, input: src, output: dst) }
                cb.commit(); cb.waitUntilCompleted()
                if let e = cb.error { throw e }
                if i >= 10 { times.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3 / 20) }
            }
            return (FrameIO.download(dst), times.sorted()[times.count / 2])
        }
        var ref = NISUpscaler.Options(); ref.halfMath = false; ref.halfStorage = false
        let (reference, _) = try run(ref)
        print("\(small.width)x\(small.height) -> \(truth.width)x\(truth.height), sharpness 0.5; PSNR vs all-float 32x24 (NVIDIA's default build):")
        for (bw, bh, t) in [(32, 24, 256), (32, 32, 256), (32, 16, 256), (32, 24, 128), (32, 32, 128), (64, 16, 256), (16, 16, 256), (32, 48, 512)] {
            for (storage, half) in [(false, false), (true, false), (true, true)] {
                var o = NISUpscaler.Options(); o.blockWidth = bw; o.blockHeight = bh; o.threads = t; o.halfMath = half; o.halfStorage = storage
                guard bw * bh % t == 0, t % bw == 0 else { continue }
                guard let (img, ms) = try? run(o) else { print("  block \(bw)x\(bh) \(t) threads: not supported"); continue }
                print(String(format: "  block %2dx%-2d %3d threads, %@ storage, %@ maths   GPU %.3f ms   PSNR %6.2f dB", bw, bh, t,
                             (storage ? "half " : "float") as NSString, (half ? "half " : "float") as NSString, ms, psnr(img, reference)))
            }
        }
    default:
        fail(usage)
    }
} catch {
    fail("\(error)")
}
