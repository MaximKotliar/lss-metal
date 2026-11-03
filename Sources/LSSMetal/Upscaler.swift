import Foundation
import Metal

/// Which of Lossless Scaling's upscalers to run.
public enum LSSMode: Sendable, Equatable, Hashable {
    /// LS1, Lossless Scaling's own: a small network that upscales brightness 2× and adds the detail it predicts to a
    /// bilinear enlargement. `sharpness` 1–5 picks one of its five trained weight sets; `performance` uses the
    /// single-pass variant.
    case ls1(sharpness: Int, performance: Bool)
    /// AMD FSR 1: EASU (edge-adaptive upscale) then RCAS sharpening, `sharpness` 0 (off) … 1 (strongest).
    case fsr(sharpness: Float)
    /// Catmull-Rom bicubic.
    case bicubic
    /// Bicubic, then AMD CAS (contrast-adaptive sharpening), `sharpness` 0 … 1: Lossless Scaling's "BCAS".
    case bcas(sharpness: Float)
    /// Nearest neighbour (integer scaling at integer ratios). Not from the DLL.
    case nearest
}

/// Upscales a region of a texture into a region of another, with one of Lossless Scaling's upscalers. Create one per
/// input size, output size and mode; it keeps the intermediate textures.
public final class LSSUpscaler: @unchecked Sendable {
    public let mode: LSSMode
    public let inputSize: MTLSize
    public let outputSize: MTLSize
    /// Input is scRGB-like HDR (LS1 converts it for its network, then back).
    public var hdr = false

    private let kernels: LSSKernels
    private var intermediates: [MTLTexture] = []
    private let nearest: MTLComputePipelineState
    private let linearClamp: MTLSamplerState

    // LS1 weight sets: three passes each (quality) and one (performance), in the DLL's order
    static let ls1Quality: [[UInt32]] = [[147, 148, 149], [150, 151, 152], [153, 154, 155], [156, 157, 158], [159, 160, 161]]
    static let ls1Performance: [UInt32] = [141, 142, 143, 144, 145]
    static let ls1Combine: UInt32 = 146
    static let fsrEASU: UInt32 = UInt32(ProcessInfo.processInfo.environment["LSS_EASU"] ?? "") ?? 138   // the half-precision EASU: same output as 137 within rounding, twice as fast
    static let fsrRCAS: UInt32 = 247
    static let cas: UInt32 = 134
    static let bicubicID: UInt32 = 133

    public init(kernels: LSSKernels, mode: LSSMode, inputSize: MTLSize, outputSize: MTLSize) throws {
        guard inputSize.width > 0, inputSize.height > 0, outputSize.width > 0, outputSize.height > 0 else {
            throw LSSError.badSize("empty input or output")
        }
        self.kernels = kernels
        self.mode = mode
        self.inputSize = inputSize
        self.outputSize = outputSize
        let device = kernels.device
        nearest = try device.makeComputePipelineState(function: kernels.own.makeFunction(name: "lss_nearest")!)
        let sd = MTLSamplerDescriptor()
        sd.minFilter = .linear
        sd.magFilter = .linear
        sd.sAddressMode = .clampToEdge
        sd.tAddressMode = .clampToEdge
        linearClamp = device.makeSamplerState(descriptor: sd)!

        func texture(_ w: Int, _ h: Int, _ format: MTLPixelFormat) -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: w, height: h, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            return device.makeTexture(descriptor: d)!
        }
        switch mode {
        case let .ls1(_, performance):
            let w = inputSize.width, h = inputSize.height
            intermediates = performance ? [] : [texture(w, h, .rgba16Float), texture(w, h, .rgba16Float)]
            intermediates.append(texture(2 * w, 2 * h, .r16Float))   // brightness detail at 2x
        case .fsr, .bcas:
            intermediates = [texture(outputSize.width, outputSize.height, .rgba16Float)]
        case .bicubic, .nearest:
            break
        }
    }

    /// The constant block Lossless Scaling's scaling shaders share (48 bytes): input region, input texture size,
    /// region origin, HDR flag; output size and origin.
    struct Consts {
        var regionW: UInt32, regionH: UInt32, textureW: UInt32, textureH: UInt32
        var originX: UInt32, originY: UInt32, hdr: UInt32, pad: UInt32 = 0
        var outW: UInt32, outH: UInt32, outX: UInt32, outY: UInt32
    }

    /// Encodes the upscale of `input`'s region at `inputOrigin` (size `inputSize`) into `output` at `outputOrigin`
    /// (size `outputSize`). `output` needs shader-write usage.
    public func encode(commandBuffer cb: MTLCommandBuffer, input: MTLTexture, inputOrigin: MTLOrigin = MTLOrigin(),
                       output: MTLTexture, outputOrigin: MTLOrigin = MTLOrigin()) {
        let iw = UInt32(inputSize.width), ih = UInt32(inputSize.height)
        let ow = UInt32(outputSize.width), oh = UInt32(outputSize.height)
        let hdrFlag: UInt32 = hdr ? 1 : 0
        let full = Consts(regionW: iw, regionH: ih, textureW: UInt32(input.width), textureH: UInt32(input.height),
                          originX: UInt32(inputOrigin.x), originY: UInt32(inputOrigin.y), hdr: hdrFlag,
                          outW: ow, outH: oh, outX: UInt32(outputOrigin.x), outY: UInt32(outputOrigin.y))
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "LSS"
        defer { enc.endEncoding() }

        func run(_ id: UInt32, consts: Consts?, cbName: String = "cb0", textures: [String: MTLTexture], grid: MTLSize,
                 extra: ((MTLComputeCommandEncoder, LSSKernels.Kernel) -> Void)? = nil) {
            let k = kernels[id]
            enc.setComputePipelineState(k.pipeline)
            if var c = consts, let i = k.index(cbName) { enc.setBytes(&c, length: MemoryLayout<Consts>.stride, index: i) }
            for (name, t) in textures { if let i = k.index(name) { enc.setTexture(t, index: i) } }
            if let i = k.index("s0") { enc.setSamplerState(linearClamp, index: i) }
            extra?(enc, k)
            enc.dispatchThreads(grid, threadsPerThreadgroup: k.threadgroupSize)
        }

        switch mode {
        case let .ls1(sharpness, performance):
            let set = max(1, min(5, sharpness)) - 1
            let detail = intermediates.last!
            // the network leaves a 2-pixel border unwritten: there the detail must be zero (plain bilinear)
            clear(detail, enc: enc)
            let inGrid = MTLSize(width: Int(iw), height: Int(ih), depth: 1)
            if performance {
                run(Self.ls1Performance[set], consts: full, textures: ["t0": input, "u0": detail], grid: inGrid)
            } else {
                let ids = Self.ls1Quality[set]
                let (f1, f2) = (intermediates[0], intermediates[1])
                var local = full
                local.originX = 0; local.originY = 0; local.textureW = iw; local.textureH = ih
                run(ids[0], consts: full, textures: ["t0": input, "u0": f1], grid: inGrid)
                run(ids[1], consts: nil, textures: ["t0": f1, "u0": f2], grid: inGrid)
                run(ids[2], consts: local, textures: ["t0": f2, "u0": detail], grid: inGrid)
            }
            // bilinear color + the network's brightness detail, at the output size
            run(Self.ls1Combine, consts: full, textures: ["t0": detail, "t1": input, "u0": output],
                grid: MTLSize(width: Int(ow), height: Int(oh), depth: 1))

        case let .fsr(sharpness):
            let mid = intermediates[0]
            var easu = Self.easuConstants(input: (Float(inputOrigin.x), Float(inputOrigin.y), Float(iw), Float(ih)),
                                          texture: (Float(input.width), Float(input.height)), output: (Float(ow), Float(oh)))
            let groups = MTLSize(width: (Int(ow) + 15) / 16, height: (Int(oh) + 15) / 16, depth: 1)
            var k = kernels[Self.fsrEASU]
            enc.setComputePipelineState(k.pipeline)
            if let i = k.index("cb0") { enc.setBytes(&easu, length: easu.count * 4, index: i) }
            if let i = k.index("t0") { enc.setTexture(input, index: i) }
            if let i = k.index("u0") { enc.setTexture(mid, index: i) }
            if let i = k.index("s0") { enc.setSamplerState(linearClamp, index: i) }
            enc.dispatchThreadgroups(groups, threadsPerThreadgroup: k.threadgroupSize)
            // RCAS: Const0.x = sharpness as FSR's FsrRcasCon gives it, exp2(-stops), 0 stops = sharpest, 2 = soft
            let stops = (1 - max(0, min(1, sharpness))) * 2
            var rcas = [Float](repeating: 0, count: 20)
            rcas[0] = exp2(-stops)
            var origin: [UInt32] = [UInt32(outputOrigin.x), UInt32(outputOrigin.y), 0, 0]
            k = kernels[Self.fsrRCAS]
            enc.setComputePipelineState(k.pipeline)
            if let i = k.index("cb0") { enc.setBytes(&rcas, length: rcas.count * 4, index: i) }
            if let i = k.index("cb1") { enc.setBytes(&origin, length: origin.count * 4, index: i) }
            if let i = k.index("t0") { enc.setTexture(mid, index: i) }
            if let i = k.index("u0") { enc.setTexture(output, index: i) }
            enc.dispatchThreadgroups(groups, threadsPerThreadgroup: k.threadgroupSize)

        case .bicubic:
            run(Self.bicubicID, consts: full, cbName: "cb1", textures: ["t0": input, "u0": output],
                grid: MTLSize(width: Int(ow), height: Int(oh), depth: 1))

        case let .bcas(sharpness):
            let mid = intermediates[0]
            run(Self.bicubicID, consts: full, cbName: "cb1", textures: ["t0": input, "u0": mid],
                grid: MTLSize(width: Int(ow), height: Int(oh), depth: 1))
            // CAS: const1.x = the negative peak, -1 / lerp(8, 5, sharpness) (AMD's CasSetup)
            let s = max(0, min(1, sharpness))
            var cas: [Float] = [0, 0, 0, 0, -1 / (8 + (5 - 8) * s), 0, 0, 0]
            var out = Consts(regionW: ow, regionH: oh, textureW: ow, textureH: oh, originX: 0, originY: 0, hdr: hdrFlag,
                             outW: ow, outH: oh, outX: UInt32(outputOrigin.x), outY: UInt32(outputOrigin.y))
            let k = kernels[Self.cas]
            enc.setComputePipelineState(k.pipeline)
            if let i = k.index("cb0") { enc.setBytes(&cas, length: cas.count * 4, index: i) }
            if let i = k.index("cb1") { enc.setBytes(&out, length: MemoryLayout<Consts>.stride, index: i) }
            if let i = k.index("t0") { enc.setTexture(mid, index: i) }
            if let i = k.index("u0") { enc.setTexture(output, index: i) }
            enc.dispatchThreadgroups(MTLSize(width: (Int(ow) + 15) / 16, height: (Int(oh) + 15) / 16, depth: 1),
                                     threadsPerThreadgroup: k.threadgroupSize)

        case .nearest:
            var a: [UInt32] = [UInt32(inputOrigin.x), UInt32(inputOrigin.y), iw, ih,
                               UInt32(outputOrigin.x), UInt32(outputOrigin.y), ow, oh]
            enc.setComputePipelineState(nearest)
            enc.setBytes(&a, length: a.count * 4, index: 0)
            enc.setTexture(input, index: 0)
            enc.setTexture(output, index: 1)
            enc.dispatchThreads(MTLSize(width: Int(ow), height: Int(oh), depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        }
    }

    /// FSR 1 EASU constants (AMD's FsrEasuConOffset), as floats: con0..con3, then the unused Sample block.
    static func easuConstants(input: (x: Float, y: Float, w: Float, h: Float), texture: (w: Float, h: Float),
                              output: (w: Float, h: Float)) -> [Float] {
        let (ix, iy, iw, ih) = input
        let (tw, th) = texture
        let (ow, oh) = output
        return [
            iw / ow, ih / oh, 0.5 * iw / ow - 0.5 + ix, 0.5 * ih / oh - 0.5 + iy,
            1 / tw, 1 / th, 1 / tw, -1 / th,
            -1 / tw, 2 / th, 1 / tw, 2 / th,
            0, 4 / th, 0, 0,
            0, 0, 0, 0,
        ]
    }

    private func clear(_ t: MTLTexture, enc: MTLComputeCommandEncoder) {
        // a 1x1 zero source scaled to the whole texture, through our nearest kernel
        if zero == nil {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: 1, height: 1, mipmapped: false)
            d.usage = .shaderRead
            zero = kernels.device.makeTexture(descriptor: d)
            var z: UInt16 = 0
            zero?.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &z, bytesPerRow: 2)
        }
        var a: [UInt32] = [0, 0, 1, 1, 0, 0, UInt32(t.width), UInt32(t.height)]
        enc.setComputePipelineState(nearest)
        enc.setBytes(&a, length: a.count * 4, index: 0)
        enc.setTexture(zero, index: 0)
        enc.setTexture(t, index: 1)
        enc.dispatchThreads(MTLSize(width: t.width, height: t.height, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    }
    private var zero: MTLTexture?
}
