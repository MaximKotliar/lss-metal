import Foundation
import LSFGCore
import Metal

/// Compute pipelines for Lossless Scaling's upscaler shaders, from the library lsfg-metal translates out of the
/// user's own Lossless.dll (`lsfg.metallib` + `groups.txt` in the cache folder). Each translated shader numbers its
/// Metal bindings its own way, so resources are bound by their names (`t0`, `u0`, `cb0`, `s0`), found by reflection.
public final class LSSKernels: @unchecked Sendable {
    public struct Kernel: @unchecked Sendable {
        public let id: UInt32
        public let pipeline: MTLComputePipelineState
        public let threadgroupSize: MTLSize
        /// Binding name (`t0`, `u0`, `cb0`, `s0`, …) -> Metal index.
        public let bindings: [String: Int]

        public func index(_ name: String) -> Int? { bindings[name] }
    }

    public let device: MTLDevice
    public let kernels: [UInt32: Kernel]
    /// Our own kernels (nearest-neighbour scaling): not from the DLL.
    let own: MTLLibrary

    /// Every shader id the upscalers here use.
    public static let ids: Set<UInt32> = Set(Array(141...161).map(UInt32.init)) .union([133, 134, 137, 138, 139, 140, 247])

    /// Loads the kernels from `cacheDirectory` (see `LSFGKernels.cacheDirectory(forDLL:)`).
    public init(cacheDirectory: URL, device: MTLDevice) throws {
        self.device = device
        let library = try device.makeLibrary(URL: cacheDirectory.appending(path: "lsfg.metallib"))
        var groups: [UInt32: MTLSize] = [:]
        let text = try String(contentsOf: cacheDirectory.appending(path: "groups.txt"), encoding: .utf8)
        for line in text.split(separator: "\n") {
            let f = line.split(separator: " ").compactMap { Int($0) }
            if f.count == 4 { groups[UInt32(f[0])] = MTLSize(width: f[1], height: f[2], depth: f[3]) }
        }
        var kernels: [UInt32: Kernel] = [:]
        for id in Self.ids {
            guard let fn = library.makeFunction(name: "lsfg_\(id)"), let size = groups[id] else {
                throw LSSError.missingKernel(id)
            }
            var reflection: MTLComputePipelineReflection?
            let pipeline = try device.makeComputePipelineState(function: fn, options: [.bindingInfo], reflection: &reflection)
            var bindings: [String: Int] = [:]
            for b in reflection?.bindings ?? [] { bindings[b.name] = b.index }
            kernels[id] = Kernel(id: id, pipeline: pipeline, threadgroupSize: size, bindings: bindings)
        }
        self.kernels = kernels
        own = try device.makeLibrary(source: Self.ownSource, options: nil)
    }

    /// Kernels for the user's Lossless.dll: its translated library in the lsfg-metal cache.
    public convenience init(dll: URL, device: MTLDevice) throws {
        try self.init(cacheDirectory: LSFGKernels.cacheDirectory(forDLL: dll), device: device)
    }

    /// Kernels from a pre-baked library (lsfg-metal's `bake --dxbc-compiler … --dxbc-ids …`: no weights inside),
    /// specialized with the weights and other numbers read from the user's own Lossless.dll. Needs no tools.
    /// Throws if a shader in this DLL doesn't have the baked structure (a Lossless Scaling update changed it).
    public init(bakedDirectory: URL, dll: URL, device: MTLDevice) throws {
        self.device = device
        let manifest = try JSONDecoder().decode(BakedManifest.self,
                                                from: Data(contentsOf: bakedDirectory.appending(path: LSFGKernels.dxbcManifestName)))
        let library = try device.makeLibrary(URL: bakedDirectory.appending(path: LSFGKernels.dxbcSkeletonLibraryName))
        let shaders = Dictionary(uniqueKeysWithValues: try LSFGShaderLibrary(dll: dll).entries.map { ($0.resourceID, $0.shader.bytes) })
        var kernels: [UInt32: Kernel] = [:]
        for id in Self.ids {
            guard let entry = manifest.kernels[String(id)], let dxbc = shaders[id], entry.group.count == 3 else {
                throw LSSError.missingKernel(id)
            }
            let values = MTLFunctionConstantValues()
            for (k, c) in try DXBCConstants.constants(dxbc, entry: entry).enumerated() {
                switch c.kind {
                case .half: var bits = UInt16(c.bits); values.setConstantValue(&bits, type: .half, index: k)
                case .float: var bits = c.bits; values.setConstantValue(&bits, type: .float, index: k)
                }
            }
            let fn = try library.makeFunction(name: "lsfg_\(id)", constantValues: values)
            var reflection: MTLComputePipelineReflection?
            let pipeline = try device.makeComputePipelineState(function: fn, options: [.bindingInfo], reflection: &reflection)
            var bindings: [String: Int] = [:]
            for b in reflection?.bindings ?? [] { bindings[b.name] = b.index }
            kernels[id] = Kernel(id: id, pipeline: pipeline,
                                 threadgroupSize: MTLSize(width: entry.group[0], height: entry.group[1], depth: entry.group[2]),
                                 bindings: bindings)
        }
        self.kernels = kernels
        own = try device.makeLibrary(source: Self.ownSource, options: nil)
    }

    public subscript(id: UInt32) -> Kernel { kernels[id]! }

    static let ownSource = """
    #include <metal_stdlib>
    using namespace metal;
    struct NearestArgs { uint2 srcOrigin; uint2 srcSize; uint2 dstOrigin; uint2 dstSize; };
    // nearest neighbour: each output pixel takes the source pixel its centre falls in (integer ratios = integer scaling)
    kernel void lss_nearest(constant NearestArgs& a [[buffer(0)]], texture2d<float> src [[texture(0)]],
                            texture2d<float, access::write> dst [[texture(1)]], uint2 g [[thread_position_in_grid]]) {
        if (any(g >= a.dstSize)) return;
        const uint2 s = min(uint2((float2(g) + 0.5f) * float2(a.srcSize) / float2(a.dstSize)), a.srcSize - 1);
        dst.write(src.read(s + a.srcOrigin), g + a.dstOrigin);
    }
    """
}

public enum LSSError: Error, CustomStringConvertible {
    case missingKernel(UInt32)
    case badSize(String)

    public var description: String {
        switch self {
        case let .missingKernel(id): "Lossless Scaling shader \(id) is missing from the translated library (run lsfg-metal's scripts/translate-shaders.sh with DXBC_COMPILER set)"
        case let .badSize(what): what
        }
    }
}
