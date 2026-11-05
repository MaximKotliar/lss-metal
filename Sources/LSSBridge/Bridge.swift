import Foundation
import LSFGCore
import LSSMetal
import Metal

// C API for hosts that aren't Swift, such as a game hook (Objective-C, x86_64 under Rosetta). Built as
// liblssbridge.dylib and dlopen()ed. Metal objects cross the boundary as unretained opaque pointers; the caller keeps
// them alive.
//
//   void *lss_bridge_create(const void *device, int inW, int inH, int outW, int outH, unsigned outFormat,
//                           int mode, float sharpness, int performance, char *error, int errorLength);
//   int   lss_bridge_encode(void *ctx, const void *commandBuffer, const void *input, const void *output,
//                           int outX, int outY);
//   void  lss_bridge_destroy(void *ctx);
//
// mode: 1 LS1, 2 FSR, 3 BCAS, 4 bicubic, 5 nearest, 6 NIS. sharpness 0…1 (LS1: its five weight sets). NIS and nearest
// need no Lossless.dll. The upscale is
// written straight into the output when it's shader-writable (a drawable of a layer with framebufferOnly off);
// otherwise it goes to a scratch texture of the output's format and is copied.

private final class Context: @unchecked Sendable {
    let upscaler: LSSUpscaler
    let device: MTLDevice
    let format: MTLPixelFormat
    let outW: Int, outH: Int
    /// Made on the first output that isn't shader-writable.
    var scratch: MTLTexture?

    init(upscaler: LSSUpscaler, device: MTLDevice, format: MTLPixelFormat, outW: Int, outH: Int) {
        self.upscaler = upscaler
        self.device = device
        self.format = format
        self.outW = outW
        self.outH = outH
    }

    func scratchTexture() -> MTLTexture? {
        if let scratch { return scratch }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: outW, height: outH, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        scratch = device.makeTexture(descriptor: d)
        return scratch
    }
}

private func object<T>(_ p: UnsafeRawPointer, as: T.Type) -> T {
    Unmanaged<AnyObject>.fromOpaque(p).takeUnretainedValue() as! T
}

private func ownDirectory() -> URL {
    var info = Dl_info()
    guard dladdr(#dsohandle, &info) != 0, let path = info.dli_fname else { return URL(fileURLWithPath: ".") }
    return URL(fileURLWithPath: String(cString: path)).deletingLastPathComponent()
}

private func writeError(_ message: String, _ buffer: UnsafeMutablePointer<CChar>?, _ length: Int32) {
    guard let buffer, length > 0 else { return }
    let bytes = Array(message.utf8.prefix(Int(length) - 1))
    for (i, b) in bytes.enumerated() { buffer[i] = CChar(bitPattern: b) }
    buffer[bytes.count] = 0
}

/// Kernels are shared by every context on a device (loading them builds ~30 pipelines).
nonisolated(unsafe) private var sharedKernels: (device: ObjectIdentifier, kernels: LSSKernels)?
private let kernelsLock = NSLock()

@_cdecl("lss_bridge_create")
public func lss_bridge_create(_ device: UnsafeRawPointer, _ inW: Int32, _ inH: Int32, _ outW: Int32, _ outH: Int32,
                              _ outFormat: UInt32, _ mode: Int32, _ sharpness: Float, _ performance: Int32,
                              _ errorBuffer: UnsafeMutablePointer<CChar>?, _ errorLength: Int32) -> UnsafeMutableRawPointer? {
    let dev = object(device, as: MTLDevice.self)
    do {
        let s = max(0, min(1, sharpness))
        let m: LSSMode = switch mode {
        case 1: .ls1(sharpness: 1 + Int((s * 4).rounded()), performance: performance != 0)
        case 2: .fsr(sharpness: s)
        case 3: .bcas(sharpness: s)
        case 4: .bicubic
        case 6: .nis(sharpness: s)
        default: .nearest
        }
        kernelsLock.lock()
        defer { kernelsLock.unlock() }
        let kernels: LSSKernels?
        if !m.needsLosslessScaling {
            kernels = nil
        } else if let k = sharedKernels, k.device == ObjectIdentifier(dev) {
            kernels = k.kernels
        } else {
            let dll = ProcessInfo.processInfo.environment["LSFG_DLL"].map { URL(fileURLWithPath: $0) }
                ?? LosslessLocator.findAll().first?.dll
            guard let dll else { throw LSSError.badSize("Lossless.dll not found") }
            // Prefer the pre-baked kernels shipped next to this library (numbers from the DLL, no tools needed);
            // fall back to kernels translated on this Mac (developer setups).
            let baked = ownDirectory().appending(path: "baked")
            let loaded: LSSKernels
            if FileManager.default.fileExists(atPath: baked.appending(path: LSFGKernels.dxbcManifestName).path) {
                do { loaded = try LSSKernels(bakedDirectory: baked, dll: dll, device: dev) }
                catch let bakedError {
                    // If translating here fails too, say why the pre-baked ones didn't load: that's the one to read.
                    do { loaded = try LSSKernels(dll: dll, device: dev) }
                    catch { throw LSSError.badSize("pre-baked kernels didn't load (\(bakedError)); translating them here failed too (\(error))") }
                }
            } else {
                loaded = try LSSKernels(dll: dll, device: dev)
            }
            sharedKernels = (ObjectIdentifier(dev), loaded)
            kernels = loaded
        }
        let up = try LSSUpscaler(kernels: kernels, device: dev, mode: m, inputSize: MTLSize(width: Int(inW), height: Int(inH), depth: 1),
                                 outputSize: MTLSize(width: Int(outW), height: Int(outH), depth: 1))
        let format = MTLPixelFormat(rawValue: UInt(outFormat)) ?? .bgra8Unorm
        return Unmanaged.passRetained(Context(upscaler: up, device: dev, format: format, outW: Int(outW), outH: Int(outH))).toOpaque()
    } catch {
        writeError("\(error)", errorBuffer, errorLength)
        return nil
    }
}

/// Encodes the upscale of `input` (whole texture) into `output` at (outX, outY). Returns 1 on success.
@_cdecl("lss_bridge_encode")
public func lss_bridge_encode(_ ctx: UnsafeMutableRawPointer, _ commandBuffer: UnsafeRawPointer, _ input: UnsafeRawPointer,
                              _ output: UnsafeRawPointer, _ outX: Int32, _ outY: Int32) -> Int32 {
    let c = Unmanaged<Context>.fromOpaque(ctx).takeUnretainedValue()
    let cb = object(commandBuffer, as: MTLCommandBuffer.self)
    let src = object(input, as: MTLTexture.self)
    let dst = object(output, as: MTLTexture.self)
    guard Int(outX) + c.outW <= dst.width, Int(outY) + c.outH <= dst.height, dst.pixelFormat == c.format else { return 0 }
    if dst.usage.contains(.shaderWrite) {
        c.upscaler.encode(commandBuffer: cb, input: src, output: dst, outputOrigin: MTLOrigin(x: Int(outX), y: Int(outY), z: 0))
        return 1
    }
    guard let scratch = c.scratchTexture() else { return 0 }
    c.upscaler.encode(commandBuffer: cb, input: src, output: scratch)
    guard let blit = cb.makeBlitCommandEncoder() else { return 0 }
    blit.copy(from: scratch, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(), sourceSize: MTLSize(width: c.outW, height: c.outH, depth: 1),
              to: dst, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin(x: Int(outX), y: Int(outY), z: 0))
    blit.endEncoding()
    return 1
}

@_cdecl("lss_bridge_destroy")
public func lss_bridge_destroy(_ ctx: UnsafeMutableRawPointer) {
    Unmanaged<Context>.fromOpaque(ctx).release()
}
