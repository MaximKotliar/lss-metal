// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LSSMetal",
    platforms: [.macOS(.v15)],
    products: [
        // Lossless Scaling's spatial upscalers (LS1, FSR 1, bicubic) on Metal, from the user's own Lossless.dll.
        .library(name: "LSSMetal", targets: ["LSSMetal"]),
        .executable(name: "lss-metal", targets: ["lss-metal"]),
        // The same as a C API (lss_bridge_*), for hosts that aren't Swift; builds for x86_64 too.
        .library(name: "LSSBridge", type: .dynamic, targets: ["LSSBridge"]),
    ],
    dependencies: [
        // Finds Lossless.dll and translates its shaders (the same cache LSFG uses).
        .package(url: "https://github.com/MaximKotliar/lsfg-metal.git", branch: "main"),
    ],
    targets: [
        .target(name: "LSSMetal", dependencies: [.product(name: "LSFGCore", package: "lsfg-metal")]),
        .target(name: "LSSBridge", dependencies: ["LSSMetal", .product(name: "LSFGCore", package: "lsfg-metal")]),
        .executableTarget(name: "lss-metal", dependencies: ["LSSMetal", .product(name: "LSFGCore", package: "lsfg-metal")],
                          swiftSettings: [.swiftLanguageMode(.v5)]),   // a plain script-style tool
    ]
)
