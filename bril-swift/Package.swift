// swift-tools-version:5.3

import PackageDescription

let package = Package(
    name: "Bril",
    products: [
        .library(
            name: "Bril",
            targets: ["Bril"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "Bril",
            dependencies: []),
        .testTarget(
            name: "BrilTests",
            dependencies: ["Bril"]),
    ]
)
