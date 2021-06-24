Swift Library
============

The [Swift][] `bril` library, which lives in the `bril-swift` directory, provides a Swift interface for Bril's JSON files. It supports the [Bril core][core] and the [SSA][] extension.

Use
---

To use this package in a SwiftPM project, add a dependency to your Package.swift:

```
let package = Package(
  name: "MyPackage",
  dependencies: [
    .package(name: "Bril", path: "../bril-swift"),
  ]
)
```

and add `"Bril"` to the dependencies array for any target that needs it:

```
targets: [
    .target(
        name: "MyTarget",
        dependencies: ["Bril"]),
// ...
```

The Bril objects conform to `Decodable`. Instantiate a program from data as follows:

```
import Bril

/// to read from stdin:
/// let data = FileHandle.standardInput.availableData

/// or from a string:
/// let data = "<Bril JSON>".data(using: .utf8)!

let program = try JSONDecoder().decode(Program.self, from: data)
```

The models conform to `CustomStringConvertible` so printing the Bril representation is simply:

```
print(program)
```

[swift]: https://swift.org/about/
[core]: ../lang/core.md
[ssa]: ../lang/ssa.md
