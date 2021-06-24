/// A convenience protocol for accessing data across the three Operation types.
public protocol Operation {
    var destinationIfPresent: String? { get }
    var typeIfPresent: Type? { get }
    var valueIfPresent: Literal? { get }

    var name: String { get }
    var arguments: [String] { get }
    var functions: [String] { get }
    var labels: [String] { get }
}

extension Operation {
    public var destinationIfPresent: String? { nil }
    public var typeIfPresent: Type? { nil }
    public var valueIfPresent: Literal? { nil }

    public var arguments: [String] { [] }
    public var functions: [String] { [] }
    public var labels: [String] { [] }
}

extension ConstantOperation: Operation {
    public var destinationIfPresent: String? { destination }
    public var typeIfPresent: Type? { type }
    public var valueIfPresent: Literal? { value }
}

extension ValueOperation: Operation {
    public var destinationIfPresent: String? { destination }
    public var typeIfPresent: Type? { type }
}

extension EffectOperation: Operation { }
