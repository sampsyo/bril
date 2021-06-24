public struct Argument: Equatable, Decodable {
    public var name: String
    public var type: Type

    public init(name: String, type: Type) {
        self.name = name
        self.type = type
    }
}

extension Argument: CustomStringConvertible {
    public var description: String { "\(name): \(type)" }
}
