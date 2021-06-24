public struct ConstantOperation: Equatable {
    static let opName = "const"

    public let name = opName
    public var destination: String
    public let type: Type
    public let value: Literal

    public init(destination: String, type: Type, value: Literal) {
        self.destination = destination
        self.type = type
        self.value = value
    }
}

public enum Literal {
    case bool(Bool)
    case int(Int)
}

extension Literal {
    public var int: Int? {
        switch self {
            case .bool: return nil
            case .int(let int): return int
        }
    }

    public var bool: Bool? {
        switch self {
            case .bool(let bool): return bool
            case .int: return nil
        }
    }
}

extension Literal: Hashable { }

extension Literal: CustomStringConvertible {
    public var description: String {
        switch self {
            case .bool(let value): return "\(value)"
            case .int(let value): return "\(value)"
        }
    }
}
