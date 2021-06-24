public struct ValueOperation: Equatable {
    public enum OpType: String {
        case add
        case sub
        case mul
        case div
        case eq
        case lt
        case gt
        case le
        case ge
        case not
        case and
        case or
        case call
        case id
        case phi
    }
    public let opType: OpType
    public var destination: String
    public let type: Type
    public var arguments: [String]
    public let functions: [String]
    public let labels: [String]

    public init(opType: OpType, destination: String, type: Type, arguments: [String] = [], functions: [String] = [], labels: [String] = []) {
        self.opType = opType
        self.destination = destination
        self.type = type
        self.arguments = arguments
        self.functions = functions
        self.labels = labels
    }
}

extension ValueOperation {
    public var name: String { opType.rawValue }
}
