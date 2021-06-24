public struct EffectOperation: Equatable {
    public enum OpType: String {
        case jmp
        case br
        case call
        case ret
        case print
        case nop
    }
    public let opType: OpType
    public var arguments: [String]
    public let functions: [String]
    public let labels: [String]

    public init(opType: OpType, arguments: [String] = [], functions: [String] = [], labels: [String] = []) {
        self.opType = opType
        self.arguments = arguments
        self.functions = functions
        self.labels = labels
    }
}

extension EffectOperation {
    public var name: String { opType.rawValue }
}
