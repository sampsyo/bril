public enum Code: Equatable {
    case label(String)
    case instruction(Instruction)
}

extension Code: Decodable {
    enum CodingKeys: CodingKey {
        case label
        case op
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let label = try container.decodeIfPresent(String.self, forKey: .label) {
            self = .label(label)
        } else if container.contains(.op) {
            self = .instruction(try Instruction(from: decoder))
        } else {
            throw BrilParseError(message: "instr entry did not contain 'label' or 'op' field")
        }
    }
}

extension Code: CustomStringConvertible {
    public var description: String {
        switch self {
            case .label(let label):
                return ".\(label):"
            case .instruction(let instruction):
                return "  \(instruction)"
        }
    }
}

// convenience properties

extension Code {
    public var operation: Operation? {
        guard case .instruction(let instruction) = self else {
            return nil
        }
        return instruction.operation
    }
}

extension Code {
    public var arguments: [String] { operation?.arguments ?? [] }
    public var functions: [String] { operation?.functions ?? [] }
    public var labels: [String] { operation?.labels ?? [] }
    public var destinationIfPresent: String? { operation?.destinationIfPresent }
    public var typeIfPresent: Type? { operation?.typeIfPresent }
}
