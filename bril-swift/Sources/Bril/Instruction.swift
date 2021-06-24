public enum Instruction: Equatable {
    case const(ConstantOperation)
    case value(ValueOperation)
    case effect(EffectOperation)
}

extension Instruction {
    public var operation: Operation {
        switch self {
            case .const(let op): return op
            case .value(let op): return op
            case .effect(let op): return op
        }
    }
}

extension Instruction: Decodable {
    enum CodingKeys: CodingKey {
        case op
        case dest
        case type
        case args
        case funcs
        case labels
        case value
    }

    private static func makeConstant(from decoder: Decoder) throws -> Instruction {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        guard let dest = try? container.decodeIfPresent(String.self, forKey: .dest) else {
            throw BrilParseError(message: "'const' field 'dest' is missing")
        }

        guard let type = try? container.decodeIfPresent(Type.self, forKey: .type) else {
            throw BrilParseError(message: "'const' field 'type' is missing")
        }

        let literal: Literal
        if let literalBool = try? container.decodeIfPresent(Bool.self, forKey: .value) {
            literal = .bool(literalBool)
        } else if let literalInt = try? container.decodeIfPresent(Int.self, forKey: .value) {
            literal = .int(literalInt)
        } else {
            throw BrilParseError(message: "const field 'value' is missing or is not a valid type")
        }
        return .const(.init(destination: dest, type: type, value: literal))
    }

    private static func makeCall(from decoder: Decoder) throws -> Instruction {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let args = (try? container.decodeIfPresent([String].self, forKey: .args)) ?? []
        let funcs = (try? container.decodeIfPresent([String].self, forKey: .funcs)) ?? []
        let labels = (try? container.decodeIfPresent([String].self, forKey: .labels)) ?? []

        if let dest = try? container.decodeIfPresent(String.self, forKey: .dest) {
            guard let type = try? container.decodeIfPresent(Type.self, forKey: .type) else {
                throw BrilParseError(message: "'call' with 'dest' must have 'type'")
            }
            return .value(.init(opType: .call, destination: dest, type: type, arguments: args, functions: funcs, labels: labels))
        } else {
            return .effect(.init(opType: .call, arguments: args, functions: funcs, labels: labels))
        }
    }

    private static func makeValue(op: ValueOperation.OpType, from decoder: Decoder) throws -> Instruction {
        let container = try decoder.container(keyedBy: CodingKeys.self)


        guard let dest = try? container.decodeIfPresent(String.self, forKey: .dest) else {
            throw BrilParseError(message: "'\(op.rawValue)' field 'dest' is missing")
        }

        guard let type = try? container.decodeIfPresent(Type.self, forKey: .type) else {
            throw BrilParseError(message: "'\(op.rawValue)' field 'type' is missing")
        }

        let args = (try? container.decodeIfPresent([String].self, forKey: .args)) ?? []
        let funcs = (try? container.decodeIfPresent([String].self, forKey: .funcs)) ?? []
        let labels = (try? container.decodeIfPresent([String].self, forKey: .labels)) ?? []

        return .value(.init(opType: op, destination: dest, type: type, arguments: args, functions: funcs, labels: labels))
    }

    private static func makeEffect(op: EffectOperation.OpType, from decoder: Decoder) throws -> Instruction {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let args = (try? container.decodeIfPresent([String].self, forKey: .args)) ?? []
        let funcs = (try? container.decodeIfPresent([String].self, forKey: .funcs)) ?? []
        let labels = (try? container.decodeIfPresent([String].self, forKey: .labels)) ?? []

        return .effect(.init(opType: op, arguments: args, functions: funcs, labels: labels))
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        guard let opStr = try? container.decodeIfPresent(String.self, forKey: .op) else {
            throw BrilParseError(message: "Instruction did not contain 'op' field")
        }
        if opStr == ConstantOperation.opName {
            self = try Self.makeConstant(from: decoder)
        } else if opStr == ValueOperation.OpType.call.rawValue {
            self = try Self.makeCall(from: decoder)
        } else if let op = ValueOperation.OpType(rawValue: opStr) {
            self = try Self.makeValue(op: op, from: decoder)
        } else if let op = EffectOperation.OpType(rawValue: opStr) {
            self = try Self.makeEffect(op: op, from: decoder)
        } else {
            throw BrilParseError(message: "unknown op '\(opStr)'")
        }
    }
}

extension Instruction: CustomStringConvertible {
    private func appendToDescription(_ description: String, functions: [String], arguments: [String], labels: [String]) -> String {
        description +
            functions.map { " @\($0)" }.joined() +
            arguments.map { " \($0)" }.joined() +
            labels.map { " .\($0)" }.joined() +
            ";"
    }

    public var description: String {
        switch self {
            case .const(let op):
                return "\(op.destination): \(op.type) = \(op.name) \(op.value);"
            case .value(let op):
                let description = "\(op.destination): \(op.type) = \(op.name)"
                return appendToDescription(description, functions: op.functions, arguments: op.arguments, labels: op.labels)
            case .effect(let op):
                return appendToDescription(op.name, functions: op.functions, arguments: op.arguments, labels: op.labels)
        }
    }
}
