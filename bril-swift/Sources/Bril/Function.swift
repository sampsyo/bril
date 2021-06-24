public struct Function: Equatable {
    public var name: String
    public var arguments: [Argument]
    public var type: Type?
    public var code: [Code]

    public init(name: String, arguments: [Argument] = [], type: Type? = nil, code: [Code] = []) {
        self.name = name
        self.arguments = arguments
        self.type = type
        self.code = code
    }
}

extension Function: Decodable {
    enum CodingKeys: String, CodingKey {
        case name
        case arguments = "args"
        case type
        case code = "instrs"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        arguments = (try? container.decodeIfPresent([Argument].self, forKey: .arguments)) ?? []
        type = try container.decodeIfPresent(Type.self, forKey: .type)
        code = (try? container.decodeIfPresent([Code].self, forKey: .code)) ?? []
    }
}

extension Function: CustomStringConvertible {
    public var description: String {
        var val = "@\(name)"

        if !arguments.isEmpty {
            val += "(" + arguments.map(String.init).joined(separator: ", ") + ")"
        }

        if let type = type {
            val += ": \(type)"
        }
        val += " {\n" + code.map(String.init).joined(separator: "\n") + "\n}"

        return val
    }
}
