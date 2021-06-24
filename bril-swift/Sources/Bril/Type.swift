public indirect enum Type: Equatable {
    case primitive(String)
    case parameterized(String, Type)
}

extension Type: Decodable {
    public init(from decoder: Decoder) throws {
        if let str = try? decoder.singleValueContainer().decode(String.self) {
            self = .primitive(str)
        } else {
            var container = try decoder.unkeyedContainer()
            let dict = try container.decode([String: Type].self)
            guard let entry = dict.first else {
                throw BrilParseError(message: "Empty Type object")
            }
            self = .parameterized(entry.key, entry.value)

        }
    }
}

extension Type: CustomStringConvertible {
    public var description: String {
        switch self {
            case .primitive(let type):
                return type
            case .parameterized(let wrapper, let type):
                return "\(wrapper)<\(type)>"
        }
    }
}
