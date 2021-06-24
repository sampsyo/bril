public struct Program: Equatable, Decodable {
    public var functions: [Function]

    public init(functions: [Function]) {
        self.functions = functions
    }
}

extension Program: CustomStringConvertible {
    public var description: String {
        functions.map(String.init).joined(separator: "\n\n")
    }
}
