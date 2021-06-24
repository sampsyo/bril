    import XCTest
    @testable import Bril

    final class BrilTests: XCTestCase {
        func testDecode() throws {
            let program = try JSONDecoder().decode(Program.self, from: source)
            XCTAssertEqual(program.functions.count, 1)
            XCTAssertEqual(program.functions[0].code.count, 15)
        }

        private let source = """
            {
              "functions": [
                {
                  "args": [
                    {
                      "name": "cond",
                      "type": "bool"
                    }
                  ],
                  "instrs": [
                    {
                      "label": "entry"
                    },
                    {
                      "dest": "a",
                      "op": "const",
                      "type": "int",
                      "value": 47
                    },
                    {
                      "dest": "b",
                      "op": "const",
                      "type": "int",
                      "value": 2
                    },
                    {
                      "args": [
                        "b"
                      ],
                      "op": "print"
                    },
                    {
                      "args": [
                        "cond"
                      ],
                      "labels": [
                        "left",
                        "right"
                      ],
                      "op": "br"
                    },
                    {
                      "label": "left"
                    },
                    {
                      "args": [
                        "a",
                        "a"
                      ],
                      "dest": "a",
                      "op": "add",
                      "type": "int"
                    },
                    {
                      "dest": "b",
                      "op": "const",
                      "type": "int",
                      "value": 3
                    },
                    {
                      "args": [
                        "b"
                      ],
                      "op": "print"
                    },
                    {
                      "labels": [
                        "exit"
                      ],
                      "op": "jmp"
                    },
                    {
                      "label": "right"
                    },
                    {
                      "args": [
                        "a",
                        "a"
                      ],
                      "dest": "a",
                      "op": "mul",
                      "type": "int"
                    },
                    {
                      "labels": [
                        "exit"
                      ],
                      "op": "jmp"
                    },
                    {
                      "label": "exit"
                    },
                    {
                      "args": [
                        "a"
                      ],
                      "op": "print"
                    }
                  ],
                  "name": "main"
                }
              ]
            }
""".data(using: .utf8)!
    }
