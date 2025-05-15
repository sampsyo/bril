{
  "functions": [
    {
      "args": [
        {
          "name": "n",
          "type": "int"
        }
      ],
      "instrs": [
        {
          "args": [
            "n"
          ],
          "op": "print"
        },
        {
          "args": [
            "n"
          ],
          "dest": "tot",
          "funcs": [
            "totient"
          ],
          "op": "call",
          "type": "int"
        },
        {
          "args": [
            "tot"
          ],
          "op": "print"
        }
      ],
      "name": "main"
    },
    {
      "args": [
        {
          "name": "n",
          "type": "int"
        }
      ],
      "instrs": [
        {
          "args": [
            "n"
          ],
          "dest": "result",
          "op": "id",
          "type": "int"
        },
        {
          "dest": "p",
          "op": "const",
          "type": "int",
          "value": 2
        },
        {
          "dest": "one",
          "op": "const",
          "type": "int",
          "value": 1
        },
        {
          "dest": "zero",
          "op": "const",
          "type": "int",
          "value": 0
        },
        {
          "label": "for.set.cond"
        },
        {
          "args": [
            "p",
            "p"
          ],
          "dest": "pp",
          "op": "mul",
          "type": "int"
        },
        {
          "args": [
            "pp",
            "n"
          ],
          "dest": "cond",
          "op": "le",
          "type": "bool"
        },
        {
          "args": [
            "cond"
          ],
          "labels": [
            "for.set.body",
            "for.set.end"
          ],
          "op": "br"
        },
        {
          "label": "for.set.body"
        },
        {
          "args": [
            "n",
            "p"
          ],
          "dest": "npmod",
          "funcs": [
            "mod"
          ],
          "op": "call",
          "type": "int"
        },
        {
          "args": [
            "npmod",
            "zero"
          ],
          "dest": "if_cond",
          "op": "eq",
          "type": "bool"
        },
        {
          "args": [
            "if_cond"
          ],
          "labels": [
            "if_lbl",
            "else_lbl"
          ],
          "op": "br"
        },
        {
          "label": "if_lbl"
        },
        {
          "label": "while.set.cond"
        },
        {
          "args": [
            "n",
            "p"
          ],
          "dest": "npmod",
          "funcs": [
            "mod"
          ],
          "op": "call",
          "type": "int"
        },
        {
          "args": [
            "npmod",
            "zero"
          ],
          "dest": "while_cond",
          "op": "eq",
          "type": "bool"
        },
        {
          "args": [
            "while_cond"
          ],
          "labels": [
            "while.body",
            "while.end"
          ],
          "op": "br"
        },
        {
          "label": "while.body"
        },
        {
          "args": [
            "n",
            "p"
          ],
          "dest": "npdiv",
          "op": "div",
          "type": "int"
        },
        {
          "args": [
            "npdiv"
          ],
          "dest": "n",
          "op": "id",
          "type": "int"
        },
        {
          "labels": [
            "while.set.cond"
          ],
          "op": "jmp"
        },
        {
          "label": "while.end"
        },
        {
          "args": [
            "result",
            "p"
          ],
          "dest": "resdiv",
          "op": "div",
          "type": "int"
        },
        {
          "args": [
            "result",
            "resdiv"
          ],
          "dest": "result",
          "op": "sub",
          "type": "int"
        },
        {
          "label": "else_lbl"
        },
        {
          "args": [
            "p",
            "one"
          ],
          "dest": "p",
          "op": "add",
          "type": "int"
        },
        {
          "labels": [
            "for.set.cond"
          ],
          "op": "jmp"
        },
        {
          "label": "for.set.end"
        },
        {
          "args": [
            "n",
            "one"
          ],
          "dest": "final_if_cond",
          "op": "gt",
          "type": "bool"
        },
        {
          "args": [
            "final_if_cond"
          ],
          "labels": [
            "final_if_label",
            "final_else_label"
          ],
          "op": "br"
        },
        {
          "label": "final_if_label"
        },
        {
          "args": [
            "result",
            "n"
          ],
          "dest": "resdiv",
          "op": "div",
          "type": "int"
        },
        {
          "args": [
            "result",
            "resdiv"
          ],
          "dest": "result",
          "op": "sub",
          "type": "int"
        },
        {
          "label": "final_else_label"
        },
        {
          "args": [
            "result"
          ],
          "op": "ret"
        }
      ],
      "name": "totient",
      "type": "int"
    },
    {
      "args": [
        {
          "name": "a",
          "type": "int"
        },
        {
          "name": "b",
          "type": "int"
        }
      ],
      "instrs": [
        {
          "args": [
            "a",
            "b"
          ],
          "dest": "ad",
          "op": "div",
          "type": "int"
        },
        {
          "args": [
            "b",
            "ad"
          ],
          "dest": "mad",
          "op": "mul",
          "type": "int"
        },
        {
          "args": [
            "a",
            "mad"
          ],
          "dest": "ans",
          "op": "sub",
          "type": "int"
        },
        {
          "args": [
            "ans"
          ],
          "op": "ret"
        }
      ],
      "name": "mod",
      "type": "int"
    }
  ]
}
