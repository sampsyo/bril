Bril Text Format
================

While Bril's canonical representation is a JSON AST, humans don't like to read and write JSON.
To accommodate our human foibles, we also have a simple textual representation.
There is a parser and pretty printer tool that can convert the text representation to and from JSON.

For example, this Bril program in JSON:

    {
      "functions": [{
        "name": "main",
        "instrs": [
          { "op": "const", "type": "int", "dest": "v0", "value": 1 },
          { "op": "const", "type": "int", "dest": "v1", "value": 2 },
          { "op": "add", "type": "int", "dest": "v2", "args": ["v0", "v1"] },
          { "op": "print", "args": ["v2"] },
          { "op": "alloc", "type": { "ptr" : "int" }, "dest": "v3", "args": ["v0"] },
          { "op": "free", "args": ["v3"] },
        ]
      }]
    }

Gets represented in text like this:

    @main {
      v0: int = const 1;
      v1: int = const 2;
      v2: int = add v0 v1;
      print v2;
      v3: ptr<int> = alloc v0;
      free v3;
    }

Tools
-----

[The `bril-txt` parser & pretty printer][briltxt] are written in Python.
You can install them with [Flit][] by doing something like:

    $ pip install --user flit
    $ cd bril-txt
    $ flit install --symlink --user

You'll now have tools called `bril2json` and `bril2txt`.
Both read from standard input and write to standard output.
You can try a "round trip" like this, for example:

    $ bril2json < test/parse/add.bril | bril2txt

The `bril2json` parser also supports a `-p` flag to include [source positions](../lang/syntax.md#source-positions).

[flit]: https://flit.readthedocs.io/
[briltxt]: https://github.com/sampsyo/bril/blob/main/bril-txt/briltxt.py
