Import
======

Typically, Bril programs are self-contained: they only use functions defined elsewhere in the same program. This *import* extension lets Bril code use functions defined in other files.

A Bril import refers to a file and lists the functions to import from it, like this:

    {
        "path": "my_library.json",
        "functions": [{"name": "libfunc"}]
    }

This import assumes that there's a Bril file called `my_library.json`, and that it declares a function `@libfunc`. The current Bril file may now invoke `@libfunc` as if it were defined locally.

Syntax
------

The top-level Bril program is extended with an `imports` field:

    { "functions": [<Function>, ...], "imports": [<Import>, ...] }

Each import object has this syntax:

    {
        "path": "<string>",
        "functions": [
            { "name": "<string>", "alias": "<string>"? },
            ...
        ]
    }

The path is a relative reference to a Bril JSON file containing the functions to import. In the objects in the `functions` list, the `name` is the *original* name of the function, and the optional `alias` is the *local* name that the program will use to refer to the function. A missing `alias` makes the local name equal to the original name.

It is an error to refer to functions that do not exist, or to create naming conflicts between imports and local functions (or between different imports). Import cycles are allowed.

Text Format
-----------

In Bril's [text format](../tools/text.md), the `import` syntax looks like this:

    from "something.json" import @libfunc, @otherfunc as @myfunc;

Search Paths
------------

We do not define the exact mechanism for using the `path` string to find the file to import. Reasonable options include:

* Resolve the path relative to the file the `import` appears in.
* Use a pre-defined set of library search paths.

We only specify what it means to import JSON files; implementations can choose to allow importing other kinds of files too (e.g., text-format source code).
