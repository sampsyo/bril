Type Inference
==============

Bril requires exhaustive type annotations on every instruction, which can quickly get tedious.
The `type-infer` directory contains a simple global type inference tool that fills in missing type annotations.
For example, it can turn this easier-to-write program:

    @main(arg: int) {
      five = const 5;
      ten = const 10;
      res = add arg five;
      cond = le res ten;
      br cond .then .else;
    .then:
      print res;
    .else:
    }

Into this actually executable program:

    @main(arg: int) {
      five: int = const 5;
      ten: int = const 10;
      res: int = add arg five;
      cond: bool = le res ten;
      br cond .then .else;
    .then:
      print res;
    .else:
    }

The tool is a simple Python program, `infer.py`, that takes JSON programs that are missing types and adds types to them.
It is also useful even on fully-typed programs as a type *checker* to rule out common run-time errors.
The included [text format tools](text.md) support missing types for both parsing and printing, so here's a shell pipeline that adds types to your text-format Bril program:

    cat myprog.bril | bril2json | python type-infer/infer.py | bril2txt

You can read [more about the inference tool][inferblog], which is originally by Christopher Roman.

[inferblog]: https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/bril-type-inference/
