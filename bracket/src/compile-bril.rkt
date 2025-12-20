#lang racket

;; Copyright (C) 2025 Joseph Maheshe.

(require
  cpsc411/compiler-lib)

(provide
 compile-bril)

;; =============================================================================
;; 1. Standard Library Definitions (Manual Translation to Bril JSON)
;; =============================================================================

;; Helpers for manual construction
(define (mk-const dest type val) (hasheq 'op "const" 'dest dest 'type type 'value val))
(define (mk-op op dest type args) (hasheq 'op op 'dest dest 'type type 'args args))
(define (mk-eff op args) (hasheq 'op op 'args args))
(define (mk-call dest type func args) (hasheq 'op "call" 'dest dest 'type type 'funcs (list func) 'args args))
(define (mk-lbl name) (hasheq 'label name))
(define (mk-jmp lbl) (hasheq 'op "jmp" 'labels (list lbl)))
(define (mk-br arg true-lbl false-lbl) (hasheq 'op "br" 'args (list arg) 'labels (list true-lbl false-lbl)))

(define std-lib-funcs
  (list
   ;; @bit (x: int): int
   (hasheq 'name "bit"
           'args (list (hasheq 'name "x" 'type "int"))
           'type "int"
           'instrs (list
                    (mk-const "two" "int" 2)
                    (mk-op "div" "div2" "int" (list "x" "two"))
                    (mk-op "mul" "mul2" "int" (list "div2" "two"))
                    (mk-op "sub" "bit" "int" (list "x" "mul2"))
                    (mk-eff "ret" (list "bit"))))

   ;; @bitand (a: int, b: int): int
   (hasheq 'name "bitand"
           'args (list (hasheq 'name "a" 'type "int") (hasheq 'name "b" 'type "int"))
           'type "int"
           'instrs (list
                    (mk-const "zero" "int" 0)
                    (mk-const "one" "int" 1)
                    (mk-const "two" "int" 2)
                    (mk-const "result" "int" 0)
                    (mk-const "pow" "int" 1)
                    ;; .loop.cond
                    (mk-lbl "loop.cond")
                    (mk-op "gt" "acond" "bool" (list "a" "zero"))
                    (mk-op "gt" "bcond" "bool" (list "b" "zero"))
                    (mk-op "or" "cond" "bool" (list "acond" "bcond"))
                    (mk-br "cond" "loop.body" "loop.end")
                    ;; .loop.body
                    (mk-lbl "loop.body")
                    (mk-call "abit" "int" "bit" (list "a"))
                    (mk-call "bbit" "int" "bit" (list "b"))
                    (mk-op "eq" "abit1" "bool" (list "abit" "one"))
                    (mk-op "eq" "bbit1" "bool" (list "bbit" "one"))
                    (mk-op "and" "rbitb" "bool" (list "abit1" "bbit1"))
                    (mk-const "rbit" "int" 0)
                    (mk-br "rbitb" "setone" "skip")
                    ;; .setone
                    (mk-lbl "setone")
                    (mk-const "rbit" "int" 1)
                    ;; .skip
                    (mk-lbl "skip")
                    (mk-op "mul" "term" "int" (list "rbit" "pow"))
                    (mk-op "add" "result" "int" (list "result" "term"))
                    (mk-op "div" "a" "int" (list "a" "two"))
                    (mk-op "div" "b" "int" (list "b" "two"))
                    (mk-op "mul" "pow" "int" (list "pow" "two"))
                    (mk-jmp "loop.cond")
                    ;; .loop.end
                    (mk-lbl "loop.end")
                    (mk-eff "ret" (list "result"))))

   ;; @bitor (a: int, b: int): int
   (hasheq 'name "bitor"
           'args (list (hasheq 'name "a" 'type "int") (hasheq 'name "b" 'type "int"))
           'type "int"
           'instrs (list
                    (mk-const "zero" "int" 0)
                    (mk-const "one" "int" 1)
                    (mk-const "two" "int" 2)
                    (mk-const "result" "int" 0)
                    (mk-const "pow" "int" 1)
                    ;; .loop.cond
                    (mk-lbl "loop.cond.or")
                    (mk-op "gt" "acond" "bool" (list "a" "zero"))
                    (mk-op "gt" "bcond" "bool" (list "b" "zero"))
                    (mk-op "or" "cond" "bool" (list "acond" "bcond"))
                    (mk-br "cond" "loop.body.or" "loop.end.or")
                    ;; .loop.body
                    (mk-lbl "loop.body.or")
                    (mk-call "abit" "int" "bit" (list "a"))
                    (mk-call "bbit" "int" "bit" (list "b"))
                    (mk-op "eq" "abit1" "bool" (list "abit" "one"))
                    (mk-op "eq" "bbit1" "bool" (list "bbit" "one"))
                    (mk-op "or" "rbitb" "bool" (list "abit1" "bbit1"))
                    (mk-const "rbit" "int" 0)
                    (mk-br "rbitb" "setone.or" "skip.or")
                    ;; .setone
                    (mk-lbl "setone.or")
                    (mk-const "rbit" "int" 1)
                    ;; .skip
                    (mk-lbl "skip.or")
                    (mk-op "mul" "term" "int" (list "rbit" "pow"))
                    (mk-op "add" "result" "int" (list "result" "term"))
                    (mk-op "div" "a" "int" (list "a" "two"))
                    (mk-op "div" "b" "int" (list "b" "two"))
                    (mk-op "mul" "pow" "int" (list "pow" "two"))
                    (mk-jmp "loop.cond.or")
                    ;; .loop.end
                    (mk-lbl "loop.end.or")
                    (mk-eff "ret" (list "result"))))

   ;; @bitxor (a: int, b: int): int
   (hasheq 'name "bitxor"
           'args (list (hasheq 'name "a" 'type "int") (hasheq 'name "b" 'type "int"))
           'type "int"
           'instrs (list
                    (mk-const "zero" "int" 0)
                    (mk-const "one" "int" 1)
                    (mk-const "two" "int" 2)
                    (mk-const "result" "int" 0)
                    (mk-const "pow" "int" 1)
                    ;; .loop.cond
                    (mk-lbl "loop.cond.xor")
                    (mk-op "gt" "acond" "bool" (list "a" "zero"))
                    (mk-op "gt" "bcond" "bool" (list "b" "zero"))
                    (mk-op "or" "cond" "bool" (list "acond" "bcond"))
                    (mk-br "cond" "loop.body.xor" "loop.end.xor")
                    ;; .loop.body
                    (mk-lbl "loop.body.xor")
                    (mk-call "abit" "int" "bit" (list "a"))
                    (mk-call "bbit" "int" "bit" (list "b"))
                    (mk-op "eq" "abit1" "bool" (list "abit" "one"))
                    (mk-op "eq" "bbit1" "bool" (list "bbit" "one"))
                    (mk-op "not" "nb" "bool" (list "bbit1"))
                    (mk-op "not" "na" "bool" (list "abit1"))
                    (mk-op "and" "t1" "bool" (list "abit1" "nb"))
                    (mk-op "and" "t2" "bool" (list "na" "bbit1"))
                    (mk-op "or" "rbitb" "bool" (list "t1" "t2"))
                    (mk-const "rbit" "int" 0)
                    (mk-br "rbitb" "setone.xor" "skip.xor")
                    ;; .setone
                    (mk-lbl "setone.xor")
                    (mk-const "rbit" "int" 1)
                    ;; .skip
                    (mk-lbl "skip.xor")
                    (mk-op "mul" "term" "int" (list "rbit" "pow"))
                    (mk-op "add" "result" "int" (list "result" "term"))
                    (mk-op "div" "a" "int" (list "a" "two"))
                    (mk-op "div" "b" "int" (list "b" "two"))
                    (mk-op "mul" "pow" "int" (list "pow" "two"))
                    (mk-jmp "loop.cond.xor")
                    ;; .loop.end
                    (mk-lbl "loop.end.xor")
                    (mk-eff "ret" (list "result"))))

   ;; @ashr (x: int, n: int): int
   (hasheq 'name "ashr"
           'args (list (hasheq 'name "x" 'type "int") (hasheq 'name "n" 'type "int"))
           'type "int"
           'instrs (list
                    (mk-const "zero" "int" 0)
                    (mk-const "one" "int" 1)
                    (mk-const "two" "int" 2)
                    ;; .loop.cond
                    (mk-lbl "loop.cond.shr")
                    (mk-op "gt" "npos" "bool" (list "n" "zero"))
                    (mk-br "npos" "loop.body.shr" "loop.end.shr")
                    ;; .loop.body
                    (mk-lbl "loop.body.shr")
                    (mk-op "div" "q" "int" (list "x" "two"))
                    (mk-op "mul" "q2" "int" (list "q" "two"))
                    (mk-op "sub" "r" "int" (list "x" "q2"))
                    (mk-op "lt" "neg" "bool" (list "x" "zero"))
                    (mk-op "eq" "rzero" "bool" (list "r" "zero"))
                    (mk-op "not" "odd" "bool" (list "rzero"))
                    (mk-op "and" "fix" "bool" (list "neg" "odd"))
                    (mk-br "fix" "adjust" "no_adjust")
                    ;; .adjust
                    (mk-lbl "adjust")
                    (mk-op "sub" "q" "int" (list "q" "one"))
                    (mk-jmp "after")
                    ;; .no_adjust
                    (mk-lbl "no_adjust")
                    (mk-jmp "after")
                    ;; .after
                    (mk-lbl "after")
                    (mk-op "id" "x" "int" (list "q"))
                    (mk-op "sub" "n" "int" (list "n" "one"))
                    (mk-jmp "loop.cond.shr")
                    ;; .loop.end
                    (mk-lbl "loop.end.shr")
                    (mk-eff "ret" (list "x"))))
   ))

;; =============================================================================
;; 2. Main Compiler
;; =============================================================================
;; Main compiler pass: proc-imp-cmf-lang-v7 -> Bril JSON
(define (compile-bril p)
  
  ;; Convert aloc/register to string
  (define (loc->string loc)
    (let ([s (cond
               [(symbol? loc) (symbol->string loc)]
               [(string? loc) loc]
               [else (error "Unknown location type" loc)])])
      ;; Sanitize to follow the IDENT rule: ("_"|"%"|LETTER) ("_"|"%"|"."|LETTER|DIGIT)*
      ;; 1. Replace '?' with '.p' (predicate)
      ;; 2. Replace '!' with '.e' (effect/bang)
      ;; 3. Replace '-' with '_' 
      (regexp-replace* #rx"-" 
        (regexp-replace* #rx"!" 
          (regexp-replace* #rx"\\?" s ".p") ".e") "_")))
  
  ;; Compile the program
  (define (compile-program program)
    (match program
      [`(module (define ,labels (lambda (,alocss ...) ,entries)) ... ,entry)
       (define funcs (map compile-function labels alocss entries))
       (define main-func (compile-main-function entry))
       ;; append std-lib-funcs to the output
       (hasheq 'functions (append std-lib-funcs funcs (list main-func)))]))
  
  ;; Compile a function definition
  (define (compile-function label alocs entry)
    (define instrs '())
    (define (emit! instr) (set! instrs (append instrs (list instr))))
    
    ;; Compile the entry tail
    (compile-tail entry emit! #f)
    
    (hasheq 'name (loc->string label)
            'args (for/list ([aloc alocs])
                    (hasheq 'name (loc->string aloc) 'type "any"))
            'type "any"
            'instrs instrs))
  
  ;; Compile main function
  (define (compile-main-function entry)
    (define instrs '())
    (define (emit! instr) (set! instrs (append instrs (list instr))))
    
    (compile-tail entry emit! #t)
    
    (hasheq 'name "main"
            'type "any"
            'instrs instrs))
  
  ;; Compile a tail expression
  (define (compile-tail tail emit! main?)
    (match tail
      [`(call ,triv ,opands ...)
       (compile-call triv opands emit! #t main?)]
      [`(begin ,effects ... ,tail)
       (for ([eff effects])
         (compile-effect eff emit! #f))
       (compile-tail tail emit! main?)]
      [`(if ,pred ,t1 ,t2)
       (define then-label (loc->string (fresh-label 'then)))
       (define else-label (loc->string (fresh-label 'else)))
       (define pred-var (compile-pred pred emit! #f))
       (emit! (make-branch pred-var then-label else-label))
       (emit! (make-label then-label))
       (compile-tail t1 emit! main?)
       (emit! (make-label else-label))
       (compile-tail t2 emit! main?)]
      [value
       (cond 
         [main?
           (define result (compile-value value emit! #f))
           (define dest result)
           (define arg1 dest)
           (define arg2 (compile-opand 8 emit!))
           (emit! (make-value-op "div" dest "any" (list arg1 arg2)))
           (emit! (make-effect-op "print" (list dest)))]
         [else
           (define result (compile-value value emit! main?))
           (emit! (make-effect-op "ret" (list result)))])]))
  
  ;; Compile a value expression
  (define (compile-value value emit! main?)
    (match value
      [`(call ,triv ,opands ...)
       (compile-call triv opands emit! #f main?)]
      [`(,binop ,op1 ,op2) 
        (match binop 
          ['bitwise-and
           (compile-call 'bitand (list op1 op2) emit! #f main?)]
          ['bitwise-ior
           (compile-call 'bitor (list op1 op2) emit! #f main?)]
          ['bitwise-xor 
           (compile-call 'bitxor (list op1 op2) emit! #f main?)]
          ['arithmetic-shift-right 
           (compile-call 'ashr (list op1 op2) emit! #f main?)]
          [else 
            (define dest (loc->string (fresh 'tmp)))
            (define arg1 (compile-opand op1 emit!))
            (define arg2 (compile-opand op2 emit!))
            (emit! (make-value-op (binop->bril binop) dest "any" (list arg1 arg2)))
            dest])]
      [triv
       (compile-triv triv emit!)]))
  
  ;; Compile an effect
  (define (compile-effect effect emit! main?)
    (match effect
      [`(set! ,aloc ,val)
       (define result (compile-value val emit! main?))
       (emit! (make-value-op "id" (loc->string aloc) "any" (list result)))]
      [`(begin ,effects ...)
       (for ([eff effects])
         (compile-effect eff emit! main?))]
      [`(if ,pred ,e1 ,e2)
       (define then-label (loc->string (fresh-label 'then)))
       (define else-label (loc->string (fresh-label 'else)))
       (define end-label (loc->string (fresh-label 'endif)))
       (define pred-var (compile-pred pred emit! main?))
       (emit! (make-branch pred-var then-label else-label))
       (emit! (make-label then-label))
       (compile-effect e1 emit! main?)
       (emit! (make-jump end-label))
       (emit! (make-label else-label))
       (compile-effect e2 emit! main?)
       (emit! (make-label end-label))]))
  
  ;; Compile a predicate
  (define (compile-pred pred emit! main?)
    (match pred
      [`(true)
       (define dest (loc->string (fresh 'tmp)))
       (emit! (make-const dest "any" #t))
       dest]
      [`(false)
       (define dest (loc->string (fresh 'tmp)))
       (emit! (make-const dest "any" #f))
       dest]
      [`(not ,p)
       (define arg (compile-pred p emit! main?))
       (define dest (loc->string (fresh 'tmp)))
       (emit! (make-value-op "not" dest "any" (list arg)))
       dest]
      [`(begin ,effects ... ,p)
       (for ([eff effects])
         (compile-effect eff emit! main?))
       (compile-pred p emit! main?)]
      [`(if ,p ,p1 ,p2)
       (define result (loc->string (fresh 'tmp)))
       (define then-label (loc->string (fresh-label 'then)))
       (define else-label (loc->string (fresh-label 'else)))
       (define end-label (loc->string (fresh-label 'endif)))
       (define pred-var (compile-pred p emit! main?))
       (emit! (make-branch pred-var then-label else-label))
       (emit! (make-label then-label))
       (define res1 (compile-pred p1 emit! main?))
       (emit! (make-value-op "id" result "any" (list res1)))
       (emit! (make-jump end-label))
       (emit! (make-label else-label))
       (define res2 (compile-pred p2 emit! main?))
       (emit! (make-value-op "id" result "any" (list res2)))
       (emit! (make-label end-label))
       result]
      [`(,relop ,op1 ,op2)
        (match relop
          ['!= 
           (compile-pred `(not (= ,op1 ,op2)) emit! main?)]
          [else 
            (define dest (loc->string (fresh 'tmp))) 
            (define arg1 (compile-opand op1 emit!))
            (define arg2 (compile-opand op2 emit!))
            (emit! (make-value-op (relop->bril relop) dest "any" (list arg1 arg2)))
            dest])]))
  
  ;; Compile a function call
  (define (compile-call triv opands emit! tail? main?)
    (define args (map (λ (op) (compile-opand op emit!)) opands))
    (define func-name (loc->string triv))
    (if tail?
        (let ([dest (loc->string (fresh 'tmp))])
          (cond 
            [main?
              (emit! (make-call dest "any" func-name args))
              (define arg1 dest)
              (define arg2 (compile-opand 8 emit!)) ;; untagging
              (emit! (make-value-op "div" dest "any" (list arg1 arg2)))
              (emit! (make-effect-op "print" (list dest)))]
            [else
              (emit! (make-call dest "any" func-name args))
              (emit! (make-effect-op "ret" (list dest)))]))
        (let ([dest (loc->string (fresh 'tmp))])
          (emit! (make-call dest "any" func-name args))
          dest)))
  
  ;; Compile an operand
  (define (compile-opand opand emit!)
    (match opand
      [(? integer? n)
       (define dest (loc->string (fresh 'tmp)))
       (emit! (make-const dest "any" n))
       dest]
      [aloc
       (loc->string aloc)]))
  
  ;; Compile a triv
  (define (compile-triv triv emit!)
    (match triv
      [(? integer? n)
       (define dest (loc->string (fresh 'tmp)))
       (emit! (make-const dest "any" n))
       dest]
      [loc
       (loc->string loc)]))
  
  ;; Helper: create a constant instruction
  (define (make-const dest type value)
    (hasheq 'op "const" 'dest dest 'type type 'value value))
  
  ;; Helper: create a value operation
  (define (make-value-op op dest type args)
    (hasheq 'op op 'dest dest 'type type 'args args))
  
  ;; Helper: create an effect operation
  (define (make-effect-op op args)
    (hasheq 'op op 'args args))
  
  ;; Helper: create a call instruction
  (define (make-call dest type func args)
    (hasheq 'op "call" 'dest dest 'type type 'funcs (list func) 'args args))
  
  ;; Helper: create a label
  (define (make-label name)
    (hasheq 'label name))
  
  ;; Helper: create a jump
  (define (make-jump label)
    (hasheq 'op "jmp" 'labels (list label)))
  
  ;; Helper: create a branch
  (define (make-branch arg then-label else-label)
    (hasheq 'op "br" 'args (list arg) 'labels (list then-label else-label)))
  
  ;; Convert binop to Bril operation
  (define (binop->bril binop)
    (match binop
      ['+ "add"]
      ['- "sub"]
      ['* "mul"]
      [else (error "Unknown binop" binop)]))
  
  ;; Convert relop to Bril operation
  (define (relop->bril relop)
    (match relop
      ['< "lt"]
      ['<= "le"]
      ['= "eq"]
      ['>= "ge"]
      ['> "gt"]
      [else (error "Unknown relop" relop)]))
  
  (compile-program p))
