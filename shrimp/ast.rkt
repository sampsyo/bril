#lang racket/base

(require json
         racket/match
         racket/format
         racket/system
         racket/set
         racket/function
         racket/file
         racket/generic)

(provide (all-defined-out))


;; ---- types ----
(struct int () #:transparent)
(struct bool () #:transparent)
(define (string->type s)
  (match s
    ["int" (int)]
    ["bool" (bool)]))

;; ---- instructions -----
(struct function (name instrs) #:transparent)
(struct label (name) #:transparent)

(struct dest-instr (dest type vals) #:transparent)
(struct binop dest-instr () #:transparent)

;; math
(struct add binop () #:transparent)
(struct sub binop () #:transparent)
(struct mul binop () #:transparent)
(struct div binop () #:transparent)

;; comparison
(struct ieq binop () #:transparent)
(struct lt binop () #:transparent)
(struct gt binop () #:transparent)
(struct le binop () #:transparent)
(struct ge binop () #:transparent)

;; logic
(struct lnot dest-instr () #:transparent)
(struct land binop () #:transparent)
(struct lor binop () #:transparent)

(struct constant dest-instr () #:transparent)
(struct id dest-instr () #:transparent)

(struct control () #:transparent)

;; control
(struct jump control (label) #:transparent)
(struct branch control (con tbranch fbranch) #:transparent)
(struct return control () #:transparent)

(struct print-val (v) #:transparent)
(struct nop () #:transparent)

(define (input-json filename)
  (with-input-from-file filename
    (lambda () (read-json))))

; Get function associated with a op name.
(define (op-name->ast name)
  (match name
    ["add" add]
    ["sub" sub]
    ["mul" mul]
    ["div" div]
    ["eq" ieq]
    ["lt" lt]
    ["gt" gt]
    ["le" le]
    ["ge" ge]
    ["not" lnot]
    ["and" land]
    ["or" lor]
    ["const" constant]
    ["id" id]
    [else (raise-argument-error 'op-name->ast (~a "Unknown op " name))]))

; Ops known in Bril. Should be exactly the domain of the op-name->ast.
(define valid-ops
  (set "add" "sub" "mul" "div" "eq" "lt" "gt" "le" "ge" "not" "and" "or" "const" "id"))

(define (json->ast instr-map)
  (cond
    [(hash-has-key? instr-map 'functions)
     (map (lambda (func)
            (function (hash-ref func 'name)
                      (map json->ast (hash-ref func 'instrs))))
          (hash-ref instr-map 'functions))]
    [(hash-has-key? instr-map 'label)
     (label (hash-ref instr-map 'label))]
    [(hash-has-key? instr-map 'op)
     (define dest (hash-ref instr-map 'dest #f))
     (define type (let ([v (hash-ref instr-map 'type #f)])
                    (if v (string->type v) #f)))
     (define vals (hash-ref instr-map 'args '()))
     (match (hash-ref instr-map 'op)
       ;; math instr-mapuctions
       [(? ((curry set-member?) valid-ops) op) ((op-name->ast op) dest type vals)]

       ;; control
       ["jmp" (jump (car vals))]
       ["br" (match vals
               [(list c t f) (branch c t f)])]
       ["ret" (return)]

       ["print" (print-val (car vals))]
       ["nop" (nop)]

       [_ (error 'json-instr-map->ast "Don't have a case for ~v" instr-map)])]
    [else
     (error 'json-instr-map->ast "Don't have a case for ~v" instr-map)]))
