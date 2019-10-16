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

(struct dest-instr (dest type vals)
  #:transparent
  #:guard (lambda (dest type vals _)
            (unless (string? dest)
              (error 'dest-instr (~a "dest: " dest " is not a string")))
            (unless (list? vals)
              (error 'dest-instr (~a "vals: " vals " is not a list")))
            (for-each
              (lambda (val)
                (when (not (or (string? val) (number? val) (boolean? val)))
                  (error 'dest-instr (~a val " is not a string"))))
              vals)
            (values dest type vals)))

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

     ;; If the op is a constant, it has value instead of 'args
     (define vals
       (let [(val (or (hash-ref instr-map 'args #f)
                      (hash-ref instr-map 'value '())))]
         (if (list? val) val (list val))))

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

(define (instr-constr instr)
  (cond [(function? instr) function]
        [(label? instr) label]
        [(add? instr) add]
        [(sub? instr) sub]
        [(mul? instr) mul]
        [(div? instr) div]
        [(ieq? instr) ieq]
        [(lt? instr) lt]
        [(gt? instr) gt]
        [(le? instr) le]
        [(ge? instr) ge]
        [(lnot? instr) lnot]
        [(land? instr) land]
        [(lor? instr) lor]
        [(constant? instr) constant]
        [(id? instr) id]
        [(jump? instr) jump]
        [(branch? instr) branch]
        [(return? instr) return]
        [(print-val? instr) print-val]
        [(nop? instr) nop])
  )
