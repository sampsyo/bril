#lang racket/base

(require json
         racket/match
         racket/system
         racket/file)

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

;; math
(struct add (dest type v0 v1) #:transparent)
(struct sub (dest type v0 v1) #:transparent)
(struct mul (dest type v0 v1) #:transparent)
(struct div (dest type v0 v1) #:transparent)

;; comparison
(struct eq (dest type v0 v1) #:transparent)
(struct lt (dest type v0 v1) #:transparent)
(struct gt (dest type v0 v1) #:transparent)
(struct le (dest type v0 v1) #:transparent)
(struct ge (dest type v0 v1) #:transparent)

;; logic
(struct lnot (dest type v) #:transparent)
(struct land (dest type v0 v1) #:transparent)
(struct lor (dest type v0 v1) #:transparent)

;; control
(struct jump (label) #:transparent)
(struct branch (con tbranch fbranch) #:transparent)
(struct return () #:transparent)

(struct constant (dest type v) #:transparent)
(struct print-val (v) #:transparent)
(struct id (dest type v) #:transparent)
(struct nop () #:transparent)

(define (input-json filename)
  (with-input-from-file filename
    (lambda () (read-json))))

(define (json->ast instr)
  (cond
    [(hash-has-key? instr 'functions)
     (map (lambda (func)
            (function (hash-ref func 'name)
                      (map json->ast (hash-ref func 'instrs))))
          (hash-ref instr 'functions))]
    [(hash-has-key? instr 'label)
     (label (hash-ref instr 'label))]
    [(hash-has-key? instr 'op)
     (define dest (hash-ref instr 'dest #f))
     (define type (let ([v (hash-ref instr 'type #f)])
                    (if v (string->type v) #f)))
     (match (hash-ref instr 'op)
       ;; math instructions
       ["add" (match (hash-ref instr 'args)
                [(list v0 v1) (add dest type v0 v1)])]
       ["sub" (match (hash-ref instr 'args)
                [(list v0 v1) (sub dest type v0 v1)])]
       ["mul" (match (hash-ref instr 'args)
                [(list v0 v1) (mul dest type v0 v1)])]
       ["div" (match (hash-ref instr 'args)
                [(list v0 v1) (div dest type v0 v1)])]

       ;; comparison
       ["eq" (match (hash-ref instr 'args)
               [(list v0 v1) (eq dest type v0 v1)])]
       ["lt" (match (hash-ref instr 'args)
               [(list v0 v1) (lt dest type v0 v1)])]
       ["gt" (match (hash-ref instr 'args)
               [(list v0 v1) (gt dest type v0 v1)])]
       ["le" (match (hash-ref instr 'args)
               [(list v0 v1) (le dest type v0 v1)])]
       ["ge" (match (hash-ref instr 'args)
               [(list v0 v1) (ge dest type v0 v1)])]

       ;; logic
       ["not" (match (hash-ref instr 'args)
                [(list v) (lnot dest type v)])]
       ["and" (match (hash-ref instr 'args)
                [(list v0 v1) (land dest type v0 v1)])]
       ["or" (match (hash-ref instr 'args)
               [(list v0 v1) (lor dest type v0 v1)])]

       ;; control
       ["jmp" (jump (car (hash-ref instr 'args)))]
       ["br" (match (hash-ref instr 'args)
               [(list c t f) (branch c t f)])]
       ["ret" (return)]

       ["const" (constant dest type (hash-ref instr 'value))]
       ["print" (print-val (car (hash-ref instr 'args)))]
       ["id" (match (hash-ref instr 'args)
               [(list v) (id dest type v)])]
       ["nop" (nop)]

       [_ (error 'json-instr->ast "Don't have a case for ~v" instr)])]
    [else
     (error 'json-instr->ast "Don't have a case for ~v" instr)]))
