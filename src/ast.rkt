#lang racket/base

(require json
         racket/match
         racket/system
         racket/file)

(provide (all-defined-out))

;; instructions
(struct function (name instrs) #:transparent)
(struct label (name) #:transparent)

(struct constant (dest type value) #:transparent)

(struct jump (label) #:transparent)
(struct branch (con tbranch fbranch) #:transparent)
(struct return () #:transparent)

(struct print-val (var) #:transparent)

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
     (match (hash-ref instr 'op)
       ["const" (constant (hash-ref instr 'dest)
                          (hash-ref instr 'type)
                          (hash-ref instr 'value))]
       ["jmp" (jump (car (hash-ref instr 'args)))]
       ["br" (match (hash-ref instr 'args)
               [(list c t f) (branch c t f)])]
       ["print" (print-val (car (hash-ref instr 'args)))]
       ["ret" (return)]
       [_ (error 'json-instr->ast "Don't have a case for ~v" instr)])]
    [else
     (error 'json-instr->ast "Don't have a case for ~v" instr)]))
