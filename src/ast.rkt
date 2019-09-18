#lang racket/base

(require json
         racket/match
         racket/system
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
(struct binop dest-intr () #:transparent)

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
(struct lnot dest-intsr () #:transparent)
(struct land binop () #:transparent)
(struct lor binop () #:transparent)

(struct constant dest-instr () #:transparent)
(struct id destr-instr () #:transparent)

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
       ["add" (add dest type vals)]
       ["sub" (add dest type vals)]
       ["mul" (mul dest type vals)]
       ["div" (div dest type vals)]

       ;; comparison
       ["eq" (ieq dest type vals)]
       ["lt" (lt dest type vals)]
       ["gt" (gt dest type vals)]
       ["le" (le dest type vals)]
       ["ge" (ge dest type vals)]

       ;; logic
       ["not" (lnot dest type vals)]
       ["and" (land dest type vals)]
       ["or" (lor dest type vals)]

       ["const" (constant dest type vals)]
       ["id" (id dest type vals)]

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
