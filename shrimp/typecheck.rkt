#lang racket/base

(require racket/hash
         racket/pretty
         racket/match
         "ast.rkt")

(provide typecheck)

(define (typecheck-assert exp1 exp2)
  (unless (equal? exp1 exp2)
    (error 'typecheck "Got ~a but expected ~a" exp1 exp2)))

;; (typecheck instrs) returns a map mapping variables to types
(define (typecheck instrs)
  (for/fold ([context (hash)])  ;; init accum
            ([instr instrs])
    (cond [(dest-instr? instr)
           (match-define (dest-instr dest type vals) instr)
           (if (hash-has-key? context dest)
               (begin
                 (typecheck-assert (hash-ref context dest) type)
                 context)
               (let-values
                   ([(arg-type res-type)
                     (cond
                       ;; (int, int) -> int instrs
                       [(or (add? instr)
                            (sub? instr)
                            (mul? instr)
                            (div? instr))
                        (values (int) (int))]

                       ;; (int, int) -> bool instrs
                       [(or (ieq? instr)
                            (lt? instr)
                            (gt? instr)
                            (le? instr)
                            (ge? instr))
                        (values (int) (bool))]

                       ;; (bool ...) -> bool instrs
                       [(or (lnot? instr)
                            (land? instr)
                            (lor? instr))
                        (values (bool) (bool))]

                       ;; (int | bool) literal -> (int | bool)
                       [(constant? instr)
                        (cond [(int? type)
                               (values integer? (int))]
                              [(bool? type)
                               (values boolean? (bool))])]

                       ;; t -> t
                       [(id? instr)
                        (values type type)])])
                 ;; check arg types
                 (for ([v vals])
                   (if (string? v)
                       (typecheck-assert (hash-ref context v) arg-type)
                       (arg-type v)))
                 ;; check result type
                 (typecheck-assert type res-type)

                 ;; update the context with the type of dest
                 (hash-set context dest type)))]

          [(branch? instr)
           (match-define (branch con _ _) instr)
           (typecheck-assert (hash-ref context con) (bool))
           context]

          [else context])))
