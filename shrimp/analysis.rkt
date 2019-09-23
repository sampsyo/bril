#lang racket/base

(require racket/match
         racket/set
         racket/pretty
         "ast.rkt"
         "cfg.rkt")

(provide live-ins)

(define (get-inps-and-out instr)
  (cond
    [(dest-instr? instr)
     (match-define (dest-instr dest _ vals) instr)
     (values vals dest)]
    [(branch? instr) (values (branch-con instr) #f)]
    [(print-val? instr) (values (print-val-v instr) #f)]
    [else (values '() #f)]))

;; Return the set of live-in variables to a basic block.
(define (live-ins block)

  (define live (mutable-set))
  (define defined (mutable-set))

  (for ([instr (basic-block-instrs block)])
    (define-values (inps out) (get-inps-and-out instr))

    (for ([inp inps])
      (unless (or (not (string? inp)) (set-member? defined inp))
        (set-add! live inp)))

    (when out (set-add! defined out)))

  (set->list live))

