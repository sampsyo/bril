#lang rosette

(require racket/match
         racket/function
         racket/hash
         graph
         racket/trace
         "helpers.rkt"
         "ast.rkt"
         "analysis.rkt"
         "cfg.rkt")

(provide interpret-block
         display-interp)

; Extract this out into a separate file
(define (empty-state) (make-hash))
(define state-key? string?)
(define state-has-key? hash-has-key?)

(define (state-ref state key)
  (if (state-key? key)
    (begin
      (when (not (state-has-key? state key))
        (raise-argument-error 'state-ref
                              "Variable in state"
                              (~a state " does not have " key)))
      (hash-ref state key))
    key))

(define (state-store state key val)
  (hash-set! state key val))
(define (state-merge! s1 s2)
  (hash-union! s1 s2
               #:combine (lambda (v1 _) v1)))

(define (dest-instr-to-func instr)
  (cond
    [(add? instr)  + ]
    [(sub? instr)  - ]
    [(mul? instr)  * ]
    [(div? instr)  / ]
    [(ieq? instr)  = ]
    [(lt? instr)   < ]
    [(gt? instr)   > ]
    [(le? instr)   <=]
    [(ge? instr)   >=]
    [(lnot? instr) not]
    [(land? instr) ((curry andmap) identity)]
    [(lor? instr)  ((curry ormap) identity)]
    [(constant? instr) identity]
    [(id? instr) identity]
    [else (raise-argument-error 'interpret-block
                                (~a "Unknown instruction: " instr ))]))

; Interpret a basic block with the given state. Returns a new state and the
; label for the next basic block to execute.
(define (interpret-block state block)

  (define (instr-to-func instr)
    (cond [(dest-instr? instr)
           (define func (dest-instr-to-func instr))
           (match-define (dest-instr dest _ vals) instr)
           (define args (map (lambda (arg) (state-ref state arg)) vals))
           (define res (apply func args))
           (state-store state dest res)]))

  (for ([instr (basic-block-instrs block)])
    (instr-to-func instr))

  state)

(define (display-interp state block)
  (displayln "Running: ")
  (pretty-print block)
  (displayln "with: ")
  (pretty-print state)
  (displayln "gives the final state: ")
  (pretty-print (interpret-block state block)))
