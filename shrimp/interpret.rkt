#lang rosette

(require racket/match
         racket/function
         graph
         "ast.rkt"
         "cfg.rkt")

(provide interpret)

; Extract this out into a separate file
(define (empty-state) (make-hash))
(define state-key? string?)
(define (state-ref state key)
  (if (state-key? key) (hash-ref state key) key))
(define (state-store state key val)
  (hash-set! state key val))

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


(define (interpret-block state block)
  (define (instr-to-func instr)
    (cond [(dest-instr? instr)
           (define func (dest-instr-to-func instr))
           (match-define (dest-instr dest _ vals) instr)
           (define args (map (lambda (arg) (state-ref state arg)) vals))
           (define res (apply func args))
           (state-store state dest res)]))

  (for ([inst (basic-block-instrs block)])
    (instr-to-func inst))

  (pretty-print state))

(define (interpret cfgs)
  (map (match-lambda
         [(cons bs g)
          (map ((curry interpret-block) (empty-state)) bs)])
       cfgs))

