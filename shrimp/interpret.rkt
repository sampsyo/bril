#lang rosette

(require racket/match
         racket/function
         racket/hash
         graph
         "ast.rkt"
         "analysis.rkt"
         "cfg.rkt")

(provide interpret)

; Extract this out into a separate file
(define (empty-state) (make-hash))
(define state-key? string?)
(define (state-ref state key)
  (if (state-key? key) (hash-ref state key) key))
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

(define (make-symbolic-unknown-type name)
  (define-symbolic* is-bool? boolean?)
  (define-symbolic* b boolean?)
  (define-symbolic* i integer?)
  (if is-bool? b i))

(define (interpret-block state block)

  (define (instr-to-func instr)
    (cond [(dest-instr? instr)
           (define func (dest-instr-to-func instr))
           (match-define (dest-instr dest _ vals) instr)
           (define args (map (lambda (arg) (state-ref state arg)) vals))
           (define res (apply func args))
           (state-store state dest res)]))

  (define sym-live-ins
    (make-hash
      (map (lambda (i) (cons i (make-symbolic-unknown-type i)))
           (live-ins block))))

  (state-merge! state sym-live-ins)

  (for ([inst (basic-block-instrs block)])
    (instr-to-func inst))

  (pretty-print state))

(define (interpret cfgs)
  (for-each (match-lambda
              [(cons bs g)
               (map ((curry interpret-block) (empty-state)) bs)])
            cfgs))

