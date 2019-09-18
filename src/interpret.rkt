#lang rosette

(require racket/match
         racket/function
         graph
         "ast.rkt"
         "cfg.rkt")

(provide interpret)

(define (interpret-block state block)
  (define (f instr)
    (cond instr
          [dest-instr?
           (match-define (dest-instr dest _ vals) instr)
           (define syms (apply ((curry state-ref) state)
                               vals))
           (cond instr
             [add? (apply + syms)]
             [sub? (apply - syms)]
             [mul? (apply * syms)]
             [div? (apply / syms)]
             [ieq? (apply = syms)]
             [lt? (apply < syms)]
             [gt? (apply > syms)]
             [le? (apply <= syms)]
             [ge? (apply >= syms)]
             [lnot? (apply not syms)]
             [land? (andmap (lambda (x) x) syms)]
             [lor? (ormap (lambda (x) x) syms)]
             [constant? (car vals)]
             [id? (car syms)])]
          [control?
           ]
          )
    )
  (foldl ()
         acc
         (basic-block-intrs block)
         )
  )

(define (interpret cfgs)
  (map (match-lambda
         [(cons bs g)
          (map interpret-block bs)])
       cfgs))

