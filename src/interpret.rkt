#lang rosette

(require racket/match
         racket/function
         graph
         "ast.rkt"
         "cfg.rkt")

(provide interpret)

(define (state-ref . other) (void))

(define (interpret-block state block)
  (define (f instr)
    (cond [(dest-instr? instr)
           (match-define (dest-instr dest _ vals) instr)
           (define syms (apply ((curry state-ref) state)
                               vals))
           (cond
             [(add? instr) (apply + syms)]
             [(sub? instr) (apply - syms)]
             [(mul? instr) (apply * syms)]
             [(div? instr) (apply / syms)]
             [(ieq? instr) (apply = syms)]
             [(lt? instr) (apply < syms)]
             [(gt? instr) (apply > syms)]
             [(le? instr) (apply <= syms)]
             [(ge? instr) (apply >= syms)]
             [(lnot? instr) (apply not syms)]
             [(land? instr) (andmap (lambda (x) x) syms)]
             [(lor? instr) (ormap (lambda (x) x) syms)]
             [(constant? instr) (car vals)]
             [(id? instr) (car syms)])]
          ;; [control?
          ;;  ]
          )
    )
  (println "nyi")
  ;; (foldl ()
  ;;        acc
  ;;        (basic-block-intrs block)
  ;;        )
  )

(define (interpret cfgs)
  (map (match-lambda
         [(cons bs g)
          (map interpret-block bs)])
       cfgs))

