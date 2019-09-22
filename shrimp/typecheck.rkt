#lang racket/base

(require racket/hash
         racket/pretty
         racket/match
         "ast.rkt")

(provide typecheck)

(define ((check! acc desired) val)
  (define type (hash-ref acc val))
  (unless (equal? type desired)
    (error 'typecheck "~a was ~a, but expected ~a"
           val type desired)))

(define (typecheck instrs)
  instrs
  ;; (foldl (lambda (x acc)
  ;;          (match x
  ;;            [(label _) acc]
  ;;            ;; arithmetic
  ;;            [(dest-instr (add) dest vals)
  ;;             (apply (check! acc (int)) vals)
  ;;             (hash-set acc dest (int))]
  ;;            [(sub dest (int) v0 v1)
  ;;             (check! acc v0 (int))
  ;;             (check! acc v1 (int))
  ;;             (hash-set acc dest (int))]
  ;;            [(mul dest (int) v0 v1)
  ;;             (check! acc v0 (int))
  ;;             (check! acc v1 (int))
  ;;             (hash-set acc dest (int))]
  ;;            [(div dest (int) v0 v1)
  ;;             (check! acc v0 (int))
  ;;             (check! acc v1 (int))
  ;;             (hash-set acc dest (int))]

  ;;            ;; comparison
  ;;            [(eq dest (bool) v0 v1)
  ;;             (check! acc v0 (int))
  ;;             (check! acc v1 (int))
  ;;             (hash-set acc dest (bool))]
  ;;            [(lt dest (bool) v0 v1)
  ;;             (check! acc v0 (int))
  ;;             (check! acc v1 (int))
  ;;             (hash-set acc dest (bool))]
  ;;            [(gt dest (bool) v0 v1)
  ;;             (check! acc v0 (int))
  ;;             (check! acc v1 (int))
  ;;             (hash-set acc dest (bool))]
  ;;            [(le dest (bool) v0 v1)
  ;;             (check! acc v0 (int))
  ;;             (check! acc v1 (int))
  ;;             (hash-set acc dest (bool))]
  ;;            [(ge dest (bool) v0 v1)
  ;;             (check! acc v0 (int))
  ;;             (check! acc v1 (int))
  ;;             (hash-set acc dest (bool))]

  ;;            ;; logic
  ;;            [(lnot dest (bool) v)
  ;;             (check! acc v (bool))
  ;;             (hash-set acc dest (bool))]
  ;;            [(land dest (bool) v0 v1)
  ;;             (check! acc v0 (bool))
  ;;             (check! acc v1 (bool))
  ;;             (hash-set acc dest (bool))]
  ;;            [(lor dest (bool) v0 v1)
  ;;             (check! acc v0 (bool))
  ;;             (check! acc v1 (bool))
  ;;             (hash-set acc dest (bool))]

  ;;            ;; control
  ;;            [(jump _) acc]
  ;;            [(branch con _ _)
  ;;             (check! acc con (bool))
  ;;             acc]
  ;;            [(return) acc]

  ;;            ;; other
  ;;            [(constant dest (bool) v)
  ;;             (unless (boolean? v)
  ;;               (error 'typecheck "~a is not a boolean" v))
  ;;             (hash-set acc dest (bool))]
  ;;            [(constant dest (int) v)
  ;;             (unless (integer? v)
  ;;               (error 'typecheck "~a is not an integer" v))
  ;;             (hash-set acc dest (int))]
  ;;            [(print-val _) acc]
  ;;            [(id dest type v)
  ;;             (check! acc v type)
  ;;             (hash-set acc dest type)]
  ;;            [(nop) acc]))
  ;;        (make-immutable-hash)
  ;;        instrs)
  )

;; (define (eval ))
