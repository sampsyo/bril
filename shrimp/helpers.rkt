#lang rosette

(provide pr debug?)

(define debug? (make-parameter #f))

(define (pr v)
  (when (debug?)
    (pretty-print v))
  v)
