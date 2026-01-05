#lang racket

;;Copyright (C) 2025 Joseph Maheshe.

(require
 json
 cpsc411/compiler-lib
 "uniquify.rkt"
 "implement-safe-primops.rkt"
 "specify-representation.rkt"
 "remove-complex-opera.rkt"
 "sequentialize-let.rkt"
 "normalize-bind.rkt"
 "compile-bril.rkt")

(provide
 bracket)

(define (bracket program)
  (define all-passes
    (list
      compile-bril
      normalize-bind
      sequentialize-let
      remove-complex-opera*
      specify-representation
      implement-safe-primops
      uniquify))

  (define pipeline
    (apply compose all-passes))
  (pipeline program))

(module+ main
  (define args (current-command-line-arguments))

  (define in
    (cond
      [(= (vector-length args) 0)
       (current-input-port)]
      [(= (vector-length args) 1)
       (open-input-file (vector-ref args 0))]
      [else
       (error 'bracket "expected at most one input file, got ~a"
              (vector-length args))]))

  (define program (read in))
  (write-json (bracket program) (current-output-port)))

