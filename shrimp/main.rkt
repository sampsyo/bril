#lang racket/base

(require racket/cmdline
         racket/match
         racket/format
         racket/pretty
         graph
         threading
         "external.rkt"
         "ast.rkt"
         "cfg.rkt"
         "typecheck.rkt"
         "interpret.rkt"
         "verify.rkt"
         "lvn.rkt")

(define convert-bril (make-parameter #f))
(define show-cfg (make-parameter #f))
(define output-blocks (make-parameter #f))
(define disable-typecheck (make-parameter #f))
(define run-interpreter (make-parameter #f))
(define run-verify (make-parameter #f))
(define lvn (make-parameter #f))

(define ((apply-when f predicate) val)
  (if predicate
      (f val)
      val))

(define (cfg-passes ast)
  (~> ast
      (map function-instrs _)
      (map cfg _)
      (map (apply-when local-value-numbering (lvn)) _)
      ))

(define (process-file filename)
  ;; check to make sure file exists
  (when (not (file-exists? filename))
    (raise-argument-error 'main
                          (~a filename " does not exist")))

  ;; read in ast, converting if necessary
  (define ast
    (json->ast
     (if (convert-bril)
         (input-bril filename)
         (input-json filename))))

  ;; typecheck
  (unless (disable-typecheck)
    (for ([func ast])
      (typecheck (function-instrs func))))

  ;; cfg
  (define cfg (cfg-passes ast))

  (values ast cfg))

(define (main filename)

  (define-values (ast cfgs) (process-file filename))

  (when (output-blocks)
    (for-each (match-lambda
                [(cons blocks _)
                 (pretty-print blocks)])
              cfgs))

  (when (show-cfg)
    (for-each (match-lambda
                [(cons blocks graph)
                 (show-graph (graphviz graph))])
              cfgs))

  ;; XXX(rachit): THIS IS WRONG. interpret block works for basic blocks.
  (when (run-interpreter)
    (println "not hooked up")
    ;; (pretty-print (interpret-block cfgs-p))
    )

  (when (run-verify)
    (define-values (compare-ast compare-cfgs) (process-file (run-verify)))
    (for ([cfg cfgs]
          [compare-cfg compare-cfgs])
      (for ([block (hash-values (car cfg))]
            [compare-block (hash-values (car compare-cfg))])
        (println "block:")
        (pretty-print block)
        (println "compare against:")
        (pretty-print compare-block)
        (verify-block block compare-block)))))

(module+ main
  (command-line
   #:program "shrimp"
   #:once-each
   [("-b" "--bril") "Convert <filename> to json before processing"
                    (convert-bril #t)]
   [("--blocks") "Output representation of basic blocks"
                 (output-blocks #t)]
   [("-p" "--plot") "Show the cfgs using xdot"
                    (show-cfg #t)]
   [("-T" "--disable-typecheck") "Disable type checking"
                            (disable-typecheck #t)]
   [("-i" "--interpret") "Interpret the given files"
                       (run-interpreter #t)]
   [("-v" "--verify") compare-file ; verify takes in an argument
                      "Verify that <filename> and <compare-file> are functionally equivalent."
                      "Assumes that they are in the same format"
                      (run-verify compare-file)]
   [("--lvn") "Enable lvn optimizations"
              (lvn #t)]
   #:args (filename)
   (main filename)))

(module+ test
  (parameterize ([convert-bril #t]
                 [run-interpreter #f]
                 [output-blocks #t]
                 [disable-typecheck #t]
                 [run-verify "../test/ts/cond.out"]
                 )
    (main "../test/ts/loopfact.out")))
