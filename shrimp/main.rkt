#lang racket/base

(require racket/cmdline
         racket/match
         racket/format
         racket/pretty
         graph
         threading
         "helpers.rkt"
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
  ;; XXX(sam) I don't really like this being a list. mabye a hash would be better
  (define type-contexts
    (for/list ([func ast])
      (typecheck (function-instrs func))))

  ;; do cfg passes
  (define cfg (cfg-passes ast))

  (values ast cfg type-contexts))

(define (main filename)

  (define-values (ast cfgs contexts) (process-file filename))

  (when (output-blocks)
    (for-each (match-lambda
                [(cons blocks _)
                 (pr blocks)])
              cfgs))

  (when (show-cfg)
    (for-each (match-lambda
                [(cons blocks graph)
                 (show-graph (graphviz graph))])
              cfgs))

  ;; XXX(rachit): Hook up the interpreter
  (when (run-interpreter)
    (println "not hooked up"))

  (when (run-verify)
    (define-values (compare-ast compare-cfgs compare-contexts) (process-file (run-verify)))
    (for ([cfg cfgs]
          [compare-cfg compare-cfgs]
          [context contexts]
          [compare-context compare-contexts])
      (for ([block-lbl (hash-keys (car cfg))])
        (define-values (block compare-block)
          (values (hash-ref (car cfg) block-lbl)
                  (hash-ref (car compare-cfg) block-lbl)))
        (unless context
          (error 'verify "Verification requires the typechecker to be run"))
        (verify-block block context compare-block compare-context)))))

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
                 [output-blocks #f]
                 [run-verify "../test/ts/loopfact.out"]
                 )
    (main "../test/ts/loopfact.out")))
