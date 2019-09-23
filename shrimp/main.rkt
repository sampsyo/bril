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
         "lvn.rkt")

(define convert-bril (make-parameter #f))
(define show-cfg (make-parameter #f))
(define output-blocks (make-parameter #f))
(define disable-typecheck (make-parameter #f))
(define run-interpreter (make-parameter #f))
(define lvn (make-parameter #f))

(define (main filename)
  (when (not (file-exists? filename))
    (raise-argument-error 'main
                          (~a filename " does not exist")))

  (define ast
    (json->ast
     (if (convert-bril)
         (input-bril filename)
         (input-json filename))))

  (unless (disable-typecheck)
    (for-each (lambda (instrs) (typecheck instrs))
              (map function-instrs ast)))

  (define cfgs
    (~> ast
        (map function-instrs _)
        (map cfg _)))

  (define cfgs-p
    (if (lvn)
        (map (match-lambda
               [(cons blocks graph)
                (cons (map local-value-numbering blocks) graph)])
             cfgs)
        cfgs))

  (when (output-blocks)
    (for-each (match-lambda
                [(cons blocks _)
                 (pretty-print blocks)])
              cfgs-p))

  (when (show-cfg)
    (for-each (match-lambda
                [(cons blocks graph)
                 (show-graph (graphviz graph))])
              cfgs-p))

  ;; XXX(rachit): THIS IS WRONG. interpret block works for basic blocks.
  (when (run-interpreter)
    (pretty-print (interpret-block cfgs-p))))

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
   [("--lvn") "Enable lvn optimizations"
              (lvn #t)]
   #:args (filename)
   (main filename)))

(module+ test
  (parameterize ([convert-bril #t]
                 [run-interpreter #f]
                 [output-blocks #t]
                 [disable-typecheck #t])
    (main "../test/ts/loopfact.out")))
