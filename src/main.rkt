#lang racket/base

(require racket/cmdline
         racket/match
         racket/pretty
         graph
         threading
         "external.rkt"
         "ast.rkt"
         "cfg.rkt"
         "interpret.rkt")

(define convert-bril (make-parameter #f))
(define show-cfg (make-parameter #f))
(define output-blocks (make-parameter #f))
(define disable-typecheck (make-parameter #f))

(define (main filename)
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

  (when (output-blocks)
    (for-each (match-lambda
                [(cons blocks _)
                 (pretty-print blocks)])
              cfgs))

  (when (show-cfg)
    (for-each (match-lambda
                [(cons blocks graph)
                 (show-graph (graphviz graph))])
              cfgs)))

(command-line
 #:program "shrimp"
 #:once-each
 [("-b" "--bril") "Convert <filename> to json before processing"
                  (convert-bril #t)]
 [("--blocks") "Output representation of basic blocks"
               (output-blocks #t)]
 [("-p" "--plot") "Show the cfgs using xdot"
                  (show-cfg #t)]
 [("--disable-typecheck") "Disable type checking"
                          (disable-typecheck #t)]
 #:args (filename)
 (main filename))
