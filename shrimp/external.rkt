#lang racket/base

(require racket/system
         racket/match
         racket/file
         racket/function
         json
         "ast.rkt")
(provide input-bril
         show-graph)

(define (input-bril filename)
  ;; (define in (open-input-file filename))
  ;; (define out (open-output-string))
  (match-define (list output input pid err info-proc) (process "bril2json"))

  (fprintf input "~a" (file->string filename))
  (close-output-port input)

  (info-proc 'wait)

  (define res (read-json output))

  (close-input-port output)
  (close-input-port err)
  res)

(define (show-graph dot-string)
  (if (find-executable-path "xdot")
      (let ([tmp-file (make-temporary-file)])
        (with-output-to-file tmp-file
          #:exists 'replace
          (thunk (display dot-string)))
        (system (format "xdot ~a 2>/dev/null" tmp-file))
        (delete-file tmp-file))
      (with-output-to-file "out.dot"
        (thunk (print dot-string)))))
