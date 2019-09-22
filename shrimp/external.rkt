#lang racket/base

(require racket/system
         racket/port
         racket/match
         racket/contract
         racket/file
         racket/function
         json
         "ast.rkt")
(provide input-bril
         show-graph)

(define/contract (input-bril filename)
  (-> string? jsexpr?)

  (match-define (list out-port input pid err-port info-proc)
    (process "bril2json"))
  (fprintf input "~a" (file->string filename))
  (close-output-port input)
  (info-proc 'wait)

  (define-values (out err)
    (values (port->string out-port)
            (port->string err-port)))
  (close-input-port out-port)
  (close-input-port err-port)

  (when (not (string=? "" err))
    (raise-result-error 'input-bril
                        "No error"
                        err))

  (string->jsexpr out))

(define (show-graph dot-string)
  (if (find-executable-path "xdot")
      (let ([tmp-file (make-temporary-file)])
        (with-output-to-file tmp-file
          #:exists 'replace
          (thunk (display dot-string)))
        (system (format "xdot ~a 2>/dev/null" tmp-file))
        (delete-file tmp-file))
      (with-output-to-file "out.dot"
        #:exists 'replace
        (thunk (display dot-string)))))
