#lang rosette

(require "ast.rkt"
         "analysis.rkt"
         "cfg.rkt")

(provide verify-block
         verify-prog)

(define (make-symbolic-unknown-type name)
  (define-symbolic* is-bool? boolean?)
  (define-symbolic* b boolean?)
  (define-symbolic* i integer?)
  (if is-bool? b i))

; Create symbolic variables for live-in variables and interpret them.
; Return the resulting state
(define (symbolic-interpret block)

  (define symbolic-live
    (make-hash
      (map (lambda (i) (cons i (make-symbolic-unknown-type i)))
           (live-ins block))))

  (interpret-block symbolic-live block))

(define (verify-block block1 block2)

  (define state1 (symbolic-interpret block1))
  (define state2 (symbolic-interpret block2))
  (define merged (hash->list
                   (hash-union state1 state2
                               #:combine (lambda (v1 v2) (cons v1 v2)))))

  (verify
    (for ([st merged])
      (match-define (cons k v)
        (when (pair? v)
          (assert (= (car v) (cdr v))))))))
