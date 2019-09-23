#lang rosette

(require "ast.rkt"
         "analysis.rkt"
         "interpret.rkt"
         "cfg.rkt"
         racket/hash)

(provide verify-block
         verify-prog)

;; (define (make-symbolic-unknown-type name)
;;   (define-symbolic* is-bool? boolean?)
;;   (define-symbolic* b boolean?)
;;   (define-symbolic* i integer?)
;;   (if is-bool? b i))

; Create symbolic variables for live-in variables and interpret them.
; Return the resulting state
(define (symbolic-interpret context block)

  (define symbolic-live
    (make-hash
     ;; for each live in, create a symbolic variable of the correc type
     (for/list ([var (live-ins block)])
       (define type (hash-ref context var))
       (define sym
         (cond [(int? type)
                (define-symbolic* i integer?)
                i]
               [(bool? type)
                (define-symbolic* b boolean?)
                b]))
       (cons var sym))))

  (interpret-block symbolic-live block))

(define (verify-block block1 context1 block2 context2)
  (define state1 (symbolic-interpret context1 block1))
  (define state2 (symbolic-interpret context2 block2))

  ;; merge state2 into state1
  (hash-union! state1 state2
               #:combine (lambda (v1 v2) (cons v1 v2)))

  ;; (println (basic-block-label block1))
  ;; (pretty-print state1)

  (verify
    (for ([st state1])
      (match-define (cons k v)
        (when (pair? v)
          (assert (= (car v) (cdr v)))))
      (void) ;; XXX(sam) do something better to get rid of define being the last expression
      )))

(define (verify-prog) (void))
