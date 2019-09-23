#lang rosette

(require "ast.rkt"
         "analysis.rkt"
         "interpret.rkt"
         "cfg.rkt"
         "helpers.rkt"
         racket/hash)

(provide verify-block
         verify-prog)


(define (generate-sym-live context block)
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
    (cons var sym)))

(define (same-block? block1 lives1 block2 lives2)
  (define state1 (interpret-block (make-hash lives1) block1))
  (define state2 (interpret-block (make-hash lives2) block2))

  ;; merge state2 into state1
  (hash-union! state1 state2
               #:combine (lambda (v1 v2) (cons v1 v2)))

  (for ([val (hash-values state1)])
    (when (pair? val)
      (assert (equal? (car val) (cdr val)))))

  (pr `(ASSERTS: ,(asserts))))


(define (verify-block block1 context1 block2 context2)
  (pr (basic-block-label block1))

  (define-values (lives1 lives2)
    (values (generate-sym-live context1 block1)
            (generate-sym-live context2 block2)))

  (define common-lives
    (hash-union (make-immutable-hash lives1)
                (make-immutable-hash lives2)
                #:combine (lambda (v1 v2) (cons v1 v2))))

  (define-values (sol _)
    (with-asserts
      (begin
        (hash-for-each common-lives
                       (lambda (_ v)
                         (when (pair? v)
                           (assert (equal? (car v) (cdr v))))))

        (verify (same-block? block1 lives1 block2 lives2)))))
  (pr sol)
  sol)

(define (verify-prog) (void))
