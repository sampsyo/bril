#lang racket

;; Copyright (C) 2023 Colby Sparks, Joseph Maheshe, Silviu Toderita.

(require
  cpsc411/compiler-lib
  "util.rkt")

(provide
 remove-complex-opera*)

;; exprs-bits-lang-v7? -> values-bits-lang-v7?
(define (remove-complex-opera* p)
  (define (rco-program p)
    (match p
      [`(module (define ,labels (lambda ,paramss ,values)) ... ,value)
       `(module ,@(rco-defines labels paramss values) ,(rco-tail value))]))

  ;; intentionally omitted:
  ;; value -> rco-triv

  ;; procedure -> procedure
  ;; compiles exprs-unsafe-data-lang-v7 procedure to values-bits-lang-v7 procedure
  (define (rco-defines labels paramss values)
    (for/fold ([defines-acc '()])
              ([label (reverse labels)]
               [params (reverse paramss)]
               [value (reverse values)])
      (cons `(define ,label (lambda ,params ,(rco-tail value))) defines-acc)))

  ;; value -> tail
  ;; operates on value which is in tail position of target language
  (define (rco-tail value)
    (match value
      [(? triv?) value]
      [`(,binop ,v1 ,v2)
       #:when (binop? binop)
       (generate-op-bindings binop v1 v2)]
      [`(call ,v ,vs ...)
       (generate-call-bindings v vs)]
      [`(let ([,as ,vs] ...) ,v)
       (generate-let-bindings as vs (rco-tail v))]
      [`(if ,pred ,v1 ,v2)
       `(if ,(rco-pred pred) ,(rco-tail v1) ,(rco-tail v2))]))

  ;; pred -> pred
  (define (rco-pred pred)
    (match pred
      [`(,relop ,v1 ,v2)
       #:when (relop? relop)
       (generate-op-bindings relop v1 v2)]
      [`(true) pred]
      [`(false) pred]
      [`(not ,pred) `(not ,(rco-pred pred))]
      [`(let ([,as ,vs] ...) ,p)
       (generate-let-bindings as vs (rco-pred p))]
      [`(if ,p1 ,p2 ,p3)
       `(if ,(rco-pred p1) ,(rco-pred p2) ,(rco-pred p3))]))

  ;; value -> opand (value -> value)
  ;; operates on value in opand position of target language
  (define (rco-opand value)
    (match value
      [(? triv?) (values value identity)]
      [`(,binop ,v1 ,v2)
       #:when (binop? binop)
       (define var (fresh))
       (define binop-binding (generate-op-bindings binop v1 v2))
       (values var (λ (context)
                     `(let ([,var ,binop-binding])
                        ,context)))]
      [`(call ,v ,vs ...)
       (define var (fresh))
       (define call-bindings (generate-call-bindings v vs))
       (values var (λ (context)
                     `(let ([,var ,call-bindings])
                        ,context)))]
      [`(let ([,as ,vs] ...) ,v)
       (define var (fresh))
       (define let-bindings (generate-let-bindings as vs (rco-tail v)))
       (values var (λ (context)
                     `(let ([,var ,let-bindings])
                        ,context)))]
      [`(if ,pred ,v1 ,v2)
       (define var (fresh))
       (values var (λ (context)
                     `(let ([,var (if ,(rco-pred pred)
                                      ,(rco-tail v1)
                                      ,(rco-tail v2))])
                        ,context)))]))

  ;; Helpers

  ;; (ListOf aloc) (ListOf value) value | pred -> value | pred
  ;; generates (possibly) a set of bindings to precede a let with (potentially) nested binop expressions
  (define (generate-let-bindings as vs expr)
    (define bindings
      (for/fold ([bindings-acc '()])
                ([aloc (reverse as)]
                 [value (reverse vs)])
        (cons `[,aloc ,(rco-tail value)] bindings-acc)))
    `(let ,bindings ,expr))

  ;; value (ListOf value) -> value
  ;; generates (possibly) a set of bindings for a call with (potentially) nested binop expresssions
  (define (generate-call-bindings v vs)
    (define-values (opands opand-bindings)
      (for/fold ([opands-acc '()]
                 [bindings-acc '()])
                ([value (cons v vs)])
        (define-values (new-opand new-binding) (rco-opand value))
        (values (cons new-opand opands-acc) (cons new-binding bindings-acc))))
    (for/fold ([arg-acc `(call ,@(reverse opands))])
              ([binding opand-bindings])
      (binding arg-acc)))

  ;; binop | relop value value -> value | pred
  ;; generates (possibly) a set of bindings for a binop with (potentially) nested binop expresssions
  (define (generate-op-bindings op v1 v2)
    (define-values (opand1 binding1) (rco-opand v1))
    (define-values (opand2 binding2) (rco-opand v2))
    (binding1 (binding2 `(,op ,opand1 ,opand2))))

  (rco-program p))
