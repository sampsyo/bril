#lang racket

;; Copyright (C) 2023 Colby Sparks, Joseph Maheshe, Silviu Toderita.

(require
  cpsc411/compiler-lib
  "util.rkt")

(provide
 uniquify)

;; EXERCISE 3
;;
;; CITATION - Copied from Colby's M2 into group's M3
;;
;; exprs-lang-v7? -> exprs-unique-lang-v7?
;; Compiles exprs-lang-v7 to exprs-unique-lang-v7 by resolving top-level lexical identifiers
;; into unique labels, and all other lexical identifiers into unique abstract locations.
(define (uniquify p)

  ;; exprs-lang-v7? -> exprs-unique-lang-v7?
  ;; Maps all lexical identifiers x to unique identifiers x.number
  (define (uniquify-program program)
    (match program
      [`(module (define ,ls (lambda ,params ,proc-vals)) ... ,val)
       ; get procedure labels
       (define env-labels
         (for/fold ([env-fold '()])
                   ([l ls])
           (dict-set env-fold l (fresh-label l))))
       `(module
            ,@(map (curryr uniquify-procedure env-labels) ls params proc-vals)
          ,(uniquify-val val env-labels))]))

  ;; label? (ListOf opand) val env -> exprs-unique-lang-v7-procedure
  ;; maps procedure name to unique label and parameter names to unique alocs
  ;; env maps names in lexical scope to their unique identifier: `((x . x.1))
  ;;       - note that procedure names are mapped to labels
  (define (uniquify-procedure l params val env)
    (define params-env
      (for/fold ([env-fold env])
                ([p params])
        (dict-set env-fold p (fresh p))))
    `(define
       ,(dict-ref env l)
       (lambda ,(map (curry dict-ref params-env) params) ,(uniquify-val val params-env))))

  ;; exprs-lang-v7-value env -> exprs-unique-lang-v7-value
  (define (uniquify-val val env)
    (match val
      [`(let ([,xs ,vals] ...) ,val)
       (define new-env
         (for/fold ([new-env env])
                   ([x xs])
           (dict-set new-env x (fresh x))))
       `(let ,(map (λ (x val) (list (dict-ref new-env x) (uniquify-val val env))) xs vals)
          ,(uniquify-val val new-env))]
      [`(if ,val_1 ,val_2 ,val_3)
       `(if ,@(map (curryr uniquify-val env) (list val_1 val_2 val_3)))]
      [`(call ,val ,vals ...)
       `(call ,@(map (curryr uniquify-val env) (cons val vals)))]
      [triv
       (uniquify-triv triv env)]))

  ;; exprs-lang-v7-triv env -> exprs-unique-lang-v7-triv
  ;; returns opand or label associated with name key in env
  (define (uniquify-triv triv env)
    (match triv
      [fixnum
        #:when(fixnum? fixnum)
        triv]
      [`#t
        triv]
      [`#f
        triv]
      [`empty
        triv]
      [`(void)
        triv]
      [`(error ,uint8)
        triv]
      [ascii-char-literal
        #:when(ascii-char-literal? ascii-char-literal)
        triv]
      [x
        (if (prim-f? x)
          x
          (dict-ref env x))]))

  ;; Trivial functions intentionally omitted:
  ;; uniquify-prim-f
  ;; uniquify-binop
  ;; uniquify-unop

  ;; Helpers
  ;;

  ;; Any -> Boolean
  ;; Checks if prim-f is a prim-f
  (define (prim-f? prim-f)
    (if (or (binop? prim-f) (unop? prim-f))
      #t
      #f))

  ;; Any -> Boolean
  ;; Checks if binop is a binop
  (define (binop? binop)
    (if (memq binop `(* + - < eq? <= > >=))
      #t
      #f))

  ;; Any -> Boolean
  ;; Checks if unop is a unop
  (define (unop? unop)
    (if (memq unop `(fixnum? boolean? empty? void? ascii-char? error? not))
      #t
      #f))

  (uniquify-program p))
