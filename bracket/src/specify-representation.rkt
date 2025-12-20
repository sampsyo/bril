#lang racket

;; Copyright (C) 2023 Colby Sparks, Joseph Maheshe, Silviu Toderita.

(require
  cpsc411/compiler-lib
  "util.rkt")

(provide
 specify-representation)

;; EXERCISE 5
;;
;; exprs-unsafe-data-lang-v7? -> exprs-bits-lang-v7?
;; Compiles immediate data and primitive operations into their implementations as ptrs and
;; primitive bitwise operations on ptrs.
(define (specify-representation p)

  ;; exprs-unsafe-data-lang-v7? -> exprs-bits-lang-v7?
  (define (specify-program program)
    (match program
      [`(module ,defs ... ,value)
       `(module ,@(map specify-def defs) ,(specify-value value))]))

  ;; exprs-unsafe-data-lang-v7-def -> exprs-bits-lang-v7-def
  (define (specify-def def)
    (match def
      [`(define ,label (lambda (,alocs ...) ,value))
       `(define ,label (lambda ,alocs ,(specify-value value)))]))

  ;; exprs-unsafe-data-lang-v7-value -> exprs-bits-lang-v7-value
  (define (specify-value value)
    (match value
      [`(call ,value ,values ...)
       `(call ,(specify-value value) ,@(map specify-value values))]
      [`(let ([,alocs ,values] ...) ,value)
       `(let ,(map (lambda (aloc value) `[,aloc ,(specify-value value)]) alocs values)
          ,(specify-value value))]
      [`(if ,value ,value1 ,value2)
       `(if (!= ,(specify-value value) ,(current-false-ptr))
            ,(specify-value value1)
            ,(specify-value value2))]
      [`(,primop ,values ...)
       #:when (primop? primop)
       (specify-primop primop values)]
      [triv
       (specify-triv triv)]))

  ;; exprs-unsafe-data-lang-v7-triv -> exprs-bits-lang-v7-triv
  (define (specify-triv triv)
    (match triv
      [`#t
       (current-true-ptr)]
      [`#f
       (current-false-ptr)]
      [`empty
       (current-empty-ptr)]
      [`(void)
       (current-void-ptr)]
      [`(error ,uint8)
       (bitwise-ior (arithmetic-shift uint8 (current-error-shift)) (current-error-tag))]
      [fixnum
       #:when(fixnum? fixnum)
       (bitwise-ior (arithmetic-shift fixnum (current-fixnum-shift)) (current-fixnum-tag))]
      [ascii-char-literal
       #:when(ascii-char-literal? ascii-char-literal)
       (bitwise-ior (arithmetic-shift (char->integer ascii-char-literal)
                                      (current-ascii-char-shift))
                    (current-ascii-char-tag))]
      [label
       triv]
      [aloc
       triv]))

  ;; exprs-unsafe-data-lang-v7-primop (listof value) -> exprs-bits-lang-v7-value
  ;; Generate a value from a primop and 1 or 2 (operand) values
  (define (specify-primop primop values)
    (match primop
      [binop
       #:when(binop? binop)
       (specify-binop binop (first values) (second values))]
      [unop
       #:when(unop? unop)
       (specify-unop unop (first values))]))

  ;; exprs-unsafe-data-lang-v7-binop
  ;; exprs-unsafe-data-lang-v7-value
  ;; exprs-unsafe-data-lang-v7-value
  ;;  -> exprs-bits-lang-v7-value
  ;; Generate a value from a binop and 2 (operand) values
  (define (specify-binop binop value1 value2)

    (define (unsafe-binop->relop b)
      (match b
        [`eq? `=]
        [`unsafe-fx< `<]
        [`unsafe-fx<= `<=]
        [`unsafe-fx> `>]
        [`unsafe-fx>= `>=]))

    (match binop
      [`unsafe-fx*
       (cond
         [(fixnum? value1)
          `(* ,value1 ,(specify-value value2))]
         [(fixnum? value2)
          `(* ,(specify-value value1) ,value2)]
         [else
          `(* ,(specify-value value1)
              (arithmetic-shift-right ,(specify-value value2) ,(current-fixnum-shift)))])]
      [`unsafe-fx+
       `(+ ,(specify-value value1) ,(specify-value value2))]
      [`unsafe-fx-
       `(- ,(specify-value value1) ,(specify-value value2))]
      ;; Any other unsafe binops must be a relop
      [relop
       `(if (,(unsafe-binop->relop relop) ,(specify-value value1) ,(specify-value value2))
            ,(current-true-ptr)
            ,(current-false-ptr))]))

  ;; exprs-unsafe-data-lang-v7-unop exprs-unsafe-data-lang-v7-value
  ;; -> exprs-bits-lang-v7-value
  ;; Generate a value from an unop and 1 (operand) value
  (define (specify-unop unop value)

    (define (check-tag value mask tag)
      `(if (= (bitwise-and ,(specify-value value) ,mask) ,tag)
           ,(current-true-ptr)
           ,(current-false-ptr)) )

    (match unop
      [`fixnum?
       (check-tag value (current-fixnum-mask) (current-fixnum-tag))]
      [`boolean?
       (check-tag value (current-boolean-mask) (current-boolean-tag))]
      [`empty?
       (check-tag value (current-empty-mask) (current-empty-tag))]
      [`void?
       (check-tag value (current-void-mask) (current-void-tag))]
      [`ascii-char?
       (check-tag value (current-ascii-char-mask) (current-ascii-char-tag))]
      [`error?
       (check-tag value (current-error-mask) (current-error-tag))]
      [`not
       `(if (!= ,(specify-value value) ,(current-false-ptr))
            ,(current-false-ptr)
            ,(current-true-ptr))]))

  ;; helpers

  ;; Any -> Boolean
  ;; Checks if p is a primop
  (define (primop? p)
    (or (binop? p) (unop? p)))

  ;; Any -> Boolean
  ;; Checks if b is a binop
  (define (binop? b)
    (match b
      [`unsafe-fx* #t]
      [`unsafe-fx+ #t]
      [`unsafe-fx- #t]
      [`eq? #t]
      [`unsafe-fx< #t]
      [`unsafe-fx<= #t]
      [`unsafe-fx> #t]
      [`unsafe-fx>= #t]
      [_ #f]))

  ;; Any -> Boolean
  ;; Checks if u is an unop
  (define (unop? u)
    (match u
      [`fixnum? #t]
      [`boolean? #t]
      [`empty? #t]
      [`void? #t]
      [`ascii-char? #t]
      [`error? #t]
      [`not #t]
      [_ #f]))

  (specify-program p))
