#lang racket

;; Copyright (C) 2023 Colby Sparks, Joseph Maheshe, Silviu Toderita.

(require
  cpsc411/compiler-lib)

(provide
  implement-safe-primops)

;; EXERCISE 4
;;
;; exprs-unique-lang-v7 -> exprs-unsafe-data-lang-v7
;; Implements safe primitive operations by inserting procedure definitions 
;; for each primitive operation which perform dynamic tag checking, to ensure type safety
(define (implement-safe-primops p)
  ;; env maps primitive procedures to primitive labels
  (define env (make-hash))
  ;; list of defines for primitive procedures
  (define prim-f-defs `())

  (define safe-prim-fs `(eq? fixnum? boolean? empty? void? ascii-char? error? not))
  (define unsafe-prim-fs `(* + - < <= > >=))
  ;; map of primitive procedures to primitive operations
  (define prim-f-proc-map `(,@(map (lambda (x) 
                                     `(,x ,(string->symbol (format "~a~a" `unsafe-fx x))))
                                   unsafe-prim-fs)
                             ,@(map (lambda (x) `(,x ,x))
                                    safe-prim-fs)))
  ;; map of unsafe primitive procedures to error codes
  (define unsafe-prim-f-code-map `((* 1) (+ 2) (- 3) (< 4) (<= 5) (> 6) (>= 7)))

  ;; exprs-unique-lang-v7 -> exprs-unsafe-data-lang-v7
  (define (implement-program program)
    (match program
      [`(module ,defs ... ,val)
        ;; implement-safe-primops entry val and populate prim-f-defs
        (define impl-val (implement-val val))
        `(module ,@(map implement-def defs) ,@prim-f-defs ,impl-val)]))

  ;; exprs-unique-lang-v7-define -> exprs-unsafe-data-lang-v7-define
  (define (implement-def def)
    (match def
      [`(define ,label (lambda (,alocs ...) ,val))
        `(define ,label (lambda ,alocs ,(implement-val val)))]))

  ;; exprs-unique-lang-v7-value -> exprs-unsafe-data-lang-v7-value
  (define (implement-val val)
    (match val
      [`(let ([,alocs ,vals] ...) ,val)
        `(let ,(map (lambda (aloc val) (list aloc (implement-val val))) alocs vals) 
           ,(implement-val val))]
      [`(if ,val_1 ,val_2 ,val_3)
        `(if ,@(map implement-val (list val_1 val_2 val_3)))]
      [`(call ,val ,vals ...)
        `(call ,@(map implement-val (cons val vals)))]
      [triv
        (implement-triv triv)]))

  ;; exprs-unique-lang-v7-triv -> exprs-unsafe-data-lang-v7-triv
  (define (implement-triv triv)
    (match triv
      [label
        #:when(label? label)
        triv]
      [aloc
        #:when(aloc? aloc)
        triv]
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
      [prim-f
        (implement-prim-f prim-f)]))

  ;; exprs-unique-lang-v7-prim-f -> exprs-unsafe-data-lang-v7-label
  ;; Compiles primitive procedures to primitive labels
  ;; associated with defines added to block of defines
  (define (implement-prim-f prim-f)
    (if (dict-has-key? env prim-f)
      (dict-ref env prim-f)
      (let ([label (fresh-label prim-f)])
        (dict-set! env prim-f label)
        (set! prim-f-defs (cons `(define ,label ,(prim-f->lambda prim-f)) prim-f-defs))
        label)))

  ;; Functions intentionally omitted:
  ;; implement-binop
  ;; implement-unop

  ;; Helpers

  ;; exprs-unique-lang-v7-prim-f -> exprs-unsafe-data-lang-v7-lambda
  (define (prim-f->lambda prim-f)
    (define proc (first (dict-ref prim-f-proc-map prim-f)))

    (match prim-f
      [binop
        #:when (memq prim-f `(* + - < eq? <= > >=)) 
        (define x (fresh)) 
        (define y (fresh))
        (if (memq prim-f safe-prim-fs)
          `(lambda (,x ,y)
             (,proc ,x ,y))
          (let ([code (first (dict-ref unsafe-prim-f-code-map prim-f))])
            `(lambda (,x ,y) 
               (if (fixnum? ,y)
                 (if (fixnum? ,x)
                   (,proc ,x ,y)
                   (error ,code))
                 (error ,code)))))]
      [unop
        (define x (fresh))
        `(lambda (,x)
           (,proc ,x))]))

  (implement-program p))
