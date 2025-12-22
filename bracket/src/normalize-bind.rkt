#lang racket

;; Copyright (C) 2023 Colby Sparks, Joseph Maheshe, Silviu Toderita.

(require
  cpsc411/compiler-lib
  rackunit
  "util.rkt")

(provide
 normalize-bind)

;; CITATION - Copied from Colby's M2 to group's M3

;; EXERCISE 7
;;
;; imp-mf-lang-v7? -> proc-imp-cmf-lang-v7?
;; Compiles Imp-mf-lang v7 to Proc-imp-cmf-lang v7, pushing set! under begin so that the
;; right-hand-side of each set! is simple value-producing operation.
(define (normalize-bind p)

  (define (normalize-prog program)
    (match program
      [`(module (define ,labels (lambda ,alocs ,tails)) ... ,tail)
       `(module
            ,@(for/foldr ([ls `()])
                ([label labels]
                 [aloc alocs]
                 [tail tails])
                (cons `(define ,label (lambda ,aloc ,(normalize-tail tail)))
                      ls))
          ,(normalize-tail tail))]))

  (define (normalize-tail tail)
    (match tail
      [`(begin ,effects ... ,tail)
       `(begin
          ,@(map normalize-effect effects)
          ,(normalize-tail tail))]
      [`(if ,pred ,tail_1 ,tail_2)
       `(if ,(normalize-pred pred)
            ,@(map normalize-tail (list tail_1 tail_2)))]
      [`(call ,triv ,opand ...)
       tail]
      [value
       tail]))

  (define (normalize-effect effect)
    (match effect
      [`(begin ,effects ...)
       `(begin ,@(map normalize-effect effects))]
      [`(if ,pred ,effect_1 ,effect_2)
       `(if ,(normalize-pred pred)
            ,@(map normalize-effect (list effect_1 effect_2)))]
      [`(set! ,aloc ,value)
       (normalize-set-effect aloc value)]))

  (define (normalize-pred pred)
    (match pred
      [`(begin ,effects ... ,pred)
       `(begin
          ,@(map normalize-effect
                 effects)
          ,(normalize-pred pred))]
      [`(if ,pred_1 ,pred_2 ,pred_3)
       `(if ,@(map normalize-pred
                   (list pred_1 pred_2 pred_3)))]
      [`(not ,pred)
       `(not ,(normalize-pred pred))]
      [`(,relop ,opand1 ,opand2)
       pred]
      [`(true)
       pred]
      [`(false)
       pred]))

  ;; Functions omitted intentionally:
  ;; normalize-tail
  ;; normalize-value
  ;; normalize-opand
  ;; normalize-triv
  ;; normalize-binop
  ;; normalize-relop

  ;; Helpers:

  ;; aloc imp-mf-lang-v7-value -> proc-imp-cmf-lang-v7-effect
  ;; Normalizes set-effect (set! aloc value)
  (define (normalize-set-effect aloc value)
    (match value
      [`(begin ,effects ... ,value)
       `(begin
          ,@(map normalize-effect effects)
          ,(normalize-effect `(set! ,aloc ,value)))]
      [`(if ,pred ,value_1 ,value_2)
       `(if ,(normalize-pred pred)
            ,@(map normalize-effect
                   (map (curry append `(set! ,aloc))
                        (list `(,value_1) `(,value_2)))))]
      [`(,binop ,opand1 ,opand2)
       `(set! ,aloc ,value)]
      [triv
       `(set! ,aloc ,value)]))

  (normalize-prog p))
