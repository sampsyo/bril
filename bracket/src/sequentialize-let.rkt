#lang racket

;; Copyright (C) 2023 Colby Sparks, Joseph Maheshe, Silviu Toderita.

(require
  cpsc411/compiler-lib
  "util.rkt")

(provide
  sequentialize-let)

;; CITATION - Copied from Colby's M2 to group's M3

;; EXERCISE 7
;;
;; values-bits-lang-v7 -> imp-mf-lang-v7
;; Compiles Values-unique-lang v7 to Imp-mf-lang v7 by picking a particular order to implement let
;; expressions using set!
(define (sequentialize-let p)

  ;; Any -> boolean
  (define (triv? t)
    (or (opand? t) (label? t)))

  ;; Any -> boolean
  (define (opand? o)
    (or (int64? o) (aloc? o)))

  ;; aloc value -> effect
  ;; Converts an aloc and value found in a let clause to an effect
  (define (let->set aloc value)
    `(set! ,aloc ,(seq-value value)))

  (define (seq-prog p)
    (match p
      [`(module (define ,ls (lambda ,ps ,p-tails)) ... ,tail)
        `(module
           ,@(map (λ (l p p-tail) `(define ,l (lambda ,p ,(seq-tail p-tail)))) ls ps p-tails)
           ,(seq-tail tail))]))

  (define (seq-pred p)
    (match p
      [`(true)
        p]
      [`(false)
        p]
      [`(not ,pred)
        `(not ,(seq-pred pred))]
      [`(let ([,alocs ,values] ...) ,pred)
        `(begin ,@(map let->set alocs values) ,(seq-pred pred))]
      [`(if ,pred1 ,pred2 ,pred3)
        `(if ,(seq-pred pred1) ,(seq-pred pred2) ,(seq-pred pred3))]
      [`(,relop ,op1 ,op2)
        p]))

  (define (seq-tail tail)
    (match tail
      [`(let ([,alocs ,values] ...) ,tail)
        `(begin ,@(map let->set alocs values) ,(seq-tail tail))]
      [`(if ,pred ,tail1 ,tail2)
        `(if ,(seq-pred pred) ,(seq-tail tail1) ,(seq-tail tail2))]
      [`(call ,l ,ops ...)
        `(call ,l ,@ops)]
      [value
        (seq-value value)]))

  (define (seq-value v)
    (match v
      [`(let ([,alocs ,values] ...) ,value)
        `(begin ,@(map let->set alocs values) ,(seq-value value))]
      [`(if ,pred ,value1 ,value2)
        `(if ,(seq-pred pred) ,(seq-value value1) ,(seq-value value2))]
      [`(,binop ,op1 ,op2)
        v]
      [`(call ,l ,ops ...)
        `(call ,l ,@ops)]
      [triv
        triv]))

  (seq-prog p))
