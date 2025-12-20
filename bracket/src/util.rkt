#lang racket

;; Copyright (C) 2023 Colby Sparks, Joseph Maheshe, Silviu Toderita.

(require
 cpsc411/compiler-lib)

(provide
 triv?
 opand?
 relop?
 binop?
 binop->ins
 arithmetic-shift-right)

;; number number -> number
;; shifts bits of lhs by rhs to the right
(define (arithmetic-shift-right lhs rhs)
  (arithmetic-shift lhs (- rhs)))

;; relop? -> boolean
;; returns true if r is relop
(define (relop? r)
  (member r '(< <= > >= = !=)))

;; binop? -> boolean
;; returns true if b is binop
(define (binop? b)
  (member b '(+ * - bitwise-and bitwise-ior bitwise-xor arithmetic-shift-right)))

;; binop? -> ((number ...) -> number)
;; returns operator associated with binop
(define (binop->ins b)
  (match b
    ['+ +]
    ['* *]
    ['- -]
    ['bitwise-and bitwise-and]
    ['bitwise-ior bitwise-ior]
    ['bitwise-xor bitwise-xor]
    ['arithmetic-shift-right arithmetic-shift-right]))

;; any -> boolean
;; Returns true if t is triv
(define (triv? t)
  (or (label? t) (opand? t)))

;; any -> boolean
;; Returns true if o is opand
(define (opand? o)
  (or (int64? o) (aloc? o)))
