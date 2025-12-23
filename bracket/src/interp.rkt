#lang racket

;; Copyright (C) 2025 Joseph Maheshe.

(require 
  cpsc411/langs/v7)

;;exprs-lang-v7 interpreter driver

;; read program from stdin
(define prog (read))

;; validate language
(unless (exprs-lang-v7? prog)
  (error 'run "not an exprs-lang-v7 program"))

;; interpret
(let ([result (interp-exprs-lang-v7 prog)])
  (if (boolean? result)
    (if result 1 0)
    result))

