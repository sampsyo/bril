#lang racket/base

(require "cfg.rkt")

(provide local-value-numbering)

;; does local value numbering on a single block
(define (local-value-numbering block)
  block)
