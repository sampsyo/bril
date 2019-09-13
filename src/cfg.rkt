#lang racket/base

(require "ast.rkt"
         "external.rkt"
         threading
         graph
         racket/pretty
         racket/list
         racket/match
         racket/hash)

(provide cfg)

(define idx 0)
(define (unique-label [pre "lbl-"])
  (define l (format "~a~a" pre idx))
  (set! idx (add1 idx))
  l)

;; function that takes a list and produces a list of pairs
;; where each element is matched with it's successor
(define (pairs l)
  (~>
   (foldl (lambda (x acc)
            (match acc
              [(cons (cons a _) tl)
               `((,x . #f) . ((,a . ,x) . tl))
               (cons (cons x #f) (cons (cons a x) tl))
               ]))
          `((,(car l) . #f))
          (cdr l))
   cdr
   reverse))

(struct basic-block (label instrs) #:transparent)

;; should be given a function
(define (cfg ast)
  (define g (unweighted-graph/undirected '()))
  (define blocks
    (~> ast
        (foldl (lambda (instr acc)
                 (match instr
                   [(label name)
                    `(,(basic-block name '()) . ,acc)]

                   [(or (jump _)
                        (branch _ _ _))
                    (let ([unique (unique-label "hang")])
                      (match acc
                        [(cons (basic-block lbl instrs) tl)
                         `(,(basic-block unique '()) . (,(basic-block lbl (cons instr instrs)) . ,tl))]))]

                   [_ (match acc
                        [(cons (basic-block lbl instrs) tl)
                         `(,(basic-block lbl (cons instr instrs)) . ,tl)])]))
               `(,(basic-block "start" '())) ;; init empty start block
               _)
        reverse
        (filter-not (compose empty? basic-block-instrs) _)
        ))

  ;; add vertex for every block
  (for-each (lambda (x)
              (match x
                [(basic-block lbl _) (add-vertex! g lbl)]))
            blocks)

  ;; construct edges, using pairs to add fall through edges
  (for-each (lambda (x)
              (match x
                [(cons (basic-block from instrs) bl1)
                 (match (car instrs)
                   [(jump to)
                    (add-directed-edge! g from to)]
                   [(branch _ tbr fbr)
                    (begin
                      (add-directed-edge! g from tbr)
                      (add-directed-edge! g from fbr))]
                   [_ (add-directed-edge! g from (basic-block-label bl1))])])
              )
            (pairs blocks))

  ;; reverse the instruction lists in the basic blocks
  (define blocks-p
    (map (lambda (b)
           (match b
             [(basic-block lbl instrs)
              (basic-block lbl (reverse instrs))]))
         blocks))

  (cons blocks-p g))
