#lang racket/base

(require "ast.rkt"
         "external.rkt"
         threading
         graph
         racket/pretty
         racket/list
         racket/match
         racket/hash)

(provide cfg
         (struct-out basic-block))

; A basic block has a label and a sequence of control-free instructions.
(struct basic-block (label instrs) #:transparent)

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

(define (extract-and-accumulate instr acc)
  (match instr
    [(label name)
     (cons (basic-block name '()) acc)]

    [(or (jump _)
         (branch _ _ _))
     (let ([unique (unique-label "hang")])
       (match acc
         [(cons (basic-block lbl instrs) tl)
          (cons (basic-block unique '())
                (cons (basic-block lbl (cons instr instrs)) tl))]))]

    [_ (match acc
         [(cons (basic-block lbl instrs) tl)
          (cons (basic-block lbl (cons instr instrs)) tl)])]))


;; should be given a function
(define (cfg ast)
  (define g (unweighted-graph/undirected '()))

  (define blocks
    (~> ast
        (foldl
          extract-and-accumulate
          (list (basic-block "start" '())) ;; init empty start block
          _)
        reverse
        (filter-not (compose empty? basic-block-instrs) _)))

  ;; add vertex for every block
  (for ([block blocks])
    (match block
      [(basic-block lbl _) (add-vertex! g lbl)]))

  ;; construct edges, using pairs to add fall through edges
  (for ([block-pair (pairs blocks)])
    (match block-pair
      [(cons (basic-block from instrs) bl1)
       (match (car instrs)
         [(jump to)
          (add-directed-edge! g from to)]
         [(branch _ tbr fbr)
          (begin
            (add-directed-edge! g from tbr)
            (add-directed-edge! g from fbr))]
         [_ (add-directed-edge! g from (basic-block-label bl1))])]))

  ;; reverse the instruction lists in the basic blocks
  (define blocks-hash
    (for/hash ([block blocks])
      (match-define (basic-block lbl instrs) block)
      (values lbl
              (struct-copy basic-block block
                   [instrs (reverse instrs)]))))

  (cons blocks-hash g))
