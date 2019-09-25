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
  (define (add-duplicate hd lst)
    (match lst
      [(cons (cons a _) tl)
       (cons (cons hd #f) (cons (cons a hd) tl))]))

  (for/fold ([accum (list (cons (car l) #f))]  ;; init accum
             #:result (reverse (cdr accum)))   ;; return reverse of tail when done
            ([hd (cdr l)])
    (add-duplicate hd accum)))

(define (extract-and-accumulate instr acc)
  (match instr
    [(label name)
     (cons (basic-block name '()) acc)]

    [(or (jump _)
         (branch _ _ _)
         (return))
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
      [(cons (basic-block from instrs)
             (basic-block fallthrough _))
       (match (car instrs)
         [(jump to)
          (add-directed-edge! g from to)]
         [(branch _ tbr fbr)
          (begin
            (add-directed-edge! g from tbr)
            (add-directed-edge! g from fbr))]
         [(return) (void)] ;; XXX(sam) sometimes add a return block
         ;; fall through edge
         [_ (add-directed-edge! g from fallthrough)])]))

  ;; reverse the instruction lists in the basic blocks
  (define blocks-hash
    (for/hash ([block blocks])
      (match-define (basic-block lbl instrs) block)
      (values lbl
              (struct-copy basic-block block
                   [instrs (reverse instrs)]))))

  (cons blocks-hash g))
