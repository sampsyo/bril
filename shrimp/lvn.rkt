#lang racket/base

(require "cfg.rkt"
         "ast.rkt"
         "helpers.rkt"
         racket/format
         racket/list
         racket/pretty
         racket/match)

(provide lvn)

(define (apply-instr f instr)
  (cond [(dest-instr? instr)
         (match-define (dest-instr dest type vals) instr)
         ((instr-constr instr) dest type
                               (map f (dest-instr-vals instr)))]
        [(branch? instr)
         (struct-copy branch instr
                      [con (f (branch-con instr))])]
        [(print-val? instr)
         (struct-copy print-val instr
                      [v (f (print-val-v instr))])]
        [else instr]))

;; (cond [(dest-instr? instr)
;;        (struct-copy
;;         dest-instr instr
;;         ;; look up vals in var2num table, if it doesn't exist, keep the var
;;         ;; XXX(sam) canonicalization
;;         [vals (map (lambda (x) (hash-ref var2num x x))
;;                          (dest-instr-vals instr))])]
;;       [(branch? instr)
;;        (let ([con (branch-con instr)])
;;          (struct-copy
;;           branch instr
;;           [con (hash-ref var2num con con)]))]
;;       [else instr])

;; set up table things
(struct row (idx value canon) #:transparent)
(define (lookup table tar-value)
  (findf (match-lambda
           [(row _ (dest-instr _ type val) _)
            (and (equal? type (dest-instr-type tar-value))
                 (equal? val (dest-instr-vals tar-value)))]
           [(row _ (branch c _ _) _)
            (equal? c (branch-con tar-value))])
         table))

;; XXX(sam) ensure that this is actually fresh
(define var-store (make-hash))
(define (fresh-var var)
  (define i (hash-ref var-store var 0))
  (define new-var (~a var "-c" i))
  (hash-set! var-store var (add1 i))
  new-var)


(define (will-be-overridden? instrs idx dest)
  (define-values (_ rst)
    (split-at instrs (add1 idx)))

  (ormap (lambda (instr)
           (cond [(dest-instr? instr)
                  (equal? dest (dest-instr-dest instr))]
                 [else #f]))
         rst))

;; does local value numbering on a single block
(define (local-value-numbering block)

  (match-define (basic-block label instrs) block)

  ;; set up data structures
  (define var2num (make-hash))
  (define table (list))

  (define instrs-p
    (for/list ([instr instrs]
               [instr-idx (in-range (length instrs))])
      (cond [;; instruction with values
             (or (dest-instr? instr)
                 (branch? instr))

             ;; replace variables in instr with value numbers
             (define value
               (apply-instr (lambda (x) (hash-ref var2num x x))
                            instr))

             (define res-instr instr)
             (define num #f)

             ;; look up value in the table
             (cond [(lookup table value)
                    ;; value is in the table
                    => (match-lambda
                         [(row idx val canon)
                          (match-let ([(dest-instr dest type _) instr])
                            (set! num idx)
                            (set! res-instr
                                  (id dest type canon)))])]

                   ;; value is not in the table
                   [(dest-instr? instr)
                    ;; XXX(sam) change dest is the instr will be overridden later

                    (define dest
                      (if (will-be-overridden? instrs instr-idx
                                              (dest-instr-dest instr))
                          (fresh-var (dest-instr-dest instr))
                          (dest-instr-dest instr)))

                    (set! num (length table))

                    ;; update the table with the new pair
                    ;; XXX(sam) list is not best for this
                    (set! table
                          (append table
                                  (list (row (length table) value dest))))

                    (match-define (dest-instr _ type vals) instr)
                    (define instr-p ((instr-constr instr) dest type vals))

                    ;; replace instr args with canonical variables for the values
                    (set! res-instr
                          (apply-instr (lambda (v)
                                         (define idx (hash-ref var2num v #f))
                                         (if idx
                                             (row-canon (list-ref table idx))
                                             v))
                                       instr-p))])

             (when (dest-instr? instr)
               (hash-set! var2num (dest-instr-dest instr) num))
             res-instr]

            ;; else, instruction has no values
            [else instr])))

  ;; (pr "-----")
  ;; (pr table)
  ;; (pr var2num)
  ;; (pr instrs)
  ;; (pr instrs-p)
  ;; (pr "-----")

  ;; (pretty-print block)
  (basic-block label instrs-p))

(define (lvn cfg)
  (match-define (cons blocks graph) cfg)
  (define blocks-p
    (hash-map blocks
              (lambda (k v)
                (cons k (local-value-numbering v)))))
  (cons (make-immutable-hash blocks-p) graph))
