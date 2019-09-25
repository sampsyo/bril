#lang racket/base

(require "cfg.rkt"
         "ast.rkt"
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
(define (lookup table value)
  (findf (match-lambda
           [(row _ val _)
            (equal? val value)])
         table))

;; does local value numbering on a single block
(define (local-value-numbering block)

  ;; set up data structures
  (define var2num (make-hash))
  (define table (list))

  (define block-p
    (for/list ([instr (basic-block-instrs block)])
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
                   [else
                    ;; XXX(sam) change dest is the instr will be overridden later
                    (define dest (if (dest-instr? instr)
                                     (dest-instr-dest instr)
                                     #f))

                    (set! num (length table))

                    ;; update the table with the new pair
                    ;; XXX(sam) list is not best for this
                    (set! table
                          (append table
                                (list (row (length table) value dest))))

                    ;; replace instr args with canonical variables for the values
                    (set! res-instr
                          (apply-instr (lambda (v)
                                         (define idx (hash-ref var2num v #f))
                                         (if idx
                                             (row-canon (list-ref table idx))
                                             v))
                                       instr))])

             (when (dest-instr? instr)
               (hash-set! var2num (dest-instr-dest instr) num))
             res-instr]

            ;; else, instruction has no values
            [else instr])))

  (println "-----")
  (pretty-print table)
  (pretty-print var2num)
  (pretty-print block-p)
  (println "-----")

  ;; (pretty-print block)
  block-p)

(define (lvn cfg)
  (match-define (cons blocks graph) cfg)
  (define blocks-p
    (hash-map blocks
              (lambda (k v)
                (cons k (local-value-numbering v)))))
  ;; use apply hash to make immutable hash
  (cons (apply hash blocks-p) graph))
