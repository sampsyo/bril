#lang racket/base

(require "cfg.rkt"
         "ast.rkt"
         "helpers.rkt"
         racket/format
         racket/list
         racket/pretty
         racket/match)

(provide lvn
         use-bug-overridden
         use-bug-assoc
         use-bug-lookup)

(define use-bug-overridden (make-parameter #f))
(define use-bug-assoc (make-parameter #f))
(define use-bug-lookup (make-parameter #f))

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

;; set up table things
;; XXX(sam) change the instruction type
(struct row (idx value canon) #:transparent)
(define (lookup table tar-value)
  (if (use-bug-lookup)
      (findf (match-lambda
               [(row _ (dest-instr _ type val) _)
                (and (dest-instr? tar-value)
                     (equal? type (dest-instr-type tar-value))
                     (equal? val (dest-instr-vals tar-value)))]
               [(row _ (branch c _ _) _)
                (and (branch? tar-value)
                     (equal? c (branch-con tar-value)))])
             table)
      (findf (lambda (row)
               (define instr (row-value row))
               (match instr
                 [(dest-instr _ type val)
                  (cond [(dest-instr? tar-value)
                         (match-define (dest-instr _ t-type t-val) tar-value)
                         (equal? ((instr-constr instr) "" type val)
                                 ((instr-constr tar-value) "" t-type t-val))]
                        [else #f])]
                 [(branch c _ _)
                  (if (branch? tar-value)
                      (equal? c (branch-con tar-value))
                      #f)]))
             table)))

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

(define (associative? instr)
  (ormap (lambda (f) (f instr))
         (list add?
               mul?
               ieq?
               land?
               lor?)))

(define (canonicalize instr)
  (cond [(dest-instr? instr)
         (match-define (dest-instr dest type vals) instr)
         (if (use-bug-assoc)
             (if (andmap string? vals)
                 ((instr-constr instr) dest type (sort vals string<?))
                 instr)
             (if (and (andmap string? vals) (associative? instr))
                 ((instr-constr instr) dest type (sort vals string<?))
                 instr))]
        [else instr]))

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
               (canonicalize
                (apply-instr (lambda (x) (hash-ref var2num x x))
                             instr)))

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
                                  (id dest type (list canon))))])]

                   ;; value is not in the table
                   [(dest-instr? instr)
                    ;; XXX(sam) change dest is the instr will be overridden later

                    (define dest
                      (if (use-bug-overridden)
                          (dest-instr-dest instr)
                          (if (will-be-overridden? instrs instr-idx
                                                   (dest-instr-dest instr))
                              (fresh-var (dest-instr-dest instr))
                              (dest-instr-dest instr))))

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
