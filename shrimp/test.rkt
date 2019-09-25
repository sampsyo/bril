#lang rosette

(require rosette/query/debug rosette/lib/synthax rosette/lib/render)

(define-symbolic x integer?)

(define (poly x)
  (+ (* x x x x) (* 6 x x x) (* 11 x x) (* 6 x)))

(define (factored x)
  (* (+ x 3) (+ x 1) (+ x 0) (+ x 2)))

(define (same p f x)
  (assert (= (p x) (f x))))

(define-symbolic i integer?)
(verify (same poly factored i))

;; (define binding
;;   (synthesize #:forall (list i)
;;               #:guarantee (same poly factored i)))

;; (print-forms binding)

;; (verify (same poly factored i))


;; (define ucore (debug [integer?] (same poly factored -6)))
;; (render ucore)
