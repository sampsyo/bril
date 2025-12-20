(module
    (define loop
      (lambda (prod n) 
        (if (call >= n 1)
          (call loop (call * n prod) (call - n 1))
          prod)))
    (let ([arg.x 8])
      (call loop 1 arg.x)))
