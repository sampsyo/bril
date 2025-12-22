(module
    (define fact
      (lambda (n) 
        (if (call <= n 1)
          1
          (call * (call fact (call - n 1)) n))))
    (let ([arg.x 8])
      (call fact arg.x)))
