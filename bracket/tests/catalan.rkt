(module
    (define loop
      (lambda (res i n)
        (if (call < i n)
          (call loop 
                (call + res (call * (call cat i) (call cat (call - (call - n 1) i)))) 
                (call + i 1) 
                n)
          res)))
    (define cat
      (lambda (n) 
        (if (call <= n 1)
          1
          (call loop 0 0 n))))
    (let ([arg.x 10])
      (call cat arg.x)))
