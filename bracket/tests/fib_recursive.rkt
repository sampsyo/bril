(module
    (define fibo
      (lambda (n) 
        (if (call <= n 1)
          n
          (call + (call fibo (call - n 1)) (call fibo (call - n 2))))))
    (let ([arg.x 10])
      (call fibo arg.x)))
