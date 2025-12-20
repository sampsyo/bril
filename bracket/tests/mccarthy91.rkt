(module
    (define mac
      (lambda (n) 
        (if (call > n 100)
          (call - n 10)
          (call mac (call mac (call + n 11))))))
    (let ([arg.x 15])
      (call mac arg.x)))
