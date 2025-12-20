(module
    (define tri
      (lambda (n) 
        (if (call eq? n 1)
          1
          (call + (call tri (call - n 1)) n))))
    (let ([arg.x 36])
      (call tri arg.x)))
