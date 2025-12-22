(module
    (define loop
      (lambda (a b) 
        (if (call eq? a b)
          a
          (if (call < a b)
            (let ([c (call - b a)])
              (call loop a c))
            (let ([c (call - a b)])
              (call loop b c))))))
    (let ([arg.x 4] [arg.y 20])
      (call loop arg.x arg.y)))
