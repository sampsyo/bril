(module
    (define delan
      (lambda (m n) 
        (if (call eq? m 0)
          1
          (if (call eq? n 0)
            1
            (call + (call + (call delan (call - m 1) n) (call delan m (call - n 1))) 
                    (call delan (call - m 1) (call - n 1)))))))
    (let ([arg.x 8])
      (call delan arg.x arg.x)))
