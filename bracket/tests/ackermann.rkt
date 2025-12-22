(module
    (define ack 
      (lambda (m n) 
        (if (call eq? m 0)
          (call + n 1)
          (if (call eq? n 0)
            (call ack (call - m 1) 1)
            (call ack (call - m 1) (call ack m (call - n 1)))))))
    (let ([arg.x 3] [arg.y 6])
      (call ack arg.x arg.y)))
