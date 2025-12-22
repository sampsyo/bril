(module
    (let ([arg.w1 12] [arg.h1 4] [arg.w2 5] [arg.h2 13])
      (let ([fc (if (call <= arg.w1 arg.w2) 
                  (if (call <= arg.h1 arg.h2) 
                    #t 
                    #f) 
                  #f)]
            [sc (if (call <= arg.w1 arg.h2) 
                  (if (call <= arg.h1 arg.w2) 
                    #t 
                    #f) 
                  #f)])
        (if fc #t sc))))
