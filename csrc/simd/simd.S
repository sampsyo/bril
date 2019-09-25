// ~37 dyanamic instructions for vec
// runtime 10k iterations is 91us over five runs

_ZN4SIMD6vecAddEv:
.LFB5218:
    .cfi_startproc
    pushq   %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset 6, -16
    movq    %rsp, %rbp
    .cfi_def_cfa_register 6
    subq    $192, %rsp
    movq    %fs:40, %rax
    movq    %rax, -8(%rbp)
    xorl    %eax, %eax
    leaq    -64(%rbp), %rax
    movq    %rax, -168(%rbp)
    movq    -168(%rbp), %rax
    movdqu  (%rax), %xmm0
    movaps  %xmm0, -160(%rbp)
    leaq    -48(%rbp), %rax
    movq    %rax, -176(%rbp)
    movq    -176(%rbp), %rax
    movdqu  (%rax), %xmm0
    movaps  %xmm0, -144(%rbp)
    movdqa  -160(%rbp), %xmm0
    movaps  %xmm0, -96(%rbp)
    movdqa  -144(%rbp), %xmm0
    movaps  %xmm0, -80(%rbp)
    movdqa  -96(%rbp), %xmm1
    movdqa  -80(%rbp), %xmm0
    paddd   %xmm1, %xmm0
    movaps  %xmm0, -128(%rbp)
    leaq    -32(%rbp), %rax
    movq    %rax, -184(%rbp)
    movdqa  -128(%rbp), %xmm0
    movaps  %xmm0, -112(%rbp)
    movdqa  -112(%rbp), %xmm0
    movq    -184(%rbp), %rax
    movups  %xmm0, (%rax)
    nop
    movq    -8(%rbp), %rax
    xorq    %fs:40, %rax
    je  .L8
    call    __stack_chk_fail@PLT
.L8:
    leave
    .cfi_def_cfa 7, 8
    ret
    .cfi_endproc


// 13 + 14 * 4 = 69 , about 2x the number of dynamic instructions of simd
// runtime 10k iterations is 163us (over five runs)

_ZN4SIMD13vecAdd_serialEv:
.LFB5219:
    .cfi_startproc
    pushq   %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset 6, -16
    movq    %rsp, %rbp
    .cfi_def_cfa_register 6
    subq    $80, %rsp
    movq    %fs:40, %rax
    movq    %rax, -8(%rbp)
    xorl    %eax, %eax
    movl    $0, -68(%rbp)
.L11:
    cmpl    $3, -68(%rbp)
    jg  .L13
    movl    -68(%rbp), %eax
    cltq
    movl    -64(%rbp,%rax,4), %edx
    movl    -68(%rbp), %eax
    cltq
    movl    -48(%rbp,%rax,4), %eax
    addl    %eax, %edx
    movl    -68(%rbp), %eax
    cltq
    movl    %edx, -32(%rbp,%rax,4)
    addl    $1, -68(%rbp)
    jmp .L11
.L13:
    nop
    movq    -8(%rbp), %rax
    xorq    %fs:40, %rax
    je  .L12
    call    __stack_chk_fail@PLT
.L12:
    leave
    .cfi_def_cfa 7, 8
    ret
    .cfi_endproc