### Strength Reduction

Strength reduction is a compiler optimization where expensive operations are replaced with equivalent but less expensive operations. A typical example of it is to convert relatively more complex multiplications inside a loop `L` to easier additions. Here we are mostly interested in 

1. loop invariants: values that do not change within the body of a loop (as we have already discussed previously)
2. induction variables: values that are being iterated each time through the loop

Here is the definition of induction variable:
it is either 
* a basic induction variable `B`: a variable `B` whose only definitions within the loop are assignments of the form: `B = B + c` or `B = B - c`, where c is either a constant or a loop-invariant variable, or
* a variable defined once within the loop, whose value is a linear function of some basic induction variable at the time of the definition `A = c1 * B + c2`

The procedure of performing strength reduction is as follows:
1. Create new variable: `A'`
2. Initialize in preheader: `A’ = c1 * B + c2`
3. Track value of B: add after `B=B+x`: `A’=A’+x*c1`
4. Replace assignment to A: replace lone `A=...` with `A=A’`

Thus, the key idea here is to first find out each induction variable `A` and then replace definition of A when executed.

To find out each induction variable, we scan through the code to
1. find out all the basic induction variables `B`
2. find out all induction variables `A` in family of `B`, where `A` refers to the `B` at the time of definition

The `A` here should be in one of the following conditions:
i. `A` has a single assignment in the loop `L` in the form of:
	`A = B * c` | `A = c * B` | `A = B / c` | `A = B + c` | `A = c + B` | `A = B - c` | `A = c - B`
ii. `A` has a single assignment in the loop `L` in the form of (`D` is an induction variable in the family of `B`) 
	`A = D * c` | `A = c * D` | `A = D / c` | `A = D + c` | `A = c + D` | `A = D - c` | `A = c - D`
Also, no definitions of `D` outside `L` reaches the assignment of `A`, and every path between the point of assignment to `D` in `L` and the assignment to `A` has the same sequence (possibly empty) of definitions of `B`.

After all induction variables are found, strength reduction is performed to add new initialization to the variable and reduce multiplications to additions following the procedure we have described above.

Hardest Part:
1. There are more similarity between basic induction variables and their families. It is sometimes tricky to differentiate them. Thus, the definition flow of each induction variable is maintained to tell them apart.