## Goal

The goal of the project is to replace expensive operations like multiplies with equivalent cheaper operations like additions. This is done with the help of loop invariant code motion and induction variable analysis. 

## Design

We divided the project into 3 parts - 
1. Preprocessing to LICM: Finding back edges and associated loops along with reaching definitions of variables.
2. LICM algorithm: Finding loop invariant code and moving it into a pre-header outside the loop.
3. Strength reduction: Using the tools like LICM to simply constant expressions and analyzing induction variables that can be computed using cheaper operations


