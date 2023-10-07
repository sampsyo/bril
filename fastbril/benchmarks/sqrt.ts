// Implements the Newton-Raphson method for 
// approximating square root.
// Implementation is somewhat based on this
// reference:
// https://www.geeksforgeeks.org/find-root-of-a-number-using-newtons-method/

var n = 327.0;
var precision = 0.00001;
var x = n;
var notdone = true;

for(;notdone;){
    var root = n/x;
    root = x + root;
    root = 0.5 * root;
    var diff = root - x;
    if(diff < 0){
        diff = 0.0 - diff;
    }
    if(diff < precision){
        notdone = false;
    }
    x = root;
}
console.log(x);
