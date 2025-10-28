// Implements an approximate sin via polynomial evaluation with coefficients from Hart et. al
// https://www.hpmuseum.org/cgi-bin/archv021.cgi?read=256069
// It appears to be very bad- I assume this is an error on my part.

var pi = 3.14159;
var a = 1.57079;
var b = 0.0 - 0.64589;
var c = 0.07943;
var d = 0.0 - 0.00433;

// Angle in radians. This value should be lower than 2pi for better accuracy
var x = 2.5;

// symmetry
var sign = 1;
if (x < 0) {
    sign = 0 - 1;
    x = 0.0 - x;
}
if (x > pi / 2) {
    x = pi - x;
}

// evaluate the polynomial
var x2 = x * x;
var result = x * (a + x2 * (b + x2 * (c + x2 * d)));

result = sign * result;

console.log(result);
