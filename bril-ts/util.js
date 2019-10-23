"use strict";
exports.__esModule = true;
/**
 * Read all the data from stdin as a string.
 */
function readStdin() {
    return new Promise(function (resolve, reject) {
        var chunks = [];
        process.stdin.on("data", function (chunk) {
            chunks.push(chunk);
        }).on("end", function () {
            resolve(chunks.join(""));
        }).setEncoding("utf8");
    });
}
exports.readStdin = readStdin;
function unreachable(x) {
    throw "impossible case reached";
}
exports.unreachable = unreachable;
