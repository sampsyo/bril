const ffi = require('ffi');
const libPath = '../thread-count/native/target/release/libthread_count';
const libWeb = ffi.Library(libPath, {
  'add': [ 'int32', [ 'int32', 'int32' ] ]
});
const { add, subtract, multiply } = libWeb;
console.log('4 + 2 = ', add(4, 2));