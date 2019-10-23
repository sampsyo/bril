const ffi = require('ffi');
const libPath = '../native/target/release/libthread_count';

const ref = require('ref');
const ArrayType = require('ref-array');
const Int32Array = ArrayType(ref.types.int32);
const ByteArray = ArrayType(ref.types.uint8);

const libWeb = ffi.Library(libPath, {
  'add':  ['int32', [Int32Array, Int32Array, Int32Array]],
  'vadd': ['int32', [Int32Array, Int32Array, Int32Array]],
  'vmul': ['int32', [Int32Array, Int32Array, Int32Array]],
  'vsub': ['int32', [Int32Array, Int32Array, Int32Array]],
});

const { add, vadd, vmul, vsub} = libWeb;
const array = [1,2,3,4];
const array1 = new Int32Array(4);
array1[0] = 1;
array1[1] = 2;
array1[2] = 3;
array1[3] = 4;
const array2 = new Int32Array(4);

console.log(array[0]);


(function(js_array, js_array1, js_array2){
  console.log("length", js_array.length)
  let a = vadd(js_array, js_array1, js_array2);
  console.log(array2[0]);
  console.log(array2[1]);
  console.log(array2[2]);
  console.log(array2[3]);
})(array, array1, array2);