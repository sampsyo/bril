#![allow(dead_code, unused_imports)]
use std::collections::HashMap;
use std::convert::Infallible;
use std::io::Read;
use std::str;

use memmap2::{Mmap, MmapMut};
use num_traits::ops::bytes;
use zerocopy::{
    FromBytes, Immutable, IntoBytes, KnownLayout, SizeError, Unaligned,
};
use zerocopy::{TryFromBytes, ValidityError};

use crate::flatten;
use crate::interp;
use crate::types::*;

/* -------------------------------------------------------------------------- */
/*                              Writing to buffer                             */
/* -------------------------------------------------------------------------- */

/// Mmaps a new file with `size` bytes, returning a handle to the mmap-ed buffer
pub fn mmap_new_file(
    filename: &str,
    size: u64,
    should_truncate: bool,
) -> MmapMut {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .truncate(should_truncate)
        .create(true)
        .open(filename)
        .unwrap();
    file.set_len(size).unwrap();
    let buffer = unsafe { MmapMut::map_mut(&file) };
    buffer.unwrap()
}

/// Writes `data` to the first `len` bytes of `buffer`,
/// (where `len = sizeof(data)`), & returns a mutable reference to
/// `buffer[len..]` (i.e. the suffix of `buffer` after the first `len` bytes).
/// - Note: Use `write_bump` when `data` has type `T` where `T` is *not* a slice
///   (i.e. when the size of `T` is not statically known)
fn write_bump<'a, T: IntoBytes + Immutable + ?Sized>(
    buffer: &'a mut [u8],
    data: &'a T,
) -> Result<&'a mut [u8], SizeError<&'a T, &'a mut [u8]>> {
    let len = size_of_val(data);
    data.write_to_prefix(buffer)?;
    Ok(&mut buffer[len..])
}

/// Writes `data` to the first `len` bytes of `buffer`
/// (where `len = sizeof(data)`), and returns a mutable reference to
/// `buffer[len..]` (i.e. the suffix of `buffer` after the first `len` bytes).
/// - Note: `write_bytes` is a specialized version of `write_bump` where
///   `data: &[u8]` (i.e. `data` is just a slice containing bytes)
fn write_bytes<'a>(buffer: &'a mut [u8], data: &[u8]) -> Option<&'a mut [u8]> {
    let len = data.len();
    buffer[0..len].copy_from_slice(data);
    Some(&mut buffer[len..])
}

/// Converts an `InstrView` to a vec of bytes
pub fn convert_instr_view_to_bytes(instr_view: &InstrView) -> Vec<u8> {
    let mut bytes_vec = vec![];

    let toc = instr_view.get_sizes();
    bytes_vec.extend_from_slice(toc.as_bytes());
    bytes_vec.extend_from_slice(instr_view.func_name.as_bytes());
    bytes_vec.extend_from_slice(instr_view.func_args.as_bytes());
    let func_ret_ty = instr_view.func_ret_ty;
    bytes_vec.extend_from_slice(func_ret_ty.as_bytes());
    bytes_vec.extend_from_slice(instr_view.var_store.as_bytes());
    bytes_vec.extend_from_slice(instr_view.arg_idxes_store.as_bytes());
    bytes_vec.extend_from_slice(instr_view.labels_idxes_store.as_bytes());
    bytes_vec.extend_from_slice(instr_view.labels_store.as_bytes());
    bytes_vec.extend_from_slice(instr_view.funcs_store.as_bytes());
    bytes_vec.extend_from_slice(instr_view.instrs.as_bytes());

    bytes_vec
}

/// Writes the `InstrView` to a buffer (note: `buffer` is modified in place)
fn dump_to_buffer<'a>(instr_view: &'a InstrView, buffer: &'a mut [u8]) {
    // Write the table of contents to the buffer
    let toc = instr_view.get_sizes();

    let new_buffer =
        write_bump(buffer, &toc).expect("error writing Toc to buffer");

    // Write the acutal contents of the `InstrView` to the buffer
    let new_buffer = write_bytes(new_buffer, instr_view.func_name).unwrap();
    let new_buffer = write_bump(new_buffer, instr_view.func_args).unwrap();
    let func_ret_ty = instr_view.func_ret_ty;
    let new_buffer = write_bump(new_buffer, &func_ret_ty).unwrap();
    let new_buffer = write_bytes(new_buffer, instr_view.var_store).unwrap();
    let new_buffer =
        write_bump(new_buffer, instr_view.arg_idxes_store).unwrap();
    let new_buffer =
        write_bump(new_buffer, instr_view.labels_idxes_store).unwrap();
    let new_buffer = write_bytes(new_buffer, instr_view.labels_store).unwrap();
    let new_buffer = write_bytes(new_buffer, instr_view.funcs_store).unwrap();
    write_bump(new_buffer, instr_view.instrs).unwrap();
}

/// Pads a vec till its length is a multiple of 4
pub fn pad_vec(mut vec: Vec<u8>) -> Vec<u8> {
    let remainder = vec.len() % 4;
    if remainder == 0 {
        return vec;
    }

    let padding_size = 4 - remainder;

    vec.extend(std::iter::repeat_n(0, padding_size));

    vec
}

/* -------------------------------------------------------------------------- */
/*                             Reading from buffer                            */
/* -------------------------------------------------------------------------- */

/// Consume `size.len` items from a byte slice,
/// skip the remainder of `size.capacity`
/// elements, and return the items and the rest of the slice.
fn slice_prefix<T: TryFromBytes + Immutable>(
    data: &[u8],
    size: usize,
) -> (&[T], &[u8]) {
    <[T]>::try_ref_from_prefix_with_elems(data, size)
        .expect("Deserialization error in slice_prefix")
}

/// Reads the table of contents from a prefix of the byte buffer
fn read_toc(data: &[u8]) -> (&Toc, &[u8]) {
    let (toc, remaining_buffer) =
        Toc::ref_from_prefix(data).expect("error deserializing ToC");
    (toc, remaining_buffer)
}

/// Get an `InstrView` backed by the data in a byte buffer
pub fn get_instr_view(data: &[u8]) -> InstrView {
    let (toc, buffer) = read_toc(data);

    let (func_name, new_buffer) = slice_prefix::<u8>(buffer, toc.func_name);
    let (func_args, new_buffer) =
        slice_prefix::<FlatFuncArg>(new_buffer, toc.func_args);

    let (func_ret_ty, new_buffer) =
        <FlatType>::try_read_from_prefix(new_buffer)
            .expect("error deserializing func_ret_ty");

    let (var_store, new_buffer) = slice_prefix::<u8>(new_buffer, toc.var_store);

    let (arg_idxes_store, new_buffer) =
        slice_prefix::<I32Pair>(new_buffer, toc.arg_idxes_store);
    let (labels_idxes_store, new_buffer) =
        slice_prefix::<I32Pair>(new_buffer, toc.labels_idxes_store);
    let (labels_store, new_buffer) =
        slice_prefix::<u8>(new_buffer, toc.labels_store);
    let (funcs_store, new_buffer) =
        slice_prefix::<u8>(new_buffer, toc.funcs_store);
    let (instrs, _) = slice_prefix::<FlatInstr>(new_buffer, toc.instrs);

    InstrView {
        func_name,
        func_args,
        func_ret_ty,
        var_store,
        arg_idxes_store,
        labels_idxes_store,
        labels_store,
        funcs_store,
        instrs,
    }
}

/* -------------------------------------------------------------------------- */
/*                                Actual logic                                */
/* -------------------------------------------------------------------------- */

/// Writes a JSON Bril program to a mmap-ed flat Bril file
pub fn json_to_fbril(output_file: String) {
    // Read in the JSON representation of a Bril file from stdin
    let mut buffer = String::new();
    std::io::stdin()
        .read_to_string(&mut buffer)
        .expect("Unable to read from stdin");

    // Parse the JSON into serde_json's `Value` datatype
    let json: serde_json::Value =
        serde_json::from_str(&buffer).expect("Unable to parse malformed JSON");
    let functions = json["functions"]
        .as_array()
        .expect("Expected `functions` to be a JSON array");

    let mut buffer: Vec<u8> = Vec::with_capacity(100000);

    // we only allow 10 functions right now
    let mut sizes_arr: [u64; 10] = [0; 10];

    for (sizes_idx, func) in functions.iter().enumerate() {
        let instr_store: InstrStore = flatten::flatten_instrs(func);

        // Convert an `InstrStore` to an `InstrView`
        let padded_func_name = pad_vec(instr_store.func_name);
        let flat_func_name = padded_func_name.as_slice();
        let flat_func_arg_vec: Vec<FlatFuncArg> = instr_store
            .func_args
            .into_iter()
            .map(|func_arg| func_arg.into())
            .collect();

        let flat_func_args: &[FlatFuncArg] = flat_func_arg_vec.as_slice();
        let flat_func_ret_ty: FlatType = instr_store.func_ret_ty.into();

        let padded_var_store = pad_vec(instr_store.var_store);
        let flat_var_store: &[u8] = padded_var_store.as_slice();
        let flat_arg_idxes_vec: Vec<I32Pair> = instr_store
            .args_idxes_store
            .into_iter()
            .map(|arg_idxes| arg_idxes.into())
            .collect();
        let flat_arg_idxes_store = flat_arg_idxes_vec.as_slice();

        let flat_label_idxes_vec: Vec<I32Pair> = instr_store
            .labels_idxes_store
            .into_iter()
            .map(|lbl_idx| lbl_idx.into())
            .collect();
        let flat_label_idxes = flat_label_idxes_vec.as_slice();

        let padded_labels_store = pad_vec(instr_store.labels_store);
        let flat_labels_store = padded_labels_store.as_slice();

        let padded_funcs_store = pad_vec(instr_store.funcs_store);
        let flat_funcs_store = padded_funcs_store.as_slice();
        let flat_instrs_vec: Vec<FlatInstr> = instr_store
            .instrs
            .into_iter()
            .map(|instr| instr.into())
            .collect();
        let flat_instrs: &[FlatInstr] = flat_instrs_vec.as_slice();
        let instr_view = InstrView {
            func_name: flat_func_name,
            func_args: flat_func_args,
            func_ret_ty: flat_func_ret_ty,
            var_store: flat_var_store,
            arg_idxes_store: flat_arg_idxes_store,
            labels_idxes_store: flat_label_idxes,
            labels_store: flat_labels_store,
            funcs_store: flat_funcs_store,
            instrs: flat_instrs,
        };

        let instr_view_bytes = convert_instr_view_to_bytes(&instr_view);
        buffer.extend_from_slice(&instr_view_bytes);
        sizes_arr[sizes_idx] = instr_view_bytes.len() as u64;
    }

    let header = Header { sizes: sizes_arr };

    // TODO: figure out some appropriate filename + size for the mmapped file
    let mut mmap = mmap_new_file(&output_file, 100000000, true);

    // Write the header (containing the offsets) to the file
    let new_mmap = write_bump(&mut mmap, &header.sizes)
        .expect("error writing offsets to file");
    // Then write the contents of the buffer to the file
    write_bytes(new_mmap, &buffer);

    // Note: we're keeping this around as a sanity check
    let _temp_instr_view = get_instr_view(&buffer);

    println!("succesfully wrote to fbril file!");
}
