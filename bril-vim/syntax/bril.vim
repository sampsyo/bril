"Mechanisms for factoring out regexps and combining regexps from syntax group/region definitions

"Argument regexps expected as a string literal in magic mode
function! s:DefSynGroup(name, regexp, options)
  execute 'syntax match ' . a:name . ' ' . '"' . '\v' . a:regexp . '"' . ' ' . a:options
endfunction

function! s:DefSynRegion(name, startRegexp, endRegexp, options)
  execute 'syntax region ' . a:name . ' ' . 'start=' . '"\v' . a:startRegexp . '"' . ' end=' . '"\v'. a:endRegexp . '"' . ' ' . a:options
endfunction

function! s:ConcatRegexp(...)
  let r = ''
  for arg in a:000
    let r = r . '('.arg.')'
  endfor
  return r
endfunction

" ---------------------------
" --- Basic syntax groups ---
" ---------------------------
" Identifiers
let identifierRegexp = '[a-zA-Z_%]%([a-zA-Z0-9_%]|\.[a-zA-Z0-9_%])*'
call s:DefSynGroup('brilVariable', identifierRegexp, 'contained')

" Function names: @ident
let funcRegexp = s:ConcatRegexp('\@', identifierRegexp)
call s:DefSynGroup('brilFuncName', funcRegexp, 'contained')

" Types
" NOTE: this overlaps with identifiers, but is given higher precedence when declared afterwards.
let typeRegexp = '<%(int|bool|float|char|any|void|ptr)>'
let anyTypeExprRegexp = '%(ptr\s*\<\s*)*' . identifierRegexp . '%(\s*\>)*'
call s:DefSynGroup('brilType', typeRegexp, 'contained')

" Keywords
syntax keyword brilStructKw struct contained
syntax keyword brilImportKw from import as contained

" Value & Constant Operations
syntax keyword brilValueOp contained
  \ const
  \ id
  \ add
  \ mul
  \ sub
  \ div
  \ eq
  \ lt
  \ gt
  \ le
  \ ge
  \ not
  \ and
  \ or
  \ call
  \ fadd
  \ fmul
  \ fsub
  \ fdiv
  \ feq
  \ flt
  \ fle
  \ fgt
  \ fge
  \ ceq
  \ clt
  \ cle
  \ cgt
  \ cge
  \ char2int
  \ int2char
  \ alloc
  \ load
  \ ptradd
  \ getmbr
  \ isnull
  \ float2bits
  \ bits2float
  \ phi
  \ get
  \ undef

" Effect Operations
let brilEffectOpRegexp = '<%(jmp|br|call|ret|print|nop|store|free|speculate|guard|commit|set)>'
call s:DefSynGroup('brilEffectOp', brilEffectOpRegexp, 'contained')

" Comments
syntax match brilComment "\v#.*$"

" Literals
syntax match brilNumber "\v(\w)@<![+-]?\d+>" contained
syntax match brilFloat "\v(\w)@<![+-]?%((\d+\.\d*|\.\d+)%([eE][+-]?\d+)?|\d+[eE][+-]?\d+)(\w)@!" contained
syntax keyword brilBool true false contained
syntax keyword brilNull nullptr contained
syntax match brilChar "\v'([^'\\]|\\[0abtnvfr\\'"])'" contained
syntax region brilString start=+"+ skip=+\\\\\|\\"+ end=+"+ oneline contained
syntax cluster brilValue contains=brilNumber,brilFloat,brilBool,brilNull,brilChar

" Labels
let labelRegexp = '%([a-zA-Z0-9_%]@<!\.' . identifierRegexp . ')|%(^\s*\zs' . identifierRegexp . '\ze\s*:\s*%($|#))'
call s:DefSynGroup('brilLabel', labelRegexp, 'contained')


" ---------------------------------
" --- Instruction Syntax Groups ---
" ---------------------------------
" Effect Instructions
call s:DefSynRegion('brilEffectInstr', brilEffectOpRegexp, ';', 'oneline contained contains=brilVariable,brilFuncName,brilLabel,brilEffectOp')

" Branch & Guard Instructions
" Highlights the boolean condition variable immediately following 'br' or 'guard'
let brilCondVariableRegexp = '%(<%(br|guard)>\s+)@<=' . identifierRegexp
call s:DefSynGroup('brilCondVariable', brilCondVariableRegexp, 'contained')
call s:DefSynRegion('brilBranchInstr', '<%(br|guard)>', ';', 'oneline contained contains=brilCondVariable,brilVariable,brilLabel,brilEffectOp')

" Value & Constant Instructions
let brilValueLhsRegexp = s:ConcatRegexp(identifierRegexp, '%(\s*:\s*' . anyTypeExprRegexp . ')?\s*\=')
call s:DefSynRegion('brilValueInstr', brilValueLhsRegexp, ';', 'oneline contained contains=brilVariable,brilType,brilFuncName,brilLabel,brilValueOp,@brilValue')


" -------------------------------
" --- Top Level Syntax Groups ---
" -------------------------------
" Imports
call s:DefSynRegion('brilImport', '<from>', ';', 'oneline contains=brilImportKw,brilString,brilFuncName,brilComment')

" Structs
let brilStructFieldRegexp = s:ConcatRegexp(identifierRegexp, '\s*:\s*', anyTypeExprRegexp)
call s:DefSynRegion('brilStructField', brilStructFieldRegexp, ';', 'oneline contained contains=brilVariable,brilType')
let brilStructStartRegexp = s:ConcatRegexp('<struct>\s+', identifierRegexp, '\s*\=\s*\{')
call s:DefSynRegion('brilStruct', brilStructStartRegexp, '}', 'contains=brilStructKw,brilVariable,brilType,brilStructField,brilComment')

" Function argument list
call s:DefSynRegion('brilFunArgs', '\(', '\)', 'contained oneline contains=brilVariable,brilType')

" Functions
let brilFunStartRegexp = s:ConcatRegexp(funcRegexp, '%(\s*\([^)]*\))?', '%(\s*:\s*' . anyTypeExprRegexp . ')?', '\s*\{')
call s:DefSynRegion('brilFun', brilFunStartRegexp, '}', 'contains=brilComment,@brilValue,brilFuncName,brilFunArgs,brilType,brilValueInstr,brilBranchInstr,brilEffectInstr,brilLabel')

"Infinite look-behind: not efficient, but probably ok for most people nowadays.
syn sync fromstart

" ----------------------------------
" --- Highlight syntactic groups ---
" ----------------------------------
highlight default link brilComment Comment
highlight default link brilLabel Label
highlight default link brilVariable Identifier
highlight default link brilFuncName Function
highlight default link brilType Type
highlight default link brilStructKw Structure
highlight default link brilImportKw Include
highlight default link brilValueOp Operator
highlight default link brilEffectOp Keyword
highlight default link brilNumber Number
highlight default link brilFloat Float
highlight default link brilBool Boolean
highlight default link brilNull Constant
highlight default link brilChar Character
highlight default link brilString String
highlight default link brilCondVariable Boolean


" Useful debugging mapping
noremap -h <esc>:echo map(synstack(line('.'), col('.')), 'synIDattr(v:val, "name")')<cr>  
