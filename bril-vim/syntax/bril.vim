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
let identifierRegexp = '(\a|_)\w*'
call s:DefSynGroup('brilVariable', identifierRegexp, 'contained')

" Tokens
let funcRegexp = s:ConcatRegexp('\@', identifierRegexp, '(\(.*\))?')
call s:DefSynGroup('brilFuncName', funcRegexp, 'contained')

" Types
" NOTE: this overlaps with identifiers, but is given higher precedence when declared afterwards.
let typeRegexp = 'int|bool'
call s:DefSynGroup('brilType', typeRegexp, 'contained')

syntax keyword brilValueOp contained
  \ id
  \ const
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

let brilEffectOpRegexp =  'jmp|br|ret|print|nop|speculate|guard|commit'
call s:DefSynGroup('brilEffectOp', brilEffectOpRegexp, 'contained')

" Comments
syntax match brilComment "\#.*$"

" Literals
syntax match brilNumber "\v<-?\d+>"
syntax match brilBool "true|false"
syntax cluster brilValue contains=brilNumber,brilBool

" Labels
let labelRegexp = s:ConcatRegexp(identifierRegexp, '\ze\s*:\s*')
call s:DefSynGroup('brilLabel', labelRegexp, 'contained')


" ---------------------------------
" --- Instruction Syntax Groups ---
" ---------------------------------
" Effect Instructions
call s:DefSynRegion('brilEffectInstr', brilEffectOpRegexp, ';', 'oneline contained contains=brilVariable,brilEffectOp')


" Branch Instructions
" NOTE: this regexp just captures the 'first of three' identifier in a semicolon
" terminated, space-separated list. This is only interesting in the context of
" branching instructions, so it illustrates how the 'contained' parameter can
" be useful.
let brilCondVariableRegexp = s:ConcatRegexp(identifierRegexp, '\s*\ze', identifierRegexp,  '\s*', identifierRegexp, '\s*;')
call s:DefSynGroup('brilCondVariable', brilCondVariableRegexp, 'contained')
call s:DefSynRegion('brilBranchInstr', 'br', ';', 'oneline contained contains=brilCondVariable,brilVariable,brilEffectOp')


" Value Instructions
let brilTypedVarRegexp = s:ConcatRegexp(identifierRegexp, s:ConcatRegexp('\s*:\s*', typeRegexp))
call s:DefSynRegion('brilValueInstr', brilTypedVarRegexp, ';', 'oneline contained contains=brilVariable,brilType,brilValueOp,@brilValue')


" ------------------------------
" --- Top Level Syntax Group ---
" ------------------------------
" Functions
let brilFunStartRegexp = s:ConcatRegexp(funcRegexp, '\s*\{')
call s:DefSynRegion('brilFun', brilFunStartRegexp, '}', 'contains=brilComment,@brilValue,brilFuncName,brilValueInstr,brilBranchInstr,brilEffectInstr,brilLabel')

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
highlight default link brilValueOp Operator
highlight default link brilEffectOp Keyword
highlight default link brilNumber Number
highlight default link brilBool Boolean
highlight default link brilCondVariable Boolean


" Useful debugging mapping
noremap -h <esc>:echo map(synstack(line('.'), col('.')), 'synIDattr(v:val, "name")')<cr>  
