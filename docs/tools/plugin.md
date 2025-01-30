Syntax Plugin for Text Editors
==============================

## VSCode

You can install the [VSCode extension](https://marketplace.visualstudio.com/items?itemName=EthanUppal.bril), made by Ethan Uppal.

## (Neo)Vim

There is a [Vim][] syntax highlighting plugin for Bril's text format available in `bril-vim`. You can use it with a Vim plugin manager. For example, if you use [vim-plug][], you can add this to your `.vimrc`:

    Plug 'sampsyo/bril', { 'for': 'bril', 'rtp': 'bril-vim' }

You can read [more about the plugin][blog], which is originally by Edwin Peguero.

If you're using Neovim, Ethan Uppal made a simple wrapper around `bril-vim` for
the [Lazy](https://lazy.folke.io) package manager, supporting LSP in addition to
the syntax highlighting (link [here](https://github.com/ethanuppal/bril.nvim/tree/main)):

```lua
{ "ethanuppal/bril.nvim" }
```

[vim]: https://www.vim.org
[blog]: https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/vim-syntax-highlighting/
[vim-plug]: https://github.com/junegunn/vim-plug
