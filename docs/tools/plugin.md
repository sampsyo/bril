Syntax Plugin for Text Editors
==============================

## Visual Studio Code

You can install the [Visual Studio Code extension](https://marketplace.visualstudio.com/items?itemName=EthanUppal.bril), made by Ethan Uppal.
Follow the instructions in the extension README.

## (Neo)Vim

There is a [Vim][] syntax highlighting plugin for Bril's text format available in `bril-vim`. You can use it with a Vim plugin manager. For example, if you use [vim-plug][], you can add this to your `.vimrc`:

    Plug 'sampsyo/bril', { 'for': 'bril', 'rtp': 'bril-vim' }

You can read [more about the plugin][blog], which is originally by Edwin Peguero.

If you're using Neovim, Ethan Uppal made a [simple wrapper around 
`bril-vim`][bril.nvim] for the [Lazy](https://lazy.folke.io) package manager, 
supporting LSP in  addition to the syntax highlighting. Simply add the following
[plugin spec](https://lazy.folke.io/spec):

```lua
{ "ethanuppal/bril.nvim" }
```

[vim]: https://www.vim.org
[blog]: https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/vim-syntax-highlighting/
[vim-plug]: https://github.com/junegunn/vim-plug
[bril.nvim]: https://github.com/ethanuppal/bril.nvim
