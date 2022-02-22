from form_blocks import block_name
class Cfg:
    def __init__(self, blocks):
        self.blocks = blocks
        self.succ = {}
        self.pred = {}

        lbl_to_block = {}
        for block in blocks:
            lbl = block_name(block)
            lbl_to_block[lbl] = block

        # Setup first block pred
        self.create_pred(None, blocks[0])
        for i in range(len(blocks)):
            block = blocks[i]
            last = block[-1]
            if 'op' in last and last['op'] == 'jmp':
                lbl = last['labels'][0]
                next_block = lbl_to_block[lbl]
                self.create_succ(block, next_block)
                self.create_pred(block, next_block)
            elif 'op' in last and last['op'] == 'br':
                true_lbl = last['labels'][0]
                true_block = lbl_to_block[true_lbl]
                self.create_succ(block, true_block)
                self.create_pred(block, true_block)

                false_lbl = last['labels'][1]
                false_block = lbl_to_block[false_lbl]
                self.create_succ(block, false_block)
                self.create_pred(block, false_block)
            else:
                more_blocks = i < len(blocks) - 1
                next_block = blocks[i+1] if more_blocks else None
                self.create_succ(block, next_block)
                self.create_pred(block, next_block)

    def create_succ(self, block, next_block):
        name = block_name(block)
        if name in self.succ.keys() and next_block != None:
            lst = self.succ[name]
            lst.append(next_block)
        elif next_block != None:
            self.succ[name] = [next_block]
        else:
            self.succ[name] = []

    def create_pred(self, block, next_block):
        if next_block is None:
            return

        next_block_name = block_name(next_block)
        if block is None:
            self.pred[next_block_name] = []
            return
        if next_block_name in self.pred.keys():
            lst = self.pred[next_block_name]
            lst.append(block)
        else :
            self.pred[next_block_name] = [block]

    def get_succ(self, name):
        return self.succ[name]

    def get_pred(self, name):
        return self.pred[name]

    def remove_vertex(self, v):
        # Remove from blocks
        idx = -1
        for i, block in enumerate(self.blocks):
            if block[0]['label'] == v:
                idx = i
                break
        if idx != -1:
            self.blocks.pop(idx)

        # Remove from all succ
        idx = -1
        for name, succs in self.succ.items():
            for i, succ in enumerate(succs):
                if succ[0]['label'] == v:
                    idx = i
                    break
            if idx != -1:
                succs.pop(idx)
                break

        # Remove from all pred
        idx = -1
        for name, preds in self.pred.items():
            for i, pred in enumerate(preds):
                if pred[0]['label'] == v:
                    idx = i
                    break
            if idx != -1:
                preds.pop(idx)
                break

    def get_exits(self):
        exits = []
        for block in self.blocks:
            if len(self.get_succ(block_name(block))) == 0:
                exits.append(block)
        return exits

    def post_order(self, block):
        """
        Returns the post order traversal of the cfg
        starting with block.
        """
        return self.post_order_aux(block, [])

    def post_order_aux(self, block, visited):
        """
        Returns the post order traversel of this cfg
        where we have already seen all verticies in
        visited
        """
        order = []
        succs = self.get_succ(block_name(block))
        for succ in succs:
            if block_name(succ) not in visited:
                visited.append(block_name(block))
                order.extend(self.post_order_aux(succ, visited))
        order.append(block)
        return order

    def reverse_post_order(self, block):
        """
        Returns the post order traversal of the cfg
        with the first block in the post order
        starting with block.
        """
        return list(reversed(self.post_order(block)))

