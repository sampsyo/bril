import json
import sys
import graphviz

# CFG class
class Node:
    def __init__(self, name, current_instrs=None, successor_blocks=None):
        self.name = name
        self.current_instrs = current_instrs if current_instrs is not None else []
        self.successor_blocks = successor_blocks

    def __repr__(self):
        return f"Node({self.name}, instrs={self.current_instrs}, succ={self.successor_blocks})"
    
def form_blocks(prog):
    # Store the blocks in a dict for easy lookup.
    blocks = {}
    # Block counter for labeling.
    block_counter = 0
    # Default block name (in case of no label, not perfect because labels could overlap)
    block_name = f"block_{block_counter}"
    # Current block. 
    cur = []

    for function in prog['functions']:
        for instr in function['instrs']:
            # Start new block for instr that is target of branch, jump.
            if (instr.get('label') is not None):

                # Append previous block (does this handle empty blocks well)
                blocks[block_name] = cur

                # Create new block.
                cur = []

                # Override default block name and make it the label
                block_name = instr['label']

                # Skip label to remove from set of instructions.
                continue
            cur.append(instr)
            # Terminate current block if we hit a branch, jump instr, or ret.
            if (instr['op'] in ('br', 'jmp', 'ret')):
                # Make current instr last instr of current block
                blocks[block_name] = cur
                
                # Adjust counter
                block_counter += 1

                # Start new block with default name
                block_name = f"block_{block_counter}"

                cur = []
            
        # Append the final block.
        blocks[block_name] = cur

    return blocks
                
def connect_blocks(blocks):
    # Initialize list to hold the cfg nodes.
    cfg = {}

    # Unpack names of all the blocks.
    block_names = list(blocks.keys())

    # Loop through blocks and assign successors
    for i, (block_name, instrs) in enumerate(blocks.items()):
        

        # Initialize list to hold successors of current block
        successors = []

        # If the last instruction in the block is a branch, jump...
        if instrs[-1].get("labels") is not None:
            # Add whole list of successor blocks
            successors.extend(instrs[-1]["labels"])
        
        # Or simply append next block (unless last block)
        elif (i + 1) < len(block_names):
            successors.append(block_names[i + 1])
    
        # Create block node
        cfg[block_name] = Node(block_name, instrs, successors)

    return cfg

def displayCfg(cfg):
    # Initialize the graph
    graph = graphviz.Digraph("png")

    # Fill the boxes with the instructions
    for block_name, content in cfg.items():
        label = f"{block_name}\\n" + "\\n".join(
        [instr['op'] for instr in content.current_instrs]
        )

        graph.node(block_name, label, shape="box", style="filled",
                    fillcolor="lightgrey")
        
    # Add the edges between the blocks
    for block_name, content in cfg.items():
        for succ in content.successor_blocks:
            graph.edge(block_name, succ)

    # Render the graph
    output_file = "cfg_output.png"
    graph.render(output_file, format="png", cleanup=True)
    
    print(f"Control Flow Graph saved as {output_file}")  # Confirmation message 

def mycfg():
    print("r")
    prog = json.load(sys.stdin)
    print("running")
    print(prog)
    print()
    print()
    print("running1")
    # I leave the labels as instructions
    blocks = form_blocks(prog)
    print("running2")
    for block in blocks.values():
        print(block)
    print("running3")
    cfg = connect_blocks(blocks)
    print("running4")

    print()
    print()
    for key, value in cfg.items():
        print(f"{key}: {value}")
    
    displayCfg(cfg)


if __name__ == '__main__':
    mycfg()