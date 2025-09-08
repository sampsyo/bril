def basic_blocks(instrs):
    blocks = [[]]
    for inst in instrs:
        if "label" in inst and len(blocks[-1]) > 0:
            blocks.append([])
        blocks[-1].append(inst)
        if "label" not in inst and inst["op"] in ("br, jmp"):
            blocks.append([])
    return [b for b in blocks if len(b) > 0]

def block_name(block):
    if "label" in block[0]:
        return block[0]["label"]
    else:
        return "_entry"

def cfg(blocks):
    graph = {}
    for i, b in enumerate(blocks):
        if "op" in b[-1] and b[-1]["op"] in ("br", "jmp"):
            graph[block_name(b)] = b[-1]["labels"]
        elif len(blocks) > i+1:
            graph[block_name(b)] = [block_name(blocks[i+1])]
        else:
            graph[block_name(b)] = []
    return graph

def all_cfgs(full_bril):
    ret = {}
    for f in full_bril["functions"]:
        ret[f["name"]] = cfg(basic_blocks(f["instrs"]))
    return ret

def call_graph(full_bril):
    ret = {}
    for f in full_bril["functions"]:
        ret[f["name"]] = set()
        for i in f["instrs"]:
            if "op" in i and i["op"] == "call":
                ret[f["name"]].update(i["funcs"])
    return ret