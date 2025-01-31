import json
import sys

terminators = ('br', 'jmp', 'ret')

def blocks(function):
  cur_block = (None, [])
  all_blocks = [cur_block]
  for instr in function:
    if "op" in instr:
      cur_block[1].append(instr)
      if(instr["op"] in terminators):
        cur_block = (None, [])
        all_blocks.append(cur_block)
    else:
      # blocks that are empty and have no labels can safely be removed 
      if(len(cur_block[1]) == 0 and cur_block[0] == None):
        all_blocks = all_blocks[:-1]
      cur_block = (instr["label"], [])
      all_blocks.append(cur_block)
  # remove empty last block  
  if(len(cur_block[1]) == 0 and cur_block[0] == None):
        all_blocks = all_blocks[:-1]
  return all_blocks
  

if __name__ == "__main__":
  filename = sys.argv[1]
  with open(filename) as file:
    program = json.load(file)
    all_blocks = blocks(program["functions"][0]["instrs"])
    for (label, contents) in all_blocks:
      print(label)
      print(contents)