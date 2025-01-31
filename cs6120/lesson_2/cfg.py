import sys
import blocks
    
def change_labels(all_blocks):
  new_name = "label "
  new_num = 0
  all_labels = [block[0] for block in all_blocks]
  for i in range(len(all_labels)):
    if(all_labels[i] == None):
      while((new_name + str(new_num)) in all_labels):
        new_num += 1
      all_labels[i] = new_name + str(new_num)
  
  for i in range(len(all_blocks)):
    all_blocks[i] = (all_labels[i], all_blocks[i][1])
  return all_blocks

def build_cfg(all_blocks):
  successors = {"entry": [], "exit": []}
  for block in all_blocks:
    successors[block[0]] = []
  # dummy entry block
  successors["entry"] = [all_blocks[0][0]]
  
  for i in range(len(all_blocks)):
    # last entry is ret -> successor = exit
    # last entry is br -> successors = br targets
    # last entry is jmp -> successor = jmp
    # last entry is nothing -> successor = next block (or exit if last block)
    last_entry = all_blocks[i][1][-1]
    if(last_entry["op"] == "br" or last_entry["op"] == "jmp"):
      successors[all_blocks[i][0]] = last_entry["labels"]
    elif(last_entry["op"] == "ret"):
      successors[all_blocks[i][0]] = ["exit"]
    else:
      if(i == len(all_blocks) - 1):
        successors[all_blocks[i][0]] = ["exit"]
      else:
        successors[all_blocks[i][0]] = [all_blocks[i + 1][0]]  
  return successors

def remove_orphans(all_blocks, cfg):
  visited = {}
  for block in all_blocks:
    visited[block[0]] = False
    
  for entry in cfg:
    for successor in cfg[entry]:
      visited[successor] = True 
  
  parsed_blocks = []
  for block in all_blocks:
    if visited[block[0]]:
      parsed_blocks.append(block)
  return parsed_blocks
if __name__ == "__main__":
  filename = sys.argv[1]
  all_blocks = change_labels(blocks.blocks(filename))
  cfg = build_cfg(all_blocks)
  print(cfg)
  # print(remove_orphans(all_blocks, cfg))