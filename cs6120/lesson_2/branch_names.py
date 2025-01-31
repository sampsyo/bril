import json
import sys

if __name__ == "__main__":
  filename = sys.argv[1]
  with open(filename) as file:
    program = json.load(file)
    for function in program["functions"]:
      print(function["name"])