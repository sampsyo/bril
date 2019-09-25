import json


def 


if __name__ == '__main__':
    bril = json.load(sys.stdin)
    #lvn(bril, '-p' in sys.argv, '-c' in sys.argv, '-f' in sys.argv)
    json.dump(bril, sys.stdout, indent=2, sort_keys=True)
