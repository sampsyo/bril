import os, subprocess, time, argparse, re

parser = argparse.ArgumentParser(description='Time brili execution')
parser.add_argument('--prog', default='/Users/katyvoor/cs6120/p2/bril/vvadd.bril', help='Path to bril file we want to time')
parser.add_argument('--brili', default='/Users/katyvoor/cs6120/p2/bril/bril-ts/', help='Path to directory with brili.ts')
args = parser.parse_args()

# number of times to average of runs
num_runs = 10
os.chdir(args.brili)
cmd = 'node brili.js  < ../examples/test1.json'


sum = 0.0

for i in range(num_runs):

  start_time = time.time()
  os.system("node brili.js  < ../examples/test1.json")
  print("--- %s seconds ---" % (time.time() - start_time))


# average
avg = sum / num_runs
print ('avg time(ms) {0}'.format(avg))