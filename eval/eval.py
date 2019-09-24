import os, subprocess, time, argparse, re

parser = argparse.ArgumentParser(description='Time brili execution')
parser.add_argument('--prog', default='/home/bedoukip/cs6120/bril/eval/proj1/vecadd.bril', help='Path to bril file we want to time')
parser.add_argument('--brili', default='/home/bedoukip/cs6120/bril/bril-ts', help='Path to directory with brili.ts')
args = parser.parse_args()

# number of times to average of runs
num_runs = 5
os.chdir(args.brili)
cmd = 'bril2json < {0} | brili'.format(args.prog)

sum = 0.0

for i in range(num_runs):

  # run cmd
  result = subprocess.check_output(cmd, shell=True)
  print(result)
  def get_time(outlog):
    float_regex = re.compile('(brili: )([+-]?([0-9]*[.])?[0-9]+)')                   
    float_match = float_regex.search(outlog)
    if (float_match):
        runtime = float(float_match.group(2))
        return runtime

  runtime = get_time(result)
  sum += runtime
  #print(runtime)


# average
avg = sum / num_runs
print ('avg time(ms) {0}'.format(avg))
