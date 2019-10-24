import os, subprocess, time, argparse, re
import plotly.graph_objects as go
import numpy as np

parser = argparse.ArgumentParser(description='Time brili execution')
parser.add_argument('--prog', default='/Users/katyvoor/cs6120/p2/bril/vvadd.bril', help='Path to bril file we want to time')
parser.add_argument('--brili', default='/Users/katyvoor/cs6120/p2/bril/bril-ts/', help='Path to directory with brili.ts')
args = parser.parse_args()

# number of times to average of runs
num_runs = 30
os.chdir(args.brili)
x0 = []
x1 = []
y0 = []
y1 = []
sum = 0.0
size = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
std0 = []
std1 = []

for p in size:
  optstd = []
  std = []
  sum = 0.0
  for i in range(num_runs):
    start_time = time.time()
    os.system("node brili.js  < ../test/proj2/vvadd{}.json".format(p))
    end = time.time()
    std.append(end - start_time)
    std1.append(end - start_time)
    sum = sum + end - start_time

  st_dev = np.std(std)
  # average
  avg1 = sum / num_runs
  print(avg1)
  x1.append(p)
  y1.append(avg1)

  sum = 0.0
  psum = 0.0
  for i in range(num_runs):
    start_time = time.time()
    os.system("node brili.js  < ../test/proj2/vvadd{}o.json".format(p))
    end = time.time()
    optstd.append(end - start_time)
    std0.append(end - start_time)
    sum = sum + end - start_time
  st_dev = np.std(optstd)
  avg = sum / num_runs
  x0.append(p)
  y0.append(avg)


# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=x0, y=y0,
                    mode='lines+markers',
                    name='lines+markers',text="serial",
                    error_y=dict(
            type='data', # value of error bar given in data coordinates
            array=std1,
            visible=True)))
fig.add_trace(go.Scatter(x=x1, y=y1,
                    mode='lines+markers',
                    name='lines+markers',text="vector",
                     error_y=dict(
            type='data', # value of error bar given in data coordinates
            array=std0,
            visible=True)))

fig.show()