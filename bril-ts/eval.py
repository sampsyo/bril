import os, subprocess, time, argparse, re

parser = argparse.ArgumentParser(description='Time brili execution')
parser.add_argument('--prog', default='/Users/katyvoor/cs6120/p2/bril/vvadd.bril', help='Path to bril file we want to time')
parser.add_argument('--brili', default='/Users/katyvoor/cs6120/p2/bril/bril-ts/', help='Path to directory with brili.ts')
args = parser.parse_args()

# number of times to average of runs
num_runs = 10
os.chdir(args.brili)

sum = 0.0

for i in range(num_runs):

  start_time = time.time()
  os.system("node brili.js  < ../examples/test1.json")
  print("--- %s seconds ---" % (time.time() - start_time))
  sum = sum + time.time() - start_time

# average
avg1 = sum / num_runs
print ('avg time(ms) {0}'.format(avg1))

sum = 0.0

for i in range(num_runs):

  start_time = time.time()
  os.system("node brili.js  < ../examples/test.json")
  print("--- %s seconds ---" % (time.time() - start_time))
  sum = sum + time.time() - start_time

# average
avg = sum / num_runs
print("serial")
print ('avg time(ms) {0}'.format(avg1))
print("vector")
print ('avg time(ms) {0}'.format(avg))

# importing the required module 
import matplotlib.pyplot as plt 
  
# x axis values 
x = [1,2,3] 
# corresponding y axis values 
y = [2,4,1] 
  
# plotting the points  
plt.plot(x, y) 
  
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 
  
# giving a title to my graph 
plt.title('My first graph!') 
  
# function to show the plot 
plt.show() 