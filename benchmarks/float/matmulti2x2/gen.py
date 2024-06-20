# generate 10 2x2 float matrix multiply programs in bril

from jinja2 import Template
import numpy as np

NUM_EXAMPLES = 12

# Define the template for Bril code
template_str = '''
@main {
  {% for i in range(NUM_EXAMPLES) %}
  a11_{{i}}: float = const {{matrices[2*i][0][0]}};
  a12_{{i}}: float = const {{matrices[2*i][0][1]}};
  a21_{{i}}: float = const {{matrices[2*i][1][0]}};
  a22_{{i}}: float = const {{matrices[2*i][1][1]}};
  b11_{{i}}: float = const {{matrices[2*i+1][0][0]}};
  b12_{{i}}: float = const {{matrices[2*i+1][0][1]}};
  b21_{{i}}: float = const {{matrices[2*i+1][1][0]}};
  b22_{{i}}: float = const {{matrices[2*i+1][1][1]}};

  c11_{{i}}_part1: float = fmul a11_{{i}} b11_{{i}};
  c11_{{i}}_part2: float = fmul a12_{{i}} b21_{{i}};
  c11_{{i}}: float = fadd c11_{{i}}_part1 c11_{{i}}_part2;

  c12_{{i}}_part1: float = fmul a11_{{i}} b12_{{i}};
  c12_{{i}}_part2: float = fmul a12_{{i}} b22_{{i}};
  c12_{{i}}: float = fadd c12_{{i}}_part1 c12_{{i}}_part2;

  c21_{{i}}_part1: float = fmul a21_{{i}} b11_{{i}};
  c21_{{i}}_part2: float = fmul a22_{{i}} b21_{{i}};
  c21_{{i}}: float = fadd c21_{{i}}_part1 c21_{{i}}_part2;

  c22_{{i}}_part1: float = fmul a21_{{i}} b12_{{i}};
  c22_{{i}}_part2: float = fmul a22_{{i}} b22_{{i}};
  c22_{{i}}: float = fadd c22_{{i}}_part1 c22_{{i}}_part2;

  print c11_{{i}};
  print c12_{{i}};
  print c21_{{i}};
  print c22_{{i}};
  {% endfor %}
  ret;
}
'''

# Define the hardcoded matrices
# [[8,3]  * [[2,7]  =  [[28,71]
#  [2,4]]    [4,5]]      20,34]]
matrices = [
    [[8,3],[2,4]],
    [[2,7],[4,5]],
    [[0,1],[2,3]],
    [[0,1],[2,3]],
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    [[9, 10], [11, 12]],
    [[13, 14], [15, 16]],
    [[17, 18], [19, 20]],
    [[21, 22], [23, 24]],
    [[25, 26], [27, 28]],
    [[29, 30], [31, 32]],
    [[33, 34], [35, 36]],
    [[37, 38], [39, 40]],
    [[41, 42], [43, 44]],
    [[45, 46], [47, 48]],
    [[49, 50], [51, 52]],
    [[53, 54], [55, 56]],
    [[57, 58], [59, 60]],
    [[61, 62], [63, 64]],
    [[65, 66], [67, 68]],
    [[69, 70], [71, 72]],
    [[73, 74], [75, 76]],
    [[77, 78], [79, 80]]
]

# Generate reference results using NumPy and write to matmulti2x2.ref
# TODO write to filename first argument to program
with open("matmulti2x2.ref", "w") as ref_file:
    for i in range(NUM_EXAMPLES):
        mat1 = np.array(matrices[2*i])
        mat2 = np.array(matrices[2*i + 1])
        result = np.matmul(mat1, mat2)
        #ref_file.write(f"Reference result for pair {i}:\n")
        #ref_file.write(str(result) + "\n")
        for row in result:
            for element in row:
                #ref_file.write(str(element) + "\n")
                ref_file.write(f"{element:.20f}\n")

# Create a Jinja2 template and render it
template = Template(template_str)
rendered_str = template.render(matrices=matrices)

# Write the rendered Bril code to matmulti2x2.bril
with open("matmulti2x2.bril", "w") as bril_file:
    bril_file.write(rendered_str)
