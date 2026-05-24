# test alpha tensor 4x4 against numpy matmul for correctness

import numpy as np

def main():
  # [A,B]
  matrices = [
    [[8, 3, 2, 4],[2, 7, 4, 5],[0, 1, 2, 3],[0, 1, 2, 3]],
    [[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12],[13, 14, 15, 16]],
  ]
  A = np.array(matrices[0])
  B = np.array(matrices[1])
  #res_np = np.matmul(A,B)
  res_at = at4x4(A,B)
  #if np.array_equal(res_np, res_at):
  #  print("Arrays equal")
  #  print(res_np)
  #else:
  #  print("Arrays are not equal")
  #  print("np")
  #  print(res_np)
  #  print("at")
  #  print(res_at)



def at4x4(A,B):
  '''Alphatensor Multiply 4-by-4 matrix A with 4-by-4 matrix B, return result C'''

  a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34, a41, a42, a43, a44 = A.ravel()
  b11, b12, b13, b14, b21, b22, b23, b24, b31, b32, b33, b34, b41, b42, b43, b44 = B.ravel()
  h1 = (a11 + a31) * (b11 + b31)
  print("h1")
  print(h1)
  h2 = (a11 - a13 + a31) * (b11 - b13 + b31)
  print("h2")
  print(h2)
  h3 = - a13 * (b11 - b13 + b31 - b33)
  print("h3")
  print(h3)
  h4 = - a33 * - b33
  print("h4")
  print(h4)
  h5 = - a31 * - b13
  print("h5")
  print(h5)
  h6 = (a11 - a13 + a31 - a33) * - b31
  print("h6")
  print(h6)
  h7 = (- a21 + a22 - a23 - a24) * (- b21 + b22 - b23 - b24)
  print("h7")
  print(h7)
  h8 = (- a21 + a22 - a23 - a24 - a41 + a42) * (- b21 + b22 - b23 - b24 - b41 + b42)
  print("h8")
  print(h8)
  h9 = (a11 - a13) * (b11 - b13)
  print("h9")
  print(h9)
  h10 = (- a21 + a22 - a41 + a42) * (- b21 + b22 - b41 + b42)
  print("h10")
  print(h10)
  h11 = (a41 - a42) * (- b23 - b24)
  print("h11")
  print(h11)
  h12 = (- a21 + a22 - a23 - a24 - a41 + a42 - a43 - a44) * (b41 - b42)
  print("h12")
  print(h12)
  h13 = (- a23 - a24) * (- b21 + b22 - b23 - b24 - b41 + b42 - b43 - b44)
  print("h13")
  print(h13)
  h14 = (a11 - a12 + a21 - a22) * (- b12 - b14)
  print("h14")
  print(h14)
  h15 = (- a12 - a14) * - b21
  print("h15")
  print(h15)
  h16 = (a12 + a14 - a21 + a22 + a23 + a24) * (b12 + b14 - b21 + b22 + b23 + b24)
  print("h16")
  print(h16)
  h17 = (a12 + a14 - a21 + a22 + a23 + a24 + a32 + a41 - a42) * (b12 + b14 - b21 + b22 + b23 + b24 + b32 + b41 - b42)
  print("h17")
  print(h17)
  h18 = (a12 - a21 + a22 + a32 + a41 - a42) * (b12 - b21 + b22 + b32 + b41 - b42)
  print("h18")
  print(h18)
  h19 = (a14 + a23 + a24) * (b12 + b14 - b21 + b22 + b23 + b24 + b32 + b34 + b41 - b42 - b43 - b44)
  print("h19")
  print(h19)
  h20 = (a12 + a14 - a21 + a22 + a23 + a24 + a32 + a34 + a41 - a42 - a43 - a44) * (b32 + b41 - b42)
  print("h20")
  print(h20)
  h21 = (a32 + a41 - a42) * (b14 + b23 + b24)
  print("h21")
  print(h21)
  h22 = (a12 + a14 + a22 + a24) * (b12 + b14 + b22 + b24)
  print("h22")
  print(h22)
  h23 = (a12 + a14 + a22 + a24 + a32 - a42) * (b12 + b14 + b22 + b24 + b32 - b42)
  print("h23")
  print(h23)
  h24 = (a14 + a24) * (b12 + b14 + b22 + b24 + b32 + b34 - b42 - b44)
  print("h24")
  print(h24)
  h25 = (a12 + a14 + a22 + a24 + a32 + a34 - a42 - a44) * (b32 - b42)
  print("h25")
  print(h25)
  h26 = (a32 - a42) * (b14 + b24)
  print("h26")
  print(h26)
  h27 = (a34 - a44) * (b34 - b44)
  print("h27")
  print(h27)
  h28 = (a34 - a43 - a44) * (b34 - b43 - b44)
  print("h28")
  print(h28)
  h29 = (a14 + a34) * - b43
  print("h29")
  print(h29)
  h30 = (a13 + a14 + a23 + a24 + a33 + a34 - a43 - a44) * (b14 + b34)
  print("h30")
  print(h30)
  h31 = (a11 - a12 - a13 - a14 + a21 - a22 - a23 - a24 + a31 - a32 - a33 - a34 - a41 + a42 + a43 + a44) * b14
  print("h31")
  print(h31)
  h32 = - a43 * (b13 + b14 + b23 + b24 + b33 + b34 - b43 - b44)
  print("h32")
  print(h32)
  h33 = a14 * (- b21 + b41)
  print("h33")
  print(h33)
  h34 = (a14 - a32) * (- b21 + b41 - b43)
  print("h34")
  print(h34)
  h35 = (a13 + a14 + a23 + a24 - a31 + a32 + a33 + a34 + a41 - a42 - a43 - a44) * (b14 - b32)
  print("h35")
  print(h35)
  h36 = (- a31 + a32 + a33 + a34 + a41 - a42 - a43 - a44) * b32
  print("h36")
  print(h36)
  h37 = (- a12 - a32) * - b23
  print("h37")
  print(h37)
  h38 = (a32 + a34) * (b41 - b43)
  print("h38")
  print(h38)
  h39 = (- a13 - a14 - a23 - a24) * (b32 + b34)
  print("h39")
  print(h39)
  h40 = a32 * (- b21 + b23 + b41 - b43)
  print("h40")
  print(h40)
  h41 = - a21 * (b11 - b12 + b21 - b22)
  print("h41")
  print(h41)
  h42 = (- a21 + a41) * (b11 - b12 - b13 - b14 + b21 - b22 - b23 - b24 + b31 - b32 - b33 - b34 - b41 + b42 + b43 + b44)
  print("h42")
  print(h42)
  h43 = (- a21 + a41 - a43) * (b13 + b14 + b23 + b24 - b31 + b32 + b33 + b34 + b41 - b42 - b43 - b44)
  print("h43")
  print(h43)
  h44 = (a12 + a22 + a32 - a42) * (b12 + b22 + b32 - b42)
  print("h44")
  print(h44)
  h45 = (- a21 + a23 + a41 - a43) * (- b31 + b32 + b33 + b34 + b41 - b42 - b43 - b44)
  print("h45")
  print(h45)
  h46 = (- a31 + a32 + a41 - a42) * (- b12 - b32)
  print("h46")
  print(h46)
  h47 = (a41 - a43) * (- b13 - b14 - b23 - b24)
  print("h47")
  print(h47)
  h48 = (- a43 - a44) * (- b43 - b44)
  print("h48")
  print(h48)
  h49 = - a23 * (- b31 + b32 + b41 - b42)
  print("h49")
  print(h49)
  c11 = h1 - h2 - h5 + h9 + h15 + h33
  print("c11")
  print(c11)
  c12 = - h15 - h16 + h17 - h18 - h21 + h22 - h23 + h26 - h33 - h41 + h44 + h49
  print("c12")
  print(c12)
  c13 = h2 + h5 + h6 - h9 - h29 - h33 + h34 + h38
  print("c13")
  print(c13)
  c14 = - h16 + h17 - h20 - h21 + h22 - h23 + h25 + h26 - h29 - h32 - h33 + h34 + h38 - h41 + h42 + h43
  print("c14")
  print(c14)
  c21 = - h7 + h8 - h10 + h11 - h14 + h15 + h16 - h17 + h18 + h21 - h31 + h33 - h35 - h36
  print("c21")
  print(c21)
  c22 = h7 - h8 + h10 - h11 - h15 - h16 + h17 - h18 - h21 + h22 - h23 + h26 - h33 + h44
  print("c22")
  print(c22)
  c23 = - h7 + h8 + h11 + h12 - h16 + h17 - h20 - h21 - h29 - h33 + h34 + h36 + h38 + h46
  print("c23")
  print(c23)
  c24 = - h7 + h8 + h11 + h12 - h16 + h17 - h20 - h21 + h22 - h23 + h25 + h26 - h29 - h33 + h34 + h38
  print("c24")
  print(c24)
  c31 = h1 - h2 + h3 - h5 + h33 - h34 + h37 - h40
  print("c31")
  print(c31)
  c32 = h17 - h18 - h19 - h21 - h23 + h24 + h26 - h33 + h34 - h37 + h40 - h43 + h44 + h45 - h47 + h49
  print("c32")
  print(c32)
  c33 = h4 + h5 - h29 - h33 + h34 + h40
  print("c33")
  print(c33)
  c34 = - h21 + h26 - h27 + h28 - h29 - h32 - h33 + h34 + h40 - h47
  print("c34")
  print(c34)
  c41 = h8 - h10 + h11 - h13 + h17 - h18 - h19 - h21 + h31 - h33 + h34 + h35 + h36 - h37 - h39 + h40
  print("c41")
  print(c41)
  c42 = - h8 + h10 - h11 + h13 - h17 + h18 + h19 + h21 + h23 - h24 - h26 + h33 - h34 + h37 - h40 - h44
  print("c42")
  print(c42)
  c43 = h11 + h21 - h28 + h29 + h30 + h33 - h34 - h35 - h36 + h39 - h40 + h48
  print("c43")
  print(c43)
  c44 = h11 + h21 - h26 + h27 - h28 + h29 + h33 - h34 - h40 + h48
  print("c44")
  print(c44)
  C = np.array([c11, c12, c13, c14, c21, c22, c23, c24, c31, c32, c33, c34, c41, c42, c43, c44]).reshape(4, 4).T
  return C




if __name__ == "__main__":
  main()


