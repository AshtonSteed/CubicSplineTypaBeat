import numpy as np
import matplotlib.pyplot as plt

def pop_matrix_legacy(n): #n is number of points, returns a n-2*n-2 matrix
  # legacy, only works when distance between points is constant due to assumptions in the origninal derivation
  a = np.zeros((n-2, n-2))
  a[0, 0] = 4
  a[0, 1] = 1
  a[n-3, n-4] = 1
  a[n-3, n-3] = 4
  for i in range(n-4):
    a[i + 1, i]=1
    a[i + 1,i + 1]= 4
    a[i + 1, i+2]=1
  return a

def pop_result_legacy(points): #returns product vector, legacy, worked with above matrix function
  b = np.zeros(len(points)-2)
  for i in range(len(points) - 2):
    h = points[i+2, 0] - points[i+1, 0]
    b[i] = (points[i,1] - 2 * points[i+1, 1]+points[i+2,1]) * 6 / (h * h)
  return b

def pop_matrix(points): #n is number of points, returns a n-2*n-2 matrix
  n = len(points)
  a = np.zeros((n-2, n-2))
  h0 = points[1, 0] - points[0,0]
  h1 = points[2, 0] - points[1,0]
  a[0, 0] = 2*(h0+h1)
  a[0, 1] = h1
  h0 = points[n-2, 0] - points[n-3,0]
  h1 = points[n-1, 0] - points[n-2,0]
  a[n-3, n-4] = h0
  a[n-3, n-3] = 2*(h0+h1)
  for i in range(n-4):
    h0 = points[i+2, 0] - points[i+1,0]
    h1 = points[i+3, 0] - points[i+2,0]
    a[i + 1, i]= h0
    a[i + 1,i + 1]= 2*(h0 + h1)
    a[i + 1, i+2]= h1
  return a

def pop_result(points): #returns product vector
  b = np.zeros(len(points)-2)
  for i in range(len(points) - 2):
    h0 = points[i+1, 0] - points[i, 0]
    h1 = points[i+2, 0] - points[i+1, 0]
    b[i] = 6 * ((points[i+2, 1] - points[i+1, 1])/h1 - (points[i+1, 1] - points[i, 1]) / h0)
  return b


def cubic_interpolation(points):
  result = pop_result(points) #vector of values, used for matrix solving
  matrix = pop_matrix(points) #matrix used for solving, is derived from cubic equation stuff
  m_vals = np.zeros(len(points))
  m_vals[1:len(points)-1] = np.linalg.solve(matrix, result) #solves for 2nd derivative values at each known point
  polynomials = np.zeros((len(points)-1, 4))
  for i in range(len(polynomials)): # solves coeficcients of polynomials using m_vals
    h = points[i+1, 0] - points[i, 0]
    a = (m_vals[i+1]-m_vals[i])/(6 * h)
    b = m_vals[i]/2 
    c = (points[i+1,1]-points[i,1])/h - (m_vals[i+1]+2 * m_vals[i]) / 6 * h
    d = points[i, 1]
    polynomials[i] = np.array([d, c, b, a])
  print("Matrix: \n", matrix)
  print("M Values: \n", np.expand_dims(m_vals, 1))
  print("Result Values: \n", np.expand_dims(result, 1))
  
  return polynomials
  
  
  
points = np.random.rand(5, 2) * 10
print(points)
points = points[points[:, 0].argsort()]
print(points)
#points = np.array([[0,0],[1.5, 0], [2, 1], [3, 1],[4, .5]])
polynomials = cubic_interpolation(points)

for i in range(len(polynomials)):
  poly = np.polynomial.polynomial.Polynomial(polynomials[i], domain=[points[i,0], points[i, 0] + 1],window=[0, 1])
  #print(poly.convert())
  x = np.linspace(points[i, 0], points[i+1, 0], 100)
  #x = np.linspace(0, 10, 100)
  y = poly(x)
  #print(y)
  plt.plot(x, y)
plt.plot(points[:, 0], points[:, 1], '.k')
plt.show()
  