import numpy as np

def pop_matrix(n): #n is number of points, returns a n-2*n-2 matrix
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

def pop_result(points): #returns product vector
  b = np.zeros(len(points)-2)
  for i in range(len(points) - 2):
    h = points[i+2, 0] - points[i+1, 0]
    b[i] = (points[i,1] + 2 * points[i+1, 1]+points[i+2,1]) * 6 / (h * h)
  return b


def cubic_interpolation(points):
  result = pop_result(points) #vector of values, used for matrix solving
  matrix = pop_matrix(len(points)) #matrix used for solving, is derived from cubic equation stuff
  m_vals = np.zeros(len(points))
  m_vals[1:len(points)-1] = np.linalg.solve(matrix, result) #solves for 2nd derivative values at each known point
  polynomials = np.zeros((len(points)-1, 4))
  for i in range(len(polynomials)): # solves coeficcients of polynomials using m_vals
    h = points[i+1, 0] - points[i, 0]
    a = (m_vals[i+1]-m_vals[i])/(6 * h)
    b = m_vals[i]/2
    c = (points[i+1,1]-points[i,1])/h - (m_vals[i+1]+2 * m_vals[i]) * h / 6
    d = points[i, 1]
    polynomials[i] = np.array([d, c, b, a])
  return polynomials
  
  
  
points = np.random.rand(10, 2) * 10
print(points)
polynomials = cubic_interpolation(points)


  