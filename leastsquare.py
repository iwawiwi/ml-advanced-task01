

"""
# Equations:
2x + 5y = 10
3x + 4y = 15
"""

import numpy as np

# 2 lines intersect at single point in 2D
# a = [[2.0, 5.0], [3.0, 4.0]]
# b = [10.0, 15.0]

# 2 lines parallel in 2D
# a = [[2.0, 5.0], [4.0, 10.0]]
# b = [10.0, 10.0]

# 2 lines coincide in 2D
# a = [[2.0, 5.0], [2.0, 5.0]]
# b = [10.0, 10.0]

# 3 lines intersect at single point in 2D
# a = [[2.0, 5.0], [3.0, 4.0], [0.0, 1.0]]
# b = [10.0, 15.0, 0.0]

# 3 lines intersect at 3 points in 2D
a = [[2.0, 5.0], [3.0, 4.0], [1.0, -1.5]]
b = [10.0, 15.0, 1.0]

# # 3 lines, 2 of them are parallel in 2D
# a = [[2.0, 5.0], [4.0, 10.0], [1.0, -1.5]]
# b = [10.0, 10.0, 1.0]

# 1 lines in 2D
# a = [[2.0, 5.0]]
# b = [10.0]


print 'A: ', np.shape(a)
print 'b: ', np.shape(b)

# create lines for visualization
x = np.linspace(0.0, 9.0, 100, True)
m1 = a[0][0]/a[0][1]
c1 = b[0]/a[0][1]
print 'm1: ', m1
y1 = c1 -  m1 * x
print 'y1: ', y1

m2 = a[1][0]/a[1][1]
c2 = b[1]/a[1][1]
print 'm2: ', m2
y2 = c2 -  m2 * x
print 'y2: ', y2

m3 = a[2][0]/a[2][1]
c3 = b[2]/a[2][1]
print 'm3: ', m3
y3 = c3 -  m3 * x
print 'y3: ', y3

ls = np.linalg.lstsq(a, b)
print 'ls: ', ls[0]

# sol = np.linalg.solve(a, b)
# print 'sol: ', sol

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y1, 'g-', linewidth=2, label='y1')
ax.plot(x, y2, 'b-', linewidth=2, label='y2')
ax.plot(x, y3, 'r-', linewidth=2, label='y3')
ax.scatter(ls[0][0], ls[0][1], s=50, facecolors='none', edgecolors='m', linewidths=2, label='solution')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Least Square Solution')
plt.legend()
plt.grid()
plt.show()

