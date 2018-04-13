
# coding: utf-8

# In[103]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.integrate as spi
from tqdm import tqdm
from matplotlib import animation

# # Lorenz Attractor

# In[104]:


#integration step
def lorenzDiff(position, t):
    global rho, b, sigma
    x, y, z = position
    dx = sigma*(y-x)
    dy = x*(rho-z) - y
    dz = x*y - b*z
    return [dx, dy, dz]


# In[146]:


#initial state
pInitial = [5, 5, 5]

#parameters
rho = 28
b = 8/3
sigma = 10

#simulation parameters
dt = 0.01
tStart = 0
tEnd = 20
time = np.arange(tStart, tEnd, dt)

#get trajectory using integration
traj = spi.odeint(lorenzDiff, pInitial, time)

x = traj[:,0]
y = traj[:,1]
z = traj[:,2]

fig = plt.figure(figsize=(12, 9))
ax = fig.gca(projection='3d')
ax.set_xlabel(r"$X$")
ax.set_ylabel(r"$Y$")
ax.set_zlabel(r"$Z$")
ax.plot(x,y,z, "k", linewidth = 1, alpha = 1)

traj2 = spi.odeint(lorenzDiff, [5.01,5,5], time)
# ax.plot(traj2[:,0],traj2[:,1],traj2[:,2], "r--", linewidth = 1, alpha = 0.7)
# fig.savefig("chaosxyz1.png", format = "png", dpi = 300)
# plt.close()

# In[147]:


# plt.plot(time, x, "k", linewidth = 1, label = r"$x = 5$")
# plt.plot(time, traj2[:,0], "r--", linewidth = 1, label = r"$x = 5.01$")
# plt.grid(True)
# plt.legend()
# plt.xlabel(r"$t$")
# plt.ylabel(r"$x(t)$")
# plt.savefig("chaosx1.png", format = "png", dpi = 300)
# plt.close()

ax = mplot3d.Axes3D(fig)
points = [[],[]]
points[0], = ax.plot([x[0]],[y[0]],[z[0]],'k:', markersize=1.5)
points[1], = ax.plot([traj2[0][0]],[traj2[0][1]],[traj2[0][2]], ':', color='red', markersize=1.5)
def update_point(n, x, y, z, point):
    point[0].set_data(np.array([x[0][:n], y[0][:n]]))
    point[0].set_3d_properties(z[0][:n], 'z')
    point[1].set_data(np.array([x[1][:n], y[1][:n]]))
    point[1].set_3d_properties(z[1][:n], 'z')
    # point.set_3d_properties(x[n], 'x')
    # point.set_3d_properties(y[n], 'y')
    return point

ax.plot(x, y, z, color='orange',lw=0.25)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

ani=animation.FuncAnimation(fig, update_point, len(time), fargs=([x, traj2[:,0]], [y, traj2[:,1]], [z, traj2[:,2]], points), blit=False, interval=1)
ani.save(filename='chaos.mp4', fps=120)
