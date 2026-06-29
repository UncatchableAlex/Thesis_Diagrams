import numpy as np
from matplotlib.patches import Ellipse, Rectangle, Polygon
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt



def cov_to_ellipse(cov, n_std=3.0):
    eigvals, eigvecs = np.linalg.eigh(cov)        # ascending order
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]  # descending: major first
    width, height = 2 * n_std * np.sqrt(eigvals)   # full axis lengths
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    return width, height, angle

# get a rotation matrix from roll, pitch, and yaw
def rot_mat(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.radians(rx_deg), np.radians(ry_deg), np.radians(rz_deg)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)]
    ])
    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0,0,1]
    ])
    return Rz @ Rx @ Ry


# spherical coordinates to cartesian
def spherical2cartesian(rho, theta_deg, phi_deg):
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    return np.array([
            rho*np.sin(theta)*np.cos(phi),  # x
            rho*np.sin(theta)*np.sin(phi),  # y
            rho*np.cos(theta)               # z
        ])

def cov3d(sx, sy, sz, rx, ry, rz):
    R = rot_mat(rx,ry,rz)
    S = np.array([
        [sx,0,0],
        [0,sy,0],
        [0,0,sz]
    ])
    return (R@S)@((R@S).T)


class Gaussian:
    def __init__(self, sx,sy,sz, rx, ry, rz, mean, color):
    #def __init__(self, sx,sy,sz, theta, phi, rho, dtheta, dphi, color):

        self.cov = cov3d(sx,sy,sz,rx,ry,rz)
      #  self.mean = spherical2cartesian(rho, dtheta, dphi)
        self.mean = np.array(mean)
        self.color = color

    def proj(self, f, rot, eye):
        W = rot
        x = self.mean
        t = W@(x - eye)
        l = np.sqrt(t[0]*t[0] + t[1]*t[1] + t[2]*t[2])
        J = np.array([
                [f/t[2], 0, -f*t[0]/(t[2]*t[2])],
                [0, f/t[2], -f*t[1]/(t[2]*t[2])],
                [t[0]/l, t[1]/l, t[2]/l]
            ])
        cov = self.cov
        ray_mean = [f*t[0]/t[2], f*t[1]/t[2], l]
        proj_cov = ((J@W)@cov@((J@W).T))[0:2, 0:2]
        width, height, angle = cov_to_ellipse(proj_cov)
        return ray_mean[0:2], width, height, angle, proj_cov
    

class Camera:
    def __init__(self, title, f, xrange, yrange, pyr, eye, ax):
        self.f = f
        self.rot = rot_mat(pyr[0], pyr[1], pyr[2])
        self.eye = np.array(eye)
        self.ax = ax
        ax.cla()
        ax.set_title(title)
        yr = yrange
        xr = xrange
        ax.set_ylim(-yr/2, yr/2)
        ax.set_xlim(-xr/2, xr/2)


    # draw a gaussian
    def draw_gaussian(self, gaussian):
        p = self.rot @ (gaussian.mean - self.eye) # camera coordinates
        if p[2] <= 0:  # behind camera
            return None
        (mx, my), w, h, ang, _ = gaussian.proj(self.f, self.rot, self.eye)
        self.ax.add_patch(Ellipse((mx, my), w, h, angle=ang, color=gaussian.color, alpha=0.5))
        theta = np.deg2rad(ang)
        self.ax.plot(mx, my, 'ko', markersize=3)
        return mx, my, w, h, ang
    

    def to_other_view(self, v, p):

         # lift the v-camera-space projection plane coordinates to world space
        world = (v.rot.T@p) + v.eye
        this_cam = self.rot @ (world - self.eye)

        # ignore points behind the camera
        if this_cam[2] <= 0:
            pass

        screen_ptx = self.f * this_cam[0] / this_cam[2]
        screen_pty = self.f * this_cam[1] / this_cam[2]
        return (screen_ptx, screen_pty)
    
    # view the projection plane defined by another camera object
    def draw_projplane(self, v, gaussians):
        # the coordinates of the corners of v's projection plane
        xr = v.ax.get_xlim()[1] * 2   # recover v's xrange/yrange from its axes
        yr = v.ax.get_ylim()[1] * 2

        # corners in v's camera space (at depth = v.f)
        corners_cam = [
            np.array([-xr/2, -yr/2, v.f]),
            np.array([ xr/2, -yr/2, v.f]),
            np.array([ xr/2,  yr/2, v.f]),
            np.array([-xr/2,  yr/2, v.f]),
        ]

        # draw v's projection plane
        screen_pts = []
        for corner_cam in corners_cam:
            screen_pts.append(self.to_other_view(v, corner_cam))

        # if there are at least 3 points on the screen, draw a polygon
        if len(screen_pts) >= 3:
            poly = Polygon(screen_pts, closed=True, fill=False, edgecolor='violet', linewidth=1.5, linestyle='--')
            self.ax.add_patch(poly)
        
        n=3
        for i in range(1, n):
            ux = (i/n)*xr - xr/2
            uy = (i/n)*yr - yr/2
            # vertical line at x=ux
            p1 = self.to_other_view(v, [ux, -yr/2, v.f])
            p2 = self.to_other_view(v, [ux,  yr/2, v.f])
            # horizontal line at y=uy
            p3 = self.to_other_view(v, [-xr/2, uy, v.f])
            p4 = self.to_other_view(v, [ xr/2, uy, v.f])
            if p1 and p2: self.ax.plot([p1[0],p2[0]], [p1[1],p2[1]], color='red', alpha=0.5)
            if p3 and p4: self.ax.plot([p3[0],p4[0]], [p3[1],p4[1]], color='red', alpha=0.5)

        # project each gaussian to g
        for g in gaussians:
            (mx, my), _, _, _, cov = g.proj(v.f, v.rot, v.eye)
            
            # get the mean of the gaussian in v's camera space
            g_cam = np.array([mx, my, v.f])

            # transform the mean of the gaussian to world space
            g_world = (v.rot.T @ g_cam) + v.eye

            cov3d = np.pad(cov, [(0,1), (0,1)], mode='constant', constant_values=0)

            # project g to v's projection plane and then get it back in 3d space
            projected_g = Gaussian(0,0,0,0,0,0, g_world, g.color)
            projected_g.cov = cov3d

            # now project that gaussian to this view
            self.draw_gaussian(projected_g)


        # draw v's center of projection
        veye_cam = self.rot @ (v.eye - self.eye)
        veyex, veyey = self.f * veye_cam[0 ] /veye_cam[2], self.f * veye_cam[1] / veye_cam[2]
        if veye_cam[2] > 0:
            self.ax.plot(veyex, veyey, 'r+', markersize=8)





plt.style.use('dark_background')

fig, axes = plt.subplots(
    1,2,
    figsize=(12,6)
)

# axes[0].set_box_aspect(1)      # square
# axes[1].set_box_aspect(0.5)    # width = 2 * height
plt.subplots_adjust(right=0.75)

# Parameters we want to play with
params = {
    'f fixed' : 2,
    'f dyna': 3,
    'pitch fixed': 22,
    "rx": 0,
    "ry": 0,
    "rz": 0,
    "sx": 1,
    "sy": 0.5,
    "sz": 2,
    'x': 0,
    'y': 0,
    'z': 10,
    'cx': 3.74,
    'cy': 1.8,
    'cz': -3.19,
    'pitch':9.4,
    'yaw': 19.3,
    'roll': 0,
    'xrange': 4,
    'yrange': 2
    
}


gaussians = [
    Gaussian(
        sx=1.637,
        sy=1.26,
        sz=1.774,
        rx=-15.6,
        ry=0,
        rz=25.6,
        mean=[10.9,18.5,38.3],
        color='blue'
    ),
    Gaussian(
        sx=2.014,
        sy=0.882,
        sz=2.048,
        rx=16.8,
        ry=-10.6,
        rz=-25.6,
        mean=[4.8, 9.4, 40.2],
        color='pink'
    ),
    Gaussian(
        sx=1,
        sy=0.5,
        sz=2,
        rx=0,
        ry=0,
        rz=0,
        mean=[0,0,30],
        color='orange'
    ),
    # Gaussian(
    #     sx=0.753,
    #     sy=0.306,
    #     sz=0.753,
    #     rx=35,
    #     ry=31.2,
    #     rz=-13.8,
    #     mean=[1.94,-14.03,13.44],
    #     color='orange'
    # ) 
    
]


def draw():
    cameras = [
        Camera(
            title='View From Origin Toward +Z',
            f=params['f fixed'],
            xrange=2,
            yrange=2, 
            pyr=[params['pitch fixed'],0,0],
            eye=[0,0,0],
            ax=axes[0]),
        Camera(
            title='Dynamic View',
            f=params['f dyna'], 
            xrange=params['xrange'],
            yrange=params['yrange'],
            pyr=[params['pitch'], params['yaw'], params['roll']],
            eye=[params['cx'],params['cy'],params['cz']], 
            ax=axes[1])
    ]
    
    # Static rectangle
    r = 2/3
    n = 3
    for i in range(1,n):
        p = (i/n)*2 - 1
        x1,y1 = [p,p], [-1,1]
        x2,y2 = [-1,1], [p,p]
        axes[0].plot(x1,y1,x2,y2, color='red', alpha=0.5)
    
    gaussian = Gaussian(
        sx=params["sx"],
        sy=params["sy"],
        sz=params["sz"],
        rx=params['rx'],
        ry=params['ry'],
        rz=params['rz'],
        mean = [params['x'],params['y'],params['z']],
        color='green'
    )
    for c1 in cameras:
        c1.draw_gaussian(gaussian)
        for g in gaussians:
            c1.draw_gaussian(g)
        for c2 in cameras:
            if c1 != c2:
                c1.draw_projplane(c2, gaussians + [gaussian])

    fig.canvas.draw_idle()

draw()



# Slider helper
def add_slider(name, y, low, high):
    ax = plt.axes([0.80, y, 0.15, 0.03])

    slider = Slider(
        ax,
        name,
        low,
        high,
        valinit=params[name]
    )

    def update(val):
        params[name] = val
        draw()

    slider.on_changed(update)
    return slider


sliders = [
    add_slider("f fixed",0.90,0.5,5),
    add_slider("f dyna",0.85,0.5,5),
    add_slider("pitch fixed",0.80,-60,60),
    add_slider("xrange",0.75,0.1,10),
    add_slider("yrange",0.70,0.1,10),
    
    add_slider("x",0.65,-10,100),
    add_slider("y",  0.60,-10,100),
    add_slider("z",  0.55,-10,100),
    add_slider("rx",0.50,-180,180),
    add_slider("ry",0.45,-180,180),
    add_slider("rz",  0.40,-180,180),
    add_slider("sx",   0.35,0.1,10),
    add_slider("sy",   0.30,0.1,10),
    add_slider("sz",   0.25,0.1,10),

    add_slider("cx",0.20,-40,40),
    add_slider("cy",  0.15,-40,40),
    add_slider("cz",  0.10,-40,40),
    add_slider("pitch",  0.05,-180,180),
    add_slider("yaw",  0.0,-180,180),
   # add_slider("roll",  -0.05,-180,180),  
]
plt.show()