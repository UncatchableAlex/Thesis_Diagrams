import numpy as np
from matplotlib.patches import Ellipse, Rectangle, Polygon
from matplotlib.widgets import Slider
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from futhark_server import Server 

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

def camera_look_at_origin(cx, cy, cz):
    target = np.array([0,0,0]) # origin
    eye = np.array([cx,cy,cz])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    
    # pick an up vector that isn't parallel to forward
    up = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(forward, up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])  # fall back to Y-up at poles
    
    right = np.cross(forward, up);  right /= np.linalg.norm(right)
    up = np.cross(right, forward);   up /= np.linalg.norm(up)
    return np.array([
        [right[0], up[0], -forward[0], cx],
        [right[1], up[1], -forward[1], cy],
        [right[2], up[2], -forward[2], cz],
        [0,    0,    0,    1 ]
    ])
    
def camera_orbit(rho, rx_deg, ry_deg, rz_deg, gx, gy, gz):
    # start rho away along +Z, looking at origin, with Y "up"
    R = rot_mat(rx_deg, ry_deg, rz_deg) 

    g = np.array([gx, gy, gz])

    offset = R @ np.array([0, 0, rho])
    eye = g + offset
    
    # camera basis = world basis rotated by the same R
    right   = R @ np.array([1, 0, 0])
    up      = R @ np.array([0, 1, 0])
    forward = R @ np.array([0, 0, 1])   # points from origin toward eye

    return np.array([
        [right[0], up[0], forward[0], eye[0]],
        [right[1], up[1], forward[1], eye[1]],
        [right[2], up[2], forward[2], eye[2]],
        [0, 0, 0, 1]
    ])

# spherical coordinates to cartesian
def spherical2cartesian(rho, theta_deg, phi_deg):
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    return (
            rho*np.cos(theta),              # x
            rho*np.sin(theta)*np.cos(phi),  # y
            rho*np.sin(theta)*np.sin(phi)   # z
            )              


def cov3d(sx, sy, sz, rx, ry, rz):
    R = rot_mat(rx,ry,rz)
    S = np.array([
        [sx,0,0],
        [0,sy,0],
        [0,0,sz]
    ])
    return (R@S)@((R@S).T)

def camera(rx, ry, rz, mx, my, mz):
    camera = rot_mat(rx,ry,rz).T
    camera = np.concatenate((camera, [[mx],[my],[mz]]),axis=1)
    return np.concatenate((camera, [[0,0,0,1]]), axis=0)

# Claude-generated function
def project_point(p_world, cam_mat, f, W):
    """Project a 3D world point to pixel coordinates."""
    world_to_cam = np.linalg.inv(cam_mat)
    p = world_to_cam @ np.array([*p_world, 1.0])
    depth = p[2]
    # if depth <= 0:
    #     return None  # behind camera
    px = (f * p[0] / depth) * W/2 + W/2
    py = -(f * p[1] / depth) * W/2 + W/2 
    return px, py

# Claude-generated function
def draw_axes_overlay(ax, cam_mat, f, W, origin=(0,0,0), length=1):
    o = project_point(origin, cam_mat, f, W)
    if o is None:
        return
    for direction, color, label in [
        ([1,0,0], '#ff4444', 'X'),
        ([0,1,0], '#44ff44', 'Y'),
        ([0,0,1], '#4488ff', 'Z'),
    ]:
        tip = [origin[i] + length * direction[i] for i in range(3)]
        t = project_point(tip, cam_mat, f, W)
        if t is None:
            continue
        ax.annotate('', xy=t, xytext=o,
            arrowprops=dict(arrowstyle='->', color=color, lw=2, mutation_scale=15))
        ax.text(t[0], t[1], label, color=color, fontsize=11, fontweight='bold',
                ha='center', va='center')

plt.style.use('dark_background')

fig, axes = plt.subplots(
    1,1,
    figsize=(12,12)
)
plt.subplots_adjust(right=0.75)

# Parameters we want to play with
params = {
    'f' : 2,
    'rho': 5,
    "rx": 0,
    "ry": 0,
    "rz": 0,
    "sx": 0.25,
    "sy": 0.12,
    "sz": 0.5,
    'x': 0.5,
    'y': 0.5,
    'z': 0.5,
    'pitch':0,
    'yaw': 0,
    'roll': 0,
    'xrange': 4,
    'yrange': 2,
    'delta': 0.1,
    'W': 50,
    'max_depth': 10
}




server = Server('./volumetric')

def draw():
    cov = cov3d(params['sx'], params['sy'], params['sz'], 0.5,0.5,0.5)
    conic = np.linalg.inv(cov)
    coef = 1 / ((2*np.pi)**1.5 * np.linalg.det(cov)**0.5)

    server.put_value('coef', np.float32(coef))
    server.put_value('conic', conic.astype(np.float32))

    #cx, cy, cz = spherical2cartesian(params['rho'], params['theta'], params['phi'])

    cam = camera_orbit(params['rho'], params['rx'], params['ry'], params['rz'], params['x'], params['y'], params['z']) #cam = camera_look_at_origin(cx,cy,cz) #camera(params['pitch'], params['yaw'], params['roll'], cx, cy, cz)
    server.put_value('W', np.int64(params['W']))
    server.put_value('camera', cam.astype(np.float32))
    server.put_value('focal_length', np.float32(params['f']))
    server.put_value('delta', np.float32(params['delta']))
    server.put_value('max_depth', np.float32(params['max_depth']))
    server.put_value('mean', np.array([params['x'], params['y'], params['z']], dtype=np.float32))
    
    server.cmd_call('rasterize', 'res', 'W', 'focal_length', 'camera', 'coef', 'conic', 'mean', 'delta', 'max_depth')
    res = server.get_value('res')

    server.cmd_free('res')
    server.cmd_free('camera')
    server.cmd_free('focal_length')
    server.cmd_free('delta')
    server.cmd_free('W')
    server.cmd_free('max_depth')
    server.cmd_free('coef')
    server.cmd_free('conic')
    server.cmd_free('mean')
    axes.cla()
    axes.imshow(1 - res, cmap='gray', vmin=0, vmax=1, origin='upper')
    draw_axes_overlay(axes, cam, params['f'], params['W'])
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
    add_slider("f",0.90,0.5,5),
    add_slider("xrange",0.75,0.1,10),
    add_slider("yrange",0.70,0.1,10),
    add_slider("delta",0.65,0.01,1),
    add_slider("max_depth", 0.60, 1,20),
    add_slider("W", 0.55, 50,500),
    add_slider("sx",0.50,0.1,1),
    add_slider("sy", 0.45, 0.1,1),
    add_slider("sz", 0.40, 0.1,1),
    # add_slider("cx",0.35,-15,15),
    # add_slider("cy",  0.30,-15,15),
    # add_slider("cz",  0.25,-15,15),
    add_slider("rho",0.35,0.1,15),
    add_slider("rx",  0.30,-180,180),
    add_slider("ry",  0.25,-180,180),
    add_slider("rz",  0.20,-180,180),
    
    # add_slider("pitch",  0.20,-180,180),
    # add_slider("yaw",  0.15,-180,180),
]
plt.show()