import numpy as np
import re
import random

import matplotlib.pyplot as plt
from matplotlib import cm

# cleans str and returns numpy array
def clean_k_arc(k_str):
    try:
        matches = re.compile(r'\d+').findall(k_str)
        nums = np.array([int(match) for match in matches])
        arc = nums.reshape(int(nums.shape[0]/2),2)
        arc_xoco(xo_arc(arc))
        0/arc.shape[0]
        return arc
    except:
        return None
    
# turns an arc array into a xo array
def xo_arc(arc):
    s = np.shape(arc)[0]
    xo = np.zeros([s,s], dtype=int)
    for i in range(s):
        for j in range(2):
            ik = i+1
            ij = arc[i,j]-1
            xo[-ik, ij] = j+1
    return xo

# turns a xo array into a xco array
def xco_xo(xo):
    xco = xo.copy()
    s = np.shape(xo)[0]
    r = range(1,s-1)
    for i in r:
        for j in r:
            if xo[i,j] == 0:
                hlz, hrz = np.count_nonzero(xo[i,:j]), np.count_nonzero(xo[i,j+1:])
                vuz, vdz = np.count_nonzero(xo[:i,j]), np.count_nonzero(xo[i+1:,j])
                if hlz==1 and hrz==1 and vuz==1 and vdz==1:
                    xco[i,j] = 3
    return xco

# turns an arc array into a xco array
def xco_arc(arc):
    return(xco_xo(xo_arc(arc)))

# turns a xo or xco array into an arc array
def arc_xoco(xoco):
    s = np.shape(xoco)[0]
    arc = np.zeros([s,2], dtype=int)
    for i in range(s):
        ik = i+1
        j1 = int(np.where(xoco[-ik]==1)[0])+1
        j2 = int(np.where(xoco[-ik]==2)[0])+1
        arc[i,0], arc[i,1] = j1, j2
    return arc

# turns a xco array into a a xabo array
def xabo_xco(xco, crossings):
    xabo = xco.copy()
    cruces = np.array(np.where(xabo==3))
    xabo[cruces[0],cruces[1]] = crossings
    return xabo


# get circle coords and circle codes
def circles_crd_cds(xabo):
    size = np.shape(xabo)[0]
    crossings = np.array(np.where(xabo>3))

    n_crossings = np.shape(crossings)[1]
    cr_check = np.zeros((n_crossings*2), dtype=int) #cross_check

    # 4 = A -> (up-right)(down-left)
    # 5 = B -> (up-left)(down-right)
    # cr_check = [c1u, c1d, c2u, c2d, ..., cnu, cnd]

    # CODES False=vertex ; True=crossing
    # p = position = [[coords in matrix], is_horizontal_bool, is_increasing_bool]

    circle_coords, circle_codes = [], []
    n_circle = 0

    while np.any(cr_check==0):
        circle_coords.append([])
        circle_codes.append([])
        fz= np.where(cr_check==0)[0][0] #first_zero
        p = [[crossings[:,int(fz/2)][0], crossings[:,int(fz/2)][1]], False, fz%2!=0]
        p0 = p[:]
        while True:
            circle_coords[n_circle].append(p[0])
            for i in range(1, size):
                r, c = 0, 0
                if not(p[2]): i=-i

                if p[1]: c = i
                else: r = i

                px = xabo[p[0][0]+r, p[0][1]+c]
                if px != 0:
                    # update p0 & p1
                    p[0], p[1] = [p[0][0]+r, p[0][1]+c], not(p[1])
                    break 
            
            # update p2 & codes
            if px==1 or px==2:
                line = xabo[p[0][0],:] if p[1] else xabo[:,p[0][1]]
                coord = p[0][1] if p[1] else p[0][0]
                p[2] = False if any(line[:coord]!=0) else True
                circle_codes[n_circle].append(False)
            elif px==5: p[2] = not(p[2])
            
            # update cr_check & codes
            if px==4 or px==5:
                circle_codes[n_circle].append(True)
                for i in range(n_crossings):
                    if all(p[0]==crossings[:,i]):
                        if not(p[1]) or px==5:
                            if p[2]: cr_check[2*i+1]=1
                            else: cr_check[2*i]=1
                        else:
                            if p[2]: cr_check[2*i]=1
                            else: cr_check[2*i+1]=1
                        break

            if np.all(p==p0): break

        n_circle += 1

    for i in range(n_circle):
        circle_codes[i] = [circle_codes[i][-1]] + circle_codes[i][:-1]
        
    return circle_coords, circle_codes


# draw circles
from matplotlib.path import Path
from matplotlib.patches import PathPatch
mpl_colors = ['blue', 'green', 'red', 'orange', 'purple',
              'cyan', 'olive', 'brown', 'pink', 'gray']

def draw_circles(fig, xabo):
    radius = .5
    size = np.shape(xabo)[0]
    circle_coords, circle_codes = circles_crd_cds(xabo)
    
    crcl_crd_path = circle_coords.copy()
    crcl_cds_path = circle_codes.copy()

    n_circles = len(circle_codes)
    for i in range(n_circles):
        n_inserts = 0 
        for j in range(len(circle_codes[i])): 
            if circle_codes[i][j]!=0:
                i_p = j + n_inserts
                ins1, ins2 = [0,0], [0,0]
                len_lcpath = len(crcl_crd_path[i])
                for k in range(2):
                    c0 = crcl_crd_path[i][i_p-1][k]
                    c1 = crcl_crd_path[i][i_p][k]
                    c2 = crcl_crd_path[i][(i_p+1)%len_lcpath][k]
                    if c0==c1: ins1[k]=c1
                    elif c0<c1: ins1[k]=c1-radius
                    elif c0>c1: ins1[k]=c1+radius
                    if c2==c1: ins2[k]=c1
                    elif c2<c1: ins2[k]=c1-radius
                    elif c2>c1: ins2[k]=c1+radius

                crcl_crd_path[i].insert(i_p, ins1)
                crcl_crd_path[i].insert(i_p+2, ins2)
                n_inserts+=2

    for i in range(n_circles):
        crcl_cds_path[i] = [Path.MOVETO]
        for code in circle_codes[i]:
            if code==0: crcl_cds_path[i]+= [Path.LINETO]
            else: crcl_cds_path[i]+= [Path.CURVE3]*2 + [Path.LINETO]
        crcl_cds_path[i].pop(-1)
        crcl_cds_path[i] += [Path.CLOSEPOLY]

    for circle in crcl_crd_path:
        for coord in circle:
            coord[0], coord[1] = coord[1], size-(coord[0]+1)
        circle += [[0,0]]


    ax = fig.add_subplot()

    for i in range(n_circles):
        path = Path(crcl_crd_path[i], crcl_cds_path[i])
        pathpatch = PathPatch(path, facecolor='none', edgecolor=mpl_colors[i%len(mpl_colors)], lw=3)
        ax.add_patch(pathpatch)

    ax.axis(False)
    ax.autoscale_view(True)
    ax.set_aspect('equal', 'box')
    
    return fig



#######################
# PLOT TURAEV SURFACE #
#######################

def set_3daxes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def crossing_surfpoints(base_size=20, height=10, curve=5, res_c=4, res_z=6):
    b, h, c = base_size, height, curve
    nc, nz = res_c, round(res_z*.5)+1
    ta, curve = np.linspace(0,1,nc), np.empty([2, nc])
    xtot, ytot = np.empty([2*nz,nc]), np.empty([2*nz,nc])
    zh, ztot = np.linspace(h/2, 0, nz), np.empty([2*nz,nc])

    P0 = np.array([0, b/2])
    for j, z in enumerate(zh):
        c1 = (z*c)**.5
        c2 = c1*.5
        P1, P2 = np.array([0, c1]), np.array([c2, c2])
        for i in  range(2):
            curve[i] = (1-ta)**2*P0[i] + 2*(1-ta)*ta*P1[i] + ta**2*P2[i]

        k = -(j+1)
        xtot[j], ytot[j], ztot[j] = curve[0], curve[1], np.linspace(z,z,nc)
        xtot[k], ytot[k], ztot[k] = -curve[0], curve[1], -np.linspace(z,z,nc)
    
    return xtot, ytot, ztot


def draw_crossing(ax, smoothing=4, coords=(10, 10),
                  colormap=cm.viridis_r, surfpoints=crossing_surfpoints()):
    
    xq, yq, zq = surfpoints
    if smoothing == 5:
        xq = -xq
    x_c, y_c = coords[0], coords[1]
    ax.plot_surface(xq + x_c, yq + y_c, zq, cmap=colormap)
    ax.plot_surface(yq + x_c, xq + y_c, zq, cmap=colormap)
    ax.plot_surface(-yq + x_c, -xq + y_c, zq, cmap=colormap)
    ax.plot_surface(-xq + x_c, -yq + y_c, zq, cmap=colormap)


def draw_line(ax, coords0, coords1, height=10, res_z=6, colormap=cm.viridis_r):
    h, nz = height*.5, round(res_z*.5)*2+1

    x_coords = np.array([coords0[0], coords1[0]])
    y_coords = np.array([coords0[1], coords1[1]])
    z_coords = np.zeros([nz, 2])
    z_coords[:,0], z_coords[:,1] = np.linspace(h,-h,nz), np.linspace(h,-h,nz)
    
    ax.plot_surface(x_coords, y_coords, z_coords, cmap=colormap)

#draw surf diagram 
def turaev_surf(fig, xabo, view=(-75, 30), size=20,
                height=10, res_z=6, cmap=cm.viridis_r):
    ax = fig.add_subplot(projection='3d')
    ax._axis3don = False
    
    s = np.shape(xabo)[0]
    
    ## VERTICAL AND HORIZONTAL LINES
    for i in range(s):
        row, col = xabo[i,:], xabo[:,i]
        
        hverts, n_hlines = np.where(row>0)[0], np.count_nonzero(row>0)-1
        for j in range(n_hlines):
            c0, c1 = hverts[j:j+2]
            if row[c0]>2: c0 += .5
            if row[c1]>2: c1 -= .5
            p0, p1 = np.array([c0, s-(i+1)])*size, np.array([c1, s-(i+1)])*size
            draw_line(ax, p0, p1,
                      height=height,
                      res_z=res_z,
                      colormap = cmap)
        
        vverts, n_vlines = np.where(col>0)[0], np.count_nonzero(col>0)-1
        for j in range(n_vlines):
            r0, r1 = vverts[j:j+2]
            if j!=0: r0 += .5
            if j!=n_vlines-1: r1 -= .5
            p0, p1 = np.array([i, s-(r0+1)])*size, np.array([i, s-(r1+1)])*size
            draw_line(ax, p0, p1,
                      height=height,
                      res_z=res_z,
                      colormap = cmap)
    
    ## CROSSINGS
    cr_surfpoints = crossing_surfpoints(base_size=size,
                                        height=height,
                                        res_z=res_z)
    c_pos, n_c = np.where(xabo>2), np.count_nonzero(xabo>2)
    for i in range(n_c):
        row, col = c_pos[0][i], c_pos[1][i]
        ab = xabo[row, col]
        cr_coords = np.array([col, s-(row+1)])*size
        draw_crossing(ax, ab, cr_coords,
                      surfpoints=cr_surfpoints,
                      colormap = cmap)

    
    ax.view_init(azim=view[0], elev=view[1])
    set_3daxes_equal(ax)

    return fig

######################
# GENERATE NEW KNOTS #
######################
def state_is_valid(state, n, pos):
    if np.count_nonzero(state>0)==2*n-1:
        sol = state.copy()
        sol[0, pos[1]] = 2
        xco_sol = xco_xo(sol)
        if np.count_nonzero(xco_sol==3)>n-1:
            return True

    return False

def get_candidates(state, n, pos):
    p = state[pos]
    cands = []
    if p == 1:
        for i in range(1, n):
            if np.count_nonzero(state[i,:]==2)==0:
                cands.append([(i, pos[1]), 2])
    elif p == 2:
        for j in range(n):
            if np.count_nonzero(state[:,j]==1)==0:
                cands.append([(pos[0], j), 1])
    return cands

def search(state, n, pos):
    if state_is_valid(state, n, pos):
        state[0, pos[1]] = 2
        return xco_xo(state)
    else:
        candidates = get_candidates(state, n, pos)
        random.shuffle(candidates)
        for candidate in candidates:
            state[candidate[0]] = candidate[1]
            if type(search(state, n, candidate[0]))!=type(None):
                return xco_xo(state)
            else:
                state[candidate[0]] = 0

def random_knot(n):
    solutions = []
    state = np.zeros((n,n), dtype=np.int8)
    pos = (0, random.randint(0,n-1))
    state[pos]=1
    return search(state, n, pos)
