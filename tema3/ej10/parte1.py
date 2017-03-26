import numpy as np
import matplotlib.pyplot as plt

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y
def add_radius(x, y, r ):
    ang, mod =  cart2pol(x, y)
    return pol2cart( ang, mod + r)

def adjustFigAspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)

def colision(r0, r1, p1x, p1y, p2x, p2y):
    d=np.linalg.norm([p1x-p2x , p1y-p2y] )
    a=(r0*r0 - r1*r1 + d*d)/(2*d)
    h = np.sqrt(r0*r0 - a*a)
    p3x = p1x + a*(p2x - p1x)/(d)
    p3y = p1y + a*(p2y - p1y)/d
    p4x = p3x - h*(p2y - p1y)/d
    p4y = p3y + h*(p2x - p1x)/d
    return p4x, p4y

def colisionM(r0, r1, p1x, p1y, p2x, p2y):
    d=np.linalg.norm([p1x-p2x , p1y-p2y] )
    a=(r0*r0 - r1*r1 + d*d)/(2*d)
    h = np.sqrt(r0*r0 - a*a)
    p3x = p1x + a*(p2x - p1x)/(d)
    p3y = p1y + a*(p2y - p1y)/d
    p4x = p3x + h*(p2y - p1y)/d
    p4y = p3y - h*(p2x - p1x)/d
    return p4x, p4y

def line(i):
        x = 1 + (1/i) * np.cos(np.arange(0 , 2*np.pi , 0.0001))
        y =  (1/(i))+(1/(i)) * np.sin(np.arange(0 , 2*np.pi , 0.0001))
        x_t , y_t = colision(1, 1/i, 0, 0, 1, 1/i)
        x_f = x[x < 1]
        y_f = y[x < 1]
        y_f = y_f[x_f > -1]
        x_f = x_f[x_f > -1]
        x_f = x_f[y_f < y_t ]
        y_f = y_f[y_f < y_t ]
        ax.plot(x_f, y_f , 'k', linewidth = 0.2)
        x_text , y_text = add_radius(x_t, y_t, 0.01)
        ax.text( x_text,
                y_text,
                str(i),
                verticalalignment='center',
                horizontalalignment='center',
                rotation=np.angle(x_t + y_t*1j, deg=True) - 90 ,
                fontsize=3)
        ##ax.plot(x_text, y_text, 'ko')

def line2(i):
    x = 1 + (1/(-1*i)) * np.cos(np.arange( -np.pi , np.pi,  0.0001))
    y =  (1/(i*-1))+(1/(i*-1)) * np.sin(np.arange(-np.pi , np.pi,  0.0001))
    x_t , y_t = colisionM(1, 1/i, 0, 0, 1, -1/i)
    x_f = x[x < 1]
    y_f = y[x < 1]
    y_f = y_f[x_f > -1]
    x_f = x_f[x_f > -1]
    x_f = x_f[y_f > y_t ]
    y_f = y_f[y_f > y_t ]
    x_text , y_text = add_radius(x_t, y_t, 0.02)
    ax.text( x_text,
            y_text,
            str(i),
            verticalalignment='center',
            horizontalalignment='center',
            rotation=np.angle(x_t + y_t*1j, deg=True) - 90 ,
            fontsize=3)

    #ax.plot(x_t, y_t, 'ko')



    ax.plot( x_f[20:] ,y_f[20:] , 'k', linewidth = 0.2)

def paint_line(i, ax):

        x = i/(1+i) + (1/(1+i)) * np.cos(np.arange(0 , 2*np.pi , 0.001))
        y = (1/(1+i)) * np.sin(np.arange(0 , 2*np.pi , 0.001))


        ax.plot(x, y, 'k', linewidth = 0.2)

        ax.text( 1-2*(1/(1+i)),
                0.02,
                str(i),
                verticalalignment='bottom',
                horizontalalignment='right',
                rotation=90,
                fontsize=3)
        line(i)
        line2(i)

def paint_text_degrees():
    positions = np.arange(0, np.pi*2, 2*np.pi / 36)
    for i, ang in enumerate(positions):
        x_t , y_t = pol2cart(ang, 1.04)
        ax.text( x_t,
                y_t,
                str(i*10),
                verticalalignment='center',
                horizontalalignment='center',
                rotation=np.angle(x_t + y_t*1j, deg=True) - 90 ,
                fontsize=3)

def paint_text_wavelength():
    positions = np.arange(np.pi, 3*np.pi, 2*np.pi / 50)
    for i, ang in enumerate(positions):
        x_t , y_t = pol2cart(ang, 1.06)
        ax.text( x_t,
                y_t,
                str(i/100),
                verticalalignment='center',
                horizontalalignment='center',
                rotation=np.angle(x_t + y_t*1j, deg=True) - 90 ,
                fontsize=3)

def imp2point(v1, v2):
    reax = v1/(1+v1)
    reay = 0
    rear = (1/(1+v1))
    imgx = 1
    imgy =  1/v2
    imgr = 1/v2
    return colisionM(rear, imgr, reax, reay, imgx, imgy)

def move_wl(x, y , wl):
    ax_ang, modulos = cart2pol(x, y)
    ax_ang += 4*np.pi*wl
    return pol2cart(ax_ang, modulos)

x_1= np.cos(np.arange(0 , 2*np.pi , 0.001))
y_1 = np.sin(np.arange(0, 2*np.pi, 0.001) )
fig = plt.figure()
adjustFigAspect(fig,aspect=1)
ax = fig.add_subplot(111)
ax.set_ylim(-1.01 , 1.01)
ax.set_xlim(-1.01, 1.01)
ax.axis('off')

ax.plot(x_1, y_1 , 'k', linewidth = 0.3)
#fig.axhline(y=0, xmin=-0.99, xmax=0.99, color='k', hold=None, linewidth = 0.5)
ax.plot([1, -1], [0, 0], 'k', linewidth = 0.3)
#ax.plot([0], [0], 'ko')
#black big lines
for i in np.arange(0.05, 0.2, 0.05):
    paint_line(i , ax)
for i in np.arange(0.2, 1, 0.1):
    paint_line(i , ax)
for i in np.arange(1, 2, 0.2):
    paint_line(i , ax)
for i in np.arange(2, 5, 1):
    paint_line(i , ax)
for i in np.array([5, 10, 20, 50]):
    paint_line(i , ax)
paint_text_degrees()
paint_text_wavelength()

p1 , p2 = imp2point(0.96, -1.62)
ax.plot(p1, p2, 'ko')
ax.plot([p1 ,0], [p2, 0], 'r')

start, modd= cart2pol(p1, p2)
p3, p4 =move_wl(p1, p2, -0.15)
ax.plot(p3, p4, 'ko')
end, modd= cart2pol(p3, p4)
data_x = modd*np.cos(np.arange(start , end , -0.0001))
data_y = modd*np.sin(np.arange(start , end , -0.0001))
ax.plot(data_x, data_y)

i = 0.23
x = i/(1+i) + (1/(1+i)) * np.cos(np.arange(0 , 2*np.pi , 0.001))
y = (1/(1+i)) * np.sin(np.arange(0 , 2*np.pi , 0.001))
ax.plot(x, y, 'r', linewidth = 0.5)
i = 0.17
x = 1 + (1/(-1*i)) * np.cos(np.arange( -np.pi , np.pi,  0.0001))
y =  (1/(i*-1))+(1/(i*-1)) * np.sin(np.arange(-np.pi , np.pi,  0.0001))
x_t , y_t = colisionM(1, 1/i, 0, 0, 1, -1/i)
x_f = x[x < 1]
y_f = y[x < 1]
y_f = y_f[x_f > -1]
x_f = x_f[x_f > -1]
x_f = x_f[y_f > y_t ]
y_f = y_f[y_f > y_t ]
ax.plot(x_f, y_f , 'r', linewidth = 0.5)



fig.savefig('images/out1.pdf')
