##resin--automatic resonance plotting and visualization

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import resonances
import re
import pytz

from skyfield.positionlib import build_position, Barycentric
from skyfield.api import load
from skyfield.functions import mxv, rot_z, mxm
from skyfield.framelib import ecliptic_frame, ecliptic_J2000_frame
from skyfield.api import PlanetaryConstants
from skyfield.planetarylib import Frame
from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2
from skyfield.timelib import Timescale


def get_orbital_elements_from_state(position):
    mu=GM_SUN_Pitjeva_2005_km3_s2
    pos,vel=position.frame_xyz_and_velocity(ecliptic_J2000_frame)
    pos=pos.km.T
    vel=vel.km_per_s.T
    radius=np.linalg.norm(pos, axis=1)[:,np.newaxis]
    velocity_radial=np.sum(pos/radius*vel, axis=1)
    #print(velocity_radial)
    h_vec=np.cross(pos,vel,axis=1)
    h=np.linalg.norm(h_vec,axis=1)
    i=np.arccos(h_vec[:,2]/h)
    K=np.column_stack((np.zeros_like(radius),np.zeros_like(radius),np.ones_like(radius)))
    N_vec=np.cross(K, h_vec, axis=1)
    N = np.linalg.norm(N_vec, axis=1)
    Omega = np.where(N_vec[:,1]>=0, np.arccos(N_vec[:,0]/N), 2 * np.pi-np.arccos(N_vec[:,0]/N))
    e_vec=np.cross(vel, h_vec, axis=1)/mu - pos/radius
    e=np.linalg.norm(e_vec, axis=1)
    w= np.where(e_vec[:,2]>=0, np.arccos(np.sum(N_vec*e_vec,axis=1)/(N*e)), 2 * np.pi -np.arccos(np.sum(N_vec*e_vec,axis=1)/(N*e)))
    nu=np.where(velocity_radial>=0, np.arccos(np.sum(pos/radius*e_vec/e[:,np.newaxis], axis=1)),2*np.pi-np.arccos(np.sum(pos/radius*e_vec/e[:,np.newaxis], axis=1)))
    a=h**2/(1-e**2)/mu
    M=np.atan2(np.sqrt(1-e**2)*np.sin(nu),e+np.cos(nu))-e*(np.sqrt(1-e**2)*np.sin((nu)))/(1+e*np.cos(nu))
    orbital_elements=pd.DataFrame({'a':a/149597870.7,'e':e,'i':i,'Omega':Omega,'w':w,'nu':nu, 'M':M, 'lon':np.unwrap(Omega+w+M),'period':(2*np.pi*np.sqrt(a**3/(mu)))/86400})
    return orbital_elements

def get_fake_elements_from_state(position):
    pos=position.xyz
    vel=position.velocity
    pos=pos.km.to_numpy()
    vel=vel.km_per_s.to_numpy()
    radius=np.linalg.norm(pos, axis=1)[:,np.newaxis]
    velocity_radial=np.sum(pos/radius*vel, axis=1)
    #print(velocity_radial)
    h_vec=np.cross(pos,vel,axis=1)
    h=np.linalg.norm(h_vec,axis=1)
    i=np.arccos(h_vec[:,2]/h)
    K=np.column_stack((np.zeros_like(radius),np.zeros_like(radius),np.ones_like(radius)))
    N_vec=np.cross(K, h_vec, axis=1)
    N = np.linalg.norm(N_vec, axis=1)
    Omega = np.where(N_vec[:,1]>=0, np.arccos(N_vec[:,0]/N), 2 * np.pi-np.arccos(N_vec[:,0]/N))
    e_vec=np.cross(vel, h_vec, axis=1)/GM_SUN_Pitjeva_2005_km3_s2 - pos/radius
    e=np.linalg.norm(e_vec, axis=1)
    w= np.where(e_vec[:,2]>=0, np.arccos(np.sum(N_vec*e_vec,axis=1)/(N*e)), 2 * np.pi -np.arccos(np.sum(N_vec*e_vec,axis=1)/(N*e)))
    nu=np.where(velocity_radial>=0, np.arccos(np.sum(pos/radius*e_vec/e[:,np.newaxis], axis=1)),2*np.pi-np.arccos(np.sum(pos/radius*e_vec/e[:,np.newaxis], axis=1)))
    a=h**2/(1-e**2)/GM_SUN_Pitjeva_2005_km3_s2
    M=np.atan2(np.sqrt(1-e**2)*np.sin(nu),e+np.cos(nu))-e*(np.sqrt(1-e**2)*np.sin((nu)))/(1+e*np.cos(nu))
    orbital_elements=pd.DataFrame({'a':a/149597870.7,'e':e,'i':i,'Omega':Omega,'w':w,'nu':nu, 'M':M,'lon':np.unwrap(Omega+w+M)})
    return orbital_elements

def res_string_to_chunks(resstring):
    #process resonance string into some names
    chunks=re.findall("[\\d+-]+",resstring)
    if len(chunks)==2:
        planet1order=int(chunks[0])
        smallorder=-1*int(chunks[-1])
        planet2order=0
    elif len(chunks)==3:
        planet1order=int(chunks[0])
        planet2order=int(chunks[1])
        smallorder=-1*int(chunks[-1])
    return smallorder, planet1order, planet2order

#bump resonance order limit because high-order res are common for TNOs
resonances.config.set('MATRIX_2BODY_PRIMARY_MAX',25)
resonances.config.set('MATRIX_2BODY_COEF_MAX',25)
resonances.config.set('MATRIX_2BODY_ORDER_MAX',25)

#general parameters
asteroid='Haumea (system barycenter)'
largebody='Neptune'
resstring='7N-12'
outputanim='cache/resin_animation_test.mp4'
outputresangle='cache/resangle.png'
frameinterval=50 #interval between animation frames in ms

#parameters for solex processing    
solex_files=False
smallfilename='../HAUMEA.OUT'
largefilename='../NEPTUNE.OUT'
precessionrate=(2*np.pi)/(365.25*25771.57534)

#parameters for rebound processing
rebound=True
autodetect=True
years_to_integrate=200000
start_year=-100000 #in years before config.date, accessible in configuration settings.
nout=100000 #number of plotting timesteps

if rebound==True:
    solex_files=False #both at once makes a mess
    if autodetect==True:
        sim = resonances.find(asteroid,largebody,sigma2=0.5)
    else:
        sim = resonances.check(asteroid, resonance=resstring, sigma2=0.5)
    sim.config.tmax=(years_to_integrate+min(start_year,0))*2*np.pi
    sim.config.dt = 1
    sim.config.Nout = int(nout*(1+start_year/years_to_integrate))
    print('Integrating forward...')
    sim.run(progress=True)
    
    #we want to autodetect resonances, or pass them as an argument
    resdata=sim.data_manager.get_simulation_summary(sim.bodies)
    resdata=resdata.iloc[resdata.status.abs().argsort()].iloc[::-1]
    resdata_first=resdata.groupby('name').first().reset_index()
    reslabel=resdata_first['resonance'][0]
    for index, row in resdata_first.iterrows():
        if row['status']==2:
            print('Minor planet '+str(row['name'])+' is in a confirmed '+row['res_short']+' resonance.')
        elif row['status']==-2:
            print('Minor planet '+str(row['name'])+' may be in a '+row['res_short']+' resonance, but this requires investigation.')
        elif row['status']==1:
            print('Minor planet '+str(row['name'])+' is in an intermittent '+row['res_short']+' resonance.')
        elif row['status']==-1:
            print('Minor planet '+str(row['name'])+' may be in an intermittent '+row['res_short']+' resonance.')
        else:
            print('Minor planet '+str(row['name'])+' is not resonant.')
    if autodetect==True:
        resstring=resdata_first['res_short'][0]

    if start_year < 0: #integrate backwards after doing so forwards
        sim2 = resonances.check(asteroid, resonance=resstring, plot_type='none', sigma2=0.5, show_all=True)
        sim2.config.tmax=start_year*2*np.pi
        sim2.config.dt = sim.config.dt
        sim2.config.Nout = int(nout*-start_year/years_to_integrate)
        print('Integrating backward...')
        sim2.run(progress=True)

#general position and time helpers
ts = load.timescale()
small_order, planet1order, planet2order = res_string_to_chunks(resstring)

#define output params from sim

if rebound==True:
    body=sim.bodies[0]
    if start_year < 0: #we want both forward and backward integrations in 1 object
        body2=sim2.bodies[0]
        #flip and stack all the variables we need
        bodyX=np.hstack([body2.X[::-1],body.X[1:]])
        bodyY=np.hstack([body2.Y[::-1],body.Y[1:]])
        bodyZ=np.hstack([body2.Z[::-1],body.Z[1:]])
        bodyVx=np.hstack([body2.Vx[::-1],body.Vx[1:]])
        bodyVy=np.hstack([body2.Vy[::-1],body.Vy[1:]])
        bodyVz=np.hstack([body2.Vz[::-1],body.Vz[1:]])
        bodylon=np.hstack([body2.longitude[::-1],body.longitude[1:]])
        bodyvarpi=np.hstack([body2.varpi[::-1],body.varpi[1:]])
        bodyangles=np.hstack([body2.angles[reslabel][::-1],body.angles[reslabel][1:]])
        planetpos1=np.vstack([body2.planetpos1[reslabel][::-1, :],body.planetpos1[reslabel][1:,:]])
        timeobject_s=ts.from_datetime(sim.config.date.astimezone(pytz.UTC))+365.25/(np.pi*2)*np.hstack([sim2.times[::-1],sim.times[1:]])
    else: #just forward
        bodyX,bodyY,bodyZ,bodyVx,bodyVy,bodyVz,bodylon,bodyvarpi, bodyangles=body.X, body.Y, body.Z, body.Vx, body.Vy, body.Vz, body.longitude, body.varpi, body.angles[reslabel]
        planetpos1=body.planetpos1[reslabel]
        timeobject_s=ts.from_datetime(sim.config.date.astimezone(pytz.UTC))+365.25/(np.pi*2)*sim.times


if solex_files==True: #read the Solex data under the assumption it's heliocentric ecliptic-of-date frame. If it's different we'll need to do something else.
    smallbody=pd.read_fwf(smallfilename,header=3,widths=[15,21,18,18,18,18,18,18,18,18]).rename(columns=lambda x: x.strip())
    largebody=pd.read_fwf(largefilename,header=3,widths=[15,21,18,18,18,18,18,18,18,18]).rename(columns=lambda x: x.strip())
    timeobject_s=ts.tdb_jd(2451545.0+365.25*smallbody['T##'].to_numpy())
    timeobject_l=ts.tdb_jd(2451545.0+365.25*largebody['T##'].to_numpy())
    smallbody_position= build_position(smallbody[['X','Y','Z']].to_numpy().T,smallbody[['Vx','Vy','Vz']].to_numpy().T/149.5978707, t=timeobject_s)
    largebody_position= build_position(largebody[['X','Y','Z']].to_numpy().T,largebody[['Vx','Vy','Vz']].to_numpy().T/149.5978707, t=timeobject_l)

elif rebound==True: #rebound+resonances provide positions in barycentric ecliptic frame, which is much easier to work with.
    timeobject_l=timeobject_s
    smallbody_position= Barycentric(np.stack([bodyX,bodyY,bodyZ]),np.stack([bodyVx,bodyVy,bodyVz]),timeobject_s,0)
    largebody_position= Barycentric(planetpos1[:,0:3].T,planetpos1[:,3:6].T,timeobject_l,0)

else: #uhhh...
    raise NotImplementedError

n_timesteps=np.shape(largebody_position.t)[0] #number of timesteps can be slightly different than intended because of integrator weirdness

#Rotating our arbitrary starting frame into planet-Sun orbit frame (planet is on the x axis and z velocity is 0.)
#rotation matrices will be numerically unstable for small z. Hence we reflect twice to get the vector in the right place.
#Our rotation matrix will be the product of two reflection matrices, that swap f and the z axis, and then the z and x axis. 
#We expect most bodies to have a reasonably small inclination (looking at you, Taowu).
#v=[1,0,-1]

largebody_u=(largebody_position.xyz.au-np.vstack((np.zeros(n_timesteps),np.zeros(n_timesteps),np.linalg.norm(largebody_position.xyz.au, axis=0)))).T
norm_u=np.square(largebody_u).sum(axis=1)
lu_values=largebody_u
eye=np.tile(np.identity(3),[n_timesteps,1,1])
entry_00=-2/norm_u*lu_values[:,0]*lu_values[:,0]-1+2*(lu_values[:,0]-lu_values[:,2])/norm_u*lu_values[:,0]
entry_01=-2/norm_u*lu_values[:,0]*lu_values[:,1]+2*(lu_values[:,0]-lu_values[:,2])/norm_u*lu_values[:,1]
entry_02=-2/norm_u*lu_values[:,0]*lu_values[:,2]+1+2*(lu_values[:,0]-lu_values[:,2])/norm_u*lu_values[:,2]
entry_10=-2/norm_u*lu_values[:,1]*lu_values[:,0]
entry_11=-2/norm_u*lu_values[:,1]*lu_values[:,1]
entry_12=-2/norm_u*lu_values[:,1]*lu_values[:,2]
entry_20=-2/norm_u*lu_values[:,2]*lu_values[:,0]+1-2*(lu_values[:,0]-lu_values[:,2])/norm_u*lu_values[:,0]
entry_21=-2/norm_u*lu_values[:,2]*lu_values[:,1]-2*(lu_values[:,0]-lu_values[:,2])/norm_u*lu_values[:,1]
entry_22=-2/norm_u*lu_values[:,2]*lu_values[:,2]-1-2*(lu_values[:,0]-lu_values[:,2])/norm_u*lu_values[:,2]
rotationmatrix1=np.block([[[1+entry_00],[entry_01],[entry_02]],[[entry_10],[1+entry_11],[entry_12]],[[entry_20],[entry_21],[1+entry_22]]]) #rotation 1 into plane

largebody_rotated=mxv(rotationmatrix1, largebody_position.xyz.au)
largebody_velocity=mxv(rotationmatrix1, largebody_position.velocity.m_per_s)
gamma=np.arctan2(largebody_velocity[2,:],largebody_velocity[1,:])#roll angle to get velocity vector onto planet plane
rotationmatrix2=np.block([[[np.ones(n_timesteps)],[np.zeros(n_timesteps)],[np.zeros(n_timesteps)]],[[np.zeros(n_timesteps)],[np.cos(gamma)],[np.sin(gamma)]],[[np.zeros(n_timesteps)],[-np.sin(gamma)],[np.cos(gamma)]]])
largebody_rotated=mxv(rotationmatrix2, largebody_rotated)
largebody_velocity=mxv(rotationmatrix2, largebody_velocity)

#now we just need to rotate our small body
smallbody_rotated=mxv(mxm(rotationmatrix2,rotationmatrix1), smallbody_position.xyz.au)
smallbody_velocity=mxv(mxm(rotationmatrix2,rotationmatrix1), smallbody_position.velocity.m_per_s)

smallbody_rotated_position= build_position(smallbody_rotated,smallbody_velocity*86400/149597870700, t=timeobject_s)
largebody_rotated_position= build_position(largebody_rotated,largebody_velocity*86400/149597870700, t=timeobject_l)

#orbital elements--for solex we rotate into ecliptic frame.
if solex_files==True:
    large_orbels=get_orbital_elements_from_state(largebody_position)
    small_orbels=get_orbital_elements_from_state(smallbody_position)
    #for a 2 body resonance, res angle is (minus) p*lam1-q*lam2-(p-q)*varpi_1
    resangle=-1*(small_order*(small_orbels['lon'])-planet1order*(large_orbels['lon'])+(-small_order+planet1order)*(np.unwrap(small_orbels['w'])+np.unwrap(small_orbels['Omega'])))%(2*np.pi)
elif rebound==True:
    resangle=bodyangles #much simpler because we can just get it from resonances.

#resangle=(-smallorder*np.unwrap(large_orbels['nu']+large_orbels['Omega']+large_orbels['w'])+largeorder*np.unwrap(small_orbels['nu']+small_orbels['Omega']+small_orbels['w'])-6*np.unwrap(small_orbels['Omega'])-5*np.unwrap(small_orbels['w']))%(2*np.pi)

#we want to plot one batch at a time--so we cut for every n orbital periods of the small body.
if solex_files==True:
    restime=2*np.lcm(small_order, planet1order)*np.pi/(planet1order*(np.average(np.diff(large_orbels['lon'])/np.diff(largebody_position.t))-precessionrate))
    #the last term is for deprecessing the slowly rotating ecliptic.of_date frame
    resperiod=restime/np.average(np.diff(largebody_position.t))
elif rebound==True: #we only have els for the small body provided by resonances, so let's use those (e.g. p*lam1-(p-q)*varpi_1)
    reslon=small_order*np.unwrap(bodylon)-(small_order+planet1order+planet2order)*np.unwrap(bodyvarpi)
    restime=2*np.lcm(small_order, planet1order)*np.pi/np.average(np.diff(reslon)/np.diff(largebody_position.t))
    resperiod=restime/np.average(np.diff(largebody_position.t))

print('resonance period: '+str(resperiod)+' timesteps, or '+str(restime)+' days.')

#output plot: resonance angle
fig2, ax2=plt.subplots(figsize=(12,6))
ax2.set(xlabel='Time (Julian year)',ylabel='Resonance angle (radians)',title='Resonance angle of '+asteroid)
ax2.plot(timeobject_s.J, resangle,'o',ms=0.2, color='b')
fig2.savefig(outputresangle)

#output animation: spirograph 
plt.style.use('dark_background')
fig,ax=plt.subplots(figsize=(8,8))
maxradius=np.max(np.linalg.norm(smallbody_position.xyz.au, axis=0))
ax.set(xlim=[-1.2*maxradius, 1.2*maxradius], ylim=[-1.2*maxradius, 1.2*maxradius], xlabel='X position (AU)', ylabel='Y position (AU)', title='Resonance of '+asteroid)
smallxpos=smallbody_rotated_position.position.au[0,:]
smallypos=smallbody_rotated_position.position.au[1,:]
largexpos=largebody_rotated_position.position.au[0,:]
largeypos=largebody_rotated_position.position.au[1,:]

#plot the first frame. We add one more to the line plot so it isn't short of a full circle.
scat_s=ax.plot(smallxpos[0:int(resperiod)+1],smallypos[0:int(resperiod)+1],lw=0.5, color='yellow')[0]
scat_l=ax.scatter(largexpos[0:int(resperiod)],largeypos[0:int(resperiod)],s=0.5,color='blue')
date=ax.text(1.2*maxradius,-1.2*maxradius,largebody_position.t[0].tt_strftime('%Y %b %d'),size='large', verticalalignment='bottom', horizontalalignment='right') #in the corner

def update(frame):
    # for each frame, update the data stored on each artist.
    smallx = smallxpos[int((frame)*resperiod):min(int((frame+1)*resperiod)+1,n_timesteps)]
    smally = smallypos[int((frame)*resperiod):min(int((frame+1)*resperiod)+1,n_timesteps)]
    # update the line plot:
    #smalldata = np.stack([smallx, smally]).T
    scat_s.set_xdata(smallx)
    scat_s.set_ydata(smally)
    # for each frame, update the data stored on each artist.
    largex = largexpos[int((frame)*resperiod):min(int((frame+1)*resperiod),n_timesteps)]
    largey = largeypos[int((frame)*resperiod):min(int((frame+1)*resperiod),n_timesteps)]
    # update the scatter plot:
    largedata = np.stack([largex, largey]).T
    scat_l.set_offsets(largedata)
    date.set_text(largebody_position.t[int((frame)*resperiod)].tt_strftime('%Y %b %d'))
    return (scat_s, scat_l)


ani = animation.FuncAnimation(fig=fig, func=update, frames=int(np.floor(n_timesteps/resperiod)), interval=frameinterval)
ani.save(outputanim)
plt.show()