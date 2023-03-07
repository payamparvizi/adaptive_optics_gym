"""
        This file presents Adaptive Optics Gym environment
"""


#----------------------------- Importing modules -----------------------------#
from hcipy import *
import gym
from gym import spaces
import numpy as np
import pickle
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import torch
import pandas as pd


#-----------------------------------------------------------------------------#
#------------------------------ Initialization -------------------------------#
class AOEnv(gym.Env):
    def __init__(self):    
        super(AOEnv, self).__init__()

        # The parameters used for the simulation
        parameters = {
            # telescope configuration:
            'telescope_diameter': 0.5,                     # diameter of the telescope in meters

            # pupil configuration
            'num_pupil_pixels' : 240,                      # Number of pupil grid pixels

            # wavefront configuration
            'wavelength_wfs' : 1.5e-6,                     # wavelength of wavefront sensing in micro-meters
            'wavelength_sci' : 2.2e-6,                     # wavelength of scientific channel in micro-meters

            # deformable mirror configuration
            'num_modes' : 64,                              # Number of actuators in Deformable mirror

            # Atmosphere configuration
            'delta_t': 1e-3,                               # in seconds, for a loop speed of 1 kHz
            'max_steps': 30,                               # Maximum number of timesteps in an episode
            'velocity': 0,                                # the velocity of attmosphere
            'fried_parameter' : 0.3,                       # The Fried parameter in meters
            'outer_scale' : 10,                            # The outer scale of the phase structure function in meters

            # Fiber configuration
            'D_pupil_fiber' : 0.5,                         # Diameter of the pupil for fiber
            'num_pupil_pixels_fiber': 128,                 # Number of pupil grid pixels for fiber
            'num_focal_pixels_fiber' : 128,                # Number of focal grid pixels for fiber
            'num_focal_pixels_fiber_subsample' : 2,        # Number of focal grid pixels for fiber subsampled for quadrant photodetector
            'multimode_fiber_core_radius' : 25 * 1e-6,     # the radius of the multi-mode fiber
            'singlemode_fiber_core_radius' : 9 * 1e-6,     # the radius of the single-mode fiber
            'fiber_NA' : 0.14,                             # Fiber numerical aperture 

            # wavefront sensor configuration
            'f_number' : 50,                               # F-ratio
            'num_lenslets' : 12,                           # Number of lenslets along one diameter
            'sh_diameter' : 5e-3,                          # diameter of the sensor in meters
            'stellar_magnitude' : -5,                      # measurement of brightness for stars and other objects in space
            }

        for param, val in parameters.items():
            if isinstance(val, str) == False:
                exec('self.' + param + ' = ' + str(val))

        # Extract out dimensions of observation and action spaces
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.num_focal_pixels_fiber_subsample**2, ), dtype=np.float16)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_modes, ), dtype=np.float16)


        #------------------------ simulating the pupil -----------------------#
        # We model the the diameter of the pupil of a telescope as a function of the telescope's diameter
        pupil_grid_diameter = self.telescope_diameter
        pupil_grid = make_pupil_grid(self.num_pupil_pixels, pupil_grid_diameter)

        # Definition of the aperture
        aperture = circular_aperture(self.telescope_diameter)(pupil_grid)


        #------------------------- Incoming wavefront ------------------------#
        # Simulation of the incoming wavefront from the satellite

        # propagation of a wavefront through a perfect lens
        spatial_resolution = self.wavelength_sci / self.telescope_diameter                         # The physical size of a resolution element
        focal_grid = make_focal_grid(q=4, num_airy=30, spatial_resolution=spatial_resolution)      # Make a grid for a focal plane

        self.propagator = FraunhoferPropagator(pupil_grid, focal_grid)

        # created unaberrated point spread function (ideal propagation) for future comparison for Strehl ratio
        wf = Wavefront(aperture, self.wavelength_sci)
        wf.total_power = 1
        self.unaberrated_PSF = self.propagator.forward(wf).power

        zero_magnitude_flux = 3.9e10 #3.9e10 photon/s for a mag 0 star
        
        # generate the wavefront sensing
        self.wf_wfs = Wavefront(aperture, self.wavelength_wfs)
        self.wf_wfs.total_power = zero_magnitude_flux * 10**(-self.stellar_magnitude / 2.5)

        # generate the wavefront sensing with total power of 1 for fiber coupling
        self.wf_wfs_fiber = Wavefront(aperture, self.wavelength_wfs)
        self.wf_wfs_fiber.total_power = 1

        # generate the wavefront of scientific channel
        self.wf_sci= Wavefront(aperture, self.wavelength_sci)
        self.wf_sci.total_power = zero_magnitude_flux * 10**(-self.stellar_magnitude / 2.5)


        #------------------------- Deformable mirror -------------------------#
        # generate the deformable mirror with number of actuators
        dm_modes = make_disk_harmonic_basis(pupil_grid, self.num_modes, self.telescope_diameter, 'neumann')
        dm_modes = ModeBasis([mode / np.ptp(mode) for mode in dm_modes], pupil_grid)
        self.deformable_mirror = DeformableMirror(dm_modes)

        # start the deformable mirror with neutral position (flat):
        self.deformable_mirror.flatten()


        #---------------------- Atmospheric Turbulence -----------------------#
        # Simulating the atmosphere with parameters given before

        # Calculate the integrated Cn^2 for a certain Fried parameter
        Cn_squared = Cn_squared_from_fried_parameter(self.fried_parameter, self.wavelength_sci)

        # create the layer for the atmosphere
        self.layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, self.outer_scale, self.velocity)

        # To save the atmosphere, we can use:
        with open('layer.pkl', 'wb') as f:
            pickle.dump(self.layer, f)

        # To load the saved atmosphere, uncomment below line and comment out line above
        # self.layer = pd.read_pickle('layer1.pkl')


        #-------------------------- Fiber coupling ---------------------------#
        pupil_grid_fiber = make_pupil_grid(self.num_pupil_pixels_fiber, self.D_pupil_fiber)

        # The diameter of the grid for fiber
        D_focus_fiber = 2.1 * self.multimode_fiber_core_radius

        # make the grid for focal plane before fiber
        focal_grid_fiber = make_pupil_grid(self.num_focal_pixels_fiber, D_focus_fiber)
        focal_grid_fiber_subsample = make_pupil_grid(self.num_focal_pixels_fiber_subsample, D_focus_fiber)

        # propagation of a wavefront through a focal plane before fiber
        focal_length = self.D_pupil_fiber/(2 * self.fiber_NA)                  # The focal length of the lens system.
        self.propagator_fiber = FraunhoferPropagator(pupil_grid_fiber, focal_grid_fiber, focal_length=focal_length)
        self.propagator_fiber_subsample = FraunhoferPropagator(pupil_grid_fiber, focal_grid_fiber_subsample, focal_length=focal_length)


        #------------------------ Shack-Hartmann WS --------------------------#
        # This part is for the initialization of the Shack Hartmann wavefront sensor
        # This part can be commented out if you are not interested in "Shack-Hartmann"

        # The diameter of the beam needs to be reshaped with a magnifier, otherwise 
        # the spots are not resolved by the pupil grid
        magnification = self.sh_diameter / self.telescope_diameter
        self.magnifier = Magnifier(magnification)
        
        # create the shack-hartmann wavefront sensor:
        self.shwfs = SquareShackHartmannWavefrontSensorOptics(pupil_grid.scaled(magnification), \
                                                              self.f_number, self.num_lenslets, self.sh_diameter)
        self.shwfse = ShackHartmannWavefrontSensorEstimator(self.shwfs.mla_grid, self.shwfs.micro_lens_array.mla_index)

        # create the noiseless detector for Shack-Hartmann
        self.camera = NoiselessDetector(focal_grid)
        wf_camera = Wavefront(aperture, self.wavelength_wfs)
        self.camera.integrate(self.shwfs(self.magnifier(wf_camera)), 1)
        image_ref = self.camera.read_out()

        # select subapertures to use for wavefront sensing, based on their flux:
        fluxes = ndimage.measurements.sum(image_ref, self.shwfse.mla_index, self.shwfse.estimation_subapertures)
        flux_limit = fluxes.max() * 0.5

        # generate the Shack-Hartmann wavefront sensor estimator:
        estimation_subapertures = self.shwfs.mla_grid.zeros(dtype='bool')
        estimation_subapertures[self.shwfse.estimation_subapertures[fluxes > flux_limit]] = True

        self.shwfse = ShackHartmannWavefrontSensorEstimator(self.shwfs.mla_grid, self.shwfs.micro_lens_array.mla_index, estimation_subapertures)

        # calculate reference slopes
        self.slopes_ref = self.shwfse.estimate([image_ref])

        # create a deformable mirror for shack-hartmann to prevent any confusion
        self.deformable_mirror_shack = DeformableMirror(dm_modes)

        # calibrating the interaction matrix:
        probe_amp = 0.01 * self.wavelength_wfs
        response_matrix = []

        wf_cal = Wavefront(aperture, self.wavelength_wfs)
        wf_cal.total_power = 1

        for i in range(self.num_modes):
            slope = 0

            # Probe the phase response
            amps = [-probe_amp, probe_amp]
            for amp in amps:
                self.deformable_mirror_shack.flatten()
                self.deformable_mirror_shack.actuators[i] = amp

                dm_wf = self.deformable_mirror_shack.forward(wf_cal)
                wfs_wf = self.shwfs(self.magnifier(dm_wf))

                self.camera.integrate(wfs_wf, 1)
                image = self.camera.read_out()

                slopes = self.shwfse.estimate([image])

                slope += amp * slopes / np.var(amps)

            response_matrix.append(slope.ravel())

        response_matrix = ModeBasis(response_matrix)

        # inversion of interaction matrix using Tikhonov regularization
        rcond = 1e-3
        self.reconstruction_matrix = inverse_tikhonov(response_matrix.transformation_matrix, rcond=rcond)

        # check if we are using Shack-Hartmann or not
        self.shack_operation = False


#-----------------------------------------------------------------------------#
#---------------------------------- reset ------------------------------------#
    def reset(self):

        # start the episode with flat mirror:
        self.deformable_mirror.flatten()

        # start the environment at time 0 sec
        done = False
        self.timestep = 0
        self.layer.t = self.timestep * self.delta_t

        # get the phase screen for plot
        self.phase_screen_phase = self.layer.phase_for(self.wavelength_wfs)    # Get the phase screen in radians
        self.phase_screen_opd = self.phase_screen_phase * (self.wavelength_wfs / (2 * np.pi)) * 1e6

        # Propagatation of wavefront through atmosphere
        wf_wfs_after_atmos = self.layer(self.wf_wfs_fiber)

        # Propagatation of wavefront through deformable mirror
        wf_wfs_after_dm = self.deformable_mirror(wf_wfs_after_atmos)

        # Propagatation of wavefront through focal plane before fiber
        self.wf_wfs_after_foc = self.propagator_fiber(wf_wfs_after_dm)
        self.wf_wfs_after_foc_subsample = self.propagator_fiber_subsample(wf_wfs_after_dm)     # subsampled for quadrant photodetector

        # The observation - Power of the wavefront propagated through the focal plane
        state = self.wf_wfs_after_foc_subsample.power

        return state


#-----------------------------------------------------------------------------#
#---------------------------------- step -------------------------------------#
    def step(self, u):

        # Shack-Hartmann creates normalized action which can be used directly in this function,
        # However, the action generated by Actor needs to be normalized in here
        if self.shack_operation == True:
            self.deformable_mirror.actuators = u

        else:
            # Normalize the DM surface to get a reasonable surface RMS
            self.deformable_mirror.actuators = u / (np.arange(self.num_modes) + 10)
            self.deformable_mirror.actuators *= 0.1 * self.wavelength_sci / (np.std(self.deformable_mirror.surface))

        # The next time of the atmospheric layer
        self.timestep += 1
        self.layer.t = self.timestep * self.delta_t

        # get the phase screen for plot
        self.phase_screen_phase = self.layer.phase_for(self.wavelength_wfs)    # Get the phase screen in radians
        self.phase_screen_opd = self.phase_screen_phase * (self.wavelength_wfs / (2 * np.pi)) * 1e6

        # Propagatation of wavefront through atmosphere
        wf_wfs_after_atmos = self.layer(self.wf_wfs_fiber)

        # Propagatation of wavefront through deformable mirror
        wf_wfs_after_dm = self.deformable_mirror(wf_wfs_after_atmos)

        # Propagatation of wavefront through focal plane before fiber
        self.wf_wfs_after_foc = self.propagator_fiber(wf_wfs_after_dm)
        self.wf_wfs_after_foc_subsample = self.propagator_fiber_subsample(wf_wfs_after_dm)     # subsampled for quadrant photodetector

        # The observation - Power of the wavefront propagated through the focal plane
        next_state = self.wf_wfs_after_foc_subsample.power

        # Propagate the Near-Infrared wavefront
        self.wf_sci_focal_plane = self.propagator(self.deformable_mirror(self.layer(self.wf_sci)))

        # calculate the strehl ratio and the cost
        strehl_ratio = get_strehl_from_focal(self.wf_sci_focal_plane.power, self.unaberrated_PSF * self.wf_wfs.total_power) * 100
        cost = - (100 - strehl_ratio)

        # check if done or not:
        if self.timestep == self.max_steps:
            done = True
        else:
            done = False

        return next_state, float(cost), done, float(strehl_ratio)


#-----------------------------------------------------------------------------#
#--------------------------------- render ------------------------------------#
    def render(self, close=False):

        plt.suptitle('timestep %d / %d' % (self.timestep+1, self.max_steps))

        # plot of the atmosphrere
        plt.subplot(2,2,1)
        plt.title('Atmosphere')
        imshow_field(self.phase_screen_opd, vmin=-6, vmax=6, cmap='RdBu')
        plt.colorbar()

        # plot of the wavefront power after focal plane before fiber
        plt.subplot(2,2,2)
        imshow_field(self.wf_wfs_after_foc.power)
        circ = plt.Circle((0, 0), self.singlemode_fiber_core_radius, edgecolor='white', fill=False, linewidth=2, alpha=0.5)
        plt.gca().add_artist(circ)
        plt.xlabel('x (um)')
        plt.ylabel('y (um)')
        
        plt.show()


#-----------------------------------------------------------------------------#
#------------------------- Shack-Hartmann action -----------------------------#
    """
            Since Shack-Hartmann uses information from the environment directly, 
            the action needs to be generated in here.
    """
    def shack_get_action(self):

        # show that we are using Shack-Hartmann
        self.shack_operation = True

        # Propagatation of wavefront through atmosphere
        wf_wfs_after_atmos = self.layer(self.wf_wfs)

        # Propagatation of wavefront through deformable mirror
        wf_wfs_after_dm = self.deformable_mirror_shack(wf_wfs_after_atmos)

        # Propagatation of wavefront through Shack-Hartmann wavefront sensor
        wf_wfs_on_sh = self.shwfs(self.magnifier(wf_wfs_after_dm))

        # Read out WFS camera
        self.camera.integrate(wf_wfs_on_sh, self.delta_t)
        self.wfs_image = self.camera.read_out()
        self.wfs_image = large_poisson(self.wfs_image).astype('float')

        # # calculate slopes from WFS image
        slopes = self.shwfse.estimate([self.wfs_image + 1e-10])
        slopes -= self.slopes_ref
        slopes = slopes.ravel()

        # generate the next action
        gain = 0.3
        leakage = 0.01
        self.deformable_mirror_shack.actuators = (1 - leakage) * \
            self.deformable_mirror_shack.actuators - gain * self.reconstruction_matrix.dot(slopes)

        u = self.deformable_mirror_shack.actuators
        log_u = torch.tensor([1])                      # # this line is just to make Shack-Hartmann compatible with other algorithms

        return u, log_u


#-----------------------------------------------------------------------------#