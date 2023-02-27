# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 10:54:46 2022

@author: payam
"""
from hcipy import *

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import os
import torch
import pickle
import pandas as pd

import gym
from gym import error, spaces, utils
from gym.utils import seeding

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
class AOEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):    
        super(AOEnv, self).__init__()
        
        parameters = {# telescope configuration
                     
                     'telescope_diameter' : 0.5, # meter
                     #'central_obscuration' : 0.075, # meter   # Ross 0.5/3
                     #'spider_width' : 0.02, # meter
                     'num_pupil_pixels' : 240,
                     
                     # wavelength configuration
                     'wavelength_wfs' : 1.5e-6,
                     'wavelength_sci' : 2.2e-6,
                     
                     # wavefront configuration
                     'f_number' : 50,
                     'num_lenslets' : 40,   # for num_lenslets = 12, num_modes > 100 does not work
                     'sh_diameter' : 5e-3, # m
                     'stellar_magnitude' : -5,
                     
                     # deformable mirror configuration
                     'num_modes' : 64,
                     
                     # atmosphere configuration
                     'delta_t' : 1e-3,
                     'max_steps' : 30,
                     'velocity': 10,
                     'velocity_mult': 0,      # velocity multiplier to make atm. faster or slower
                     
                     'fried_parameter' : 0.1, # arcsec @ 500nm (convention)
                     'outer_scale' : 10, # meter
                     'tau0' : 0.001, # seconds
                     
                     # Fiber configuration
                     'num_pupil_pixels_fiber' : 128,
                     'num_focal_pixels_fiber' : 128,
                     'num_focal_pixels_fiber_subsample' : 2,
                     'D_pupil_fiber' : 0.5,
                     'multimode_fiber_core_radius' : 25 * 1e-6,
                     'singlemode_fiber_core_radius' : 4 * 1e-6,
                     'fiber_NA' : 0.14,
                     'fiber_length' : 10,
                     }
        
        for param, val in parameters.items():
            if isinstance(val, str) == False:
                exec('self.' + param + ' = ' + str(val))
        
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.num_focal_pixels_fiber_subsample**2, ), dtype=np.float16)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_modes, ), dtype=np.float16) 
        
        self.shack_count = 0
            
        #------------------------ simulating the pupil -----------------------#
        #central_obscuration_ratio = self.central_obscuration/ self.telescope_diameter
        
        num_pupil_pixels = self.num_pupil_pixels
        pupil_grid_diameter = self.telescope_diameter
        pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)
        
        #aperture_generator = make_obstructed_circular_aperture(self.telescope_diameter,
        #    central_obscuration_ratio, num_spiders=4, spider_width=self.spider_width)
        
        #self.aperture = evaluate_supersampled(aperture_generator, pupil_grid, 4)
        self.aperture = circular_aperture(self.telescope_diameter)(pupil_grid)
        #------------------------- Incoming wavefront ------------------------#
        wf = Wavefront(self.aperture, self.wavelength_sci)
        wf.total_power = 1
        
        #-------------------------- wavefront sensor -------------------------#
        spatial_resolution = self.wavelength_sci / self.telescope_diameter
        self.focal_grid = make_focal_grid(q=4, num_airy=30, spatial_resolution=spatial_resolution)
        
        self.propagator = FraunhoferPropagator(pupil_grid, self.focal_grid)
        
        self.focal_grid_subsample = make_pupil_grid(self.num_focal_pixels_fiber_subsample, pupil_grid_diameter)
        
        self.propagator_subsample = FraunhoferPropagator(pupil_grid, self.focal_grid_subsample)
        
        self.unaberrated_PSF = self.propagator.forward(wf).power
        self.unaberrated_PSF_subsample = self.propagator_subsample.forward(wf).power
        
        #------------------------- Deformable mirror -------------------------#
        dm_modes = make_disk_harmonic_basis(pupil_grid, self.num_modes, self.telescope_diameter, 'neumann')
        dm_modes = ModeBasis([mode / np.ptp(mode) for mode in dm_modes], pupil_grid)
        self.deformable_mirror = DeformableMirror(dm_modes)
        
        zero_magnitude_flux = 3.9e10 #3.9e10 photon/s for a mag 0 star
        
        self.wf_wfs = Wavefront(self.aperture, self.wavelength_wfs)
        self.wf_wfs_fiber = Wavefront(self.aperture, self.wavelength_wfs)
        self.wf_sci= Wavefront(self.aperture, self.wavelength_sci)
        
        self.wf_wfs.total_power = zero_magnitude_flux * 10**(-self.stellar_magnitude / 2.5)
        self.wf_wfs_fiber.total_power = 1
        self.wf_sci.total_power = zero_magnitude_flux * 10**(-self.stellar_magnitude / 2.5)
        
        self.deformable_mirror.actuators = np.random.randn(self.num_modes) / (np.arange(self.num_modes) + 10)
        self.deformable_mirror.actuators *= 0.3 * self.wavelength_sci / np.std(self.deformable_mirror.surface)
        
        Cn_squared = Cn_squared_from_fried_parameter(self.fried_parameter, self.wavelength_sci)
        # velocity = 0.314 * fried_parameter / self.tau0
        velocity = self.velocity
        
        self.layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, self.outer_scale, velocity)
        
        with open('layer.pkl', 'wb') as f:
            pickle.dump(self.layer, f)

        # self.layer = pd.read_pickle('layer.pkl')
        
    
        phase_screen_phase = self.layer.phase_for(self.wavelength_wfs) # in radian
        phase_screen_opd = phase_screen_phase * (self.wavelength_wfs / (2 * np.pi)) * 1e6

        pupil_grid_fiber = make_pupil_grid(self.num_pupil_pixels_fiber, self.D_pupil_fiber)
        
        aperture_fiber = circular_aperture(self.D_pupil_fiber)(pupil_grid_fiber)
        
        self.multi_mode_fiber = StepIndexFiber(self.multimode_fiber_core_radius, self.fiber_NA, self.fiber_length)
        single_mode_fiber = StepIndexFiber(self.singlemode_fiber_core_radius, self.fiber_NA, self.fiber_length)
        
        D_focus_fiber = 2.1 * self.multimode_fiber_core_radius
        focal_grid_fiber = make_pupil_grid(self.num_focal_pixels_fiber, D_focus_fiber)
        focal_grid_fiber_subsample = make_pupil_grid(self.num_focal_pixels_fiber_subsample, D_focus_fiber)
        
        focal_length = self.D_pupil_fiber/(2 * self.fiber_NA)
        self.propagator_fiber = FraunhoferPropagator(pupil_grid_fiber, focal_grid_fiber, focal_length=focal_length)
        self.propagator_fiber_subsample = FraunhoferPropagator(pupil_grid_fiber, focal_grid_fiber_subsample, focal_length=focal_length)
        
        self.deformable_mirror.flatten()
        
        self.layer.t = 0
        self.timestep_render = 0

#--------------------------- shack __init__ ----------------------------------#
        magnification = self.sh_diameter / self.telescope_diameter
        self.magnifier = Magnifier(magnification)
        
        self.shwfs = SquareShackHartmannWavefrontSensorOptics(pupil_grid.scaled(magnification), self.f_number, \
                                                          self.num_lenslets, self.sh_diameter)
        self.shwfse = ShackHartmannWavefrontSensorEstimator(self.shwfs.mla_grid, self.shwfs.micro_lens_array.mla_index)
        
        self.camera = NoiselessDetector(self.focal_grid)
        
        wf = Wavefront(self.aperture, self.wavelength_wfs)
        self.camera.integrate(self.shwfs(self.magnifier(wf)), 1)
        
        image_ref = self.camera.read_out()
        
        fluxes = ndimage.measurements.sum(image_ref, self.shwfse.mla_index, self.shwfse.estimation_subapertures)
        flux_limit = fluxes.max() * 0.5
        
        estimation_subapertures = self.shwfs.mla_grid.zeros(dtype='bool')
        estimation_subapertures[self.shwfse.estimation_subapertures[fluxes > flux_limit]] = True
        
        self.shwfse = ShackHartmannWavefrontSensorEstimator(self.shwfs.mla_grid, self.shwfs.micro_lens_array.mla_index, estimation_subapertures)
        
        self.slopes_ref = self.shwfse.estimate([image_ref])
        
        self.deformable_mirror_shack = DeformableMirror(dm_modes)
        
        probe_amp = 0.01 * self.wavelength_wfs
        response_matrix = []
        
        wf = Wavefront(self.aperture, self.wavelength_wfs)
        wf.total_power = 1
        
        # Set up animation
        plt.figure(figsize=(10, 6))
        
        for i in range(self.num_modes):
            slope = 0
        
            # Probe the phase response
            amps = [-probe_amp, probe_amp]
            for amp in amps:
                self.deformable_mirror_shack.flatten()
                self.deformable_mirror_shack.actuators[i] = amp
        
                dm_wf = self.deformable_mirror_shack.forward(wf)
                wfs_wf = self.shwfs(self.magnifier(dm_wf))
        
                self.camera.integrate(wfs_wf, 1)
                image = self.camera.read_out()
        
                slopes = self.shwfse.estimate([image])
        
                slope += amp * slopes / np.var(amps)
        
            response_matrix.append(slope.ravel())
        
            # Only show all modes for the first 40 modes
            if i > 40 and (i + 1) % 20 != 0:
                continue
        
        response_matrix = ModeBasis(response_matrix)
        
        rcond = 1e-3
        
        self.reconstruction_matrix = inverse_tikhonov(response_matrix.transformation_matrix, rcond=rcond)
        
        
#-------------------------------- reset --------------------------------------#
    def reset(self):
        
        
        self.deformable_mirror.flatten()

        self.layer.t = self.timestep_render * self.velocity_mult * self.delta_t
        self.timestep = 0
        
        self.phase_screen_phase = self.layer.phase_for(self.wavelength_wfs) # in radian
        self.phase_screen_opd = self.phase_screen_phase * (self.wavelength_wfs / (2 * np.pi)) * 1e6
        
        # Propagate through atmosphere and deformable mirror.
        wf_wfs_after_atmos = self.layer(self.wf_wfs)
        # wf_wfs_after_dm = self.deformable_mirror_shack(wf_wfs_after_atmos)
        
        wf_wfs_after_atmos_fiber = self.layer(self.wf_wfs_fiber)
        wf_wfs_after_dm_fiber = self.deformable_mirror(wf_wfs_after_atmos_fiber)
        self.wf_wfs_after_foc = self.propagator_fiber(wf_wfs_after_dm_fiber)
        self.wf_wfs_after_foc_subsample = self.propagator_fiber_subsample(wf_wfs_after_dm_fiber)
        
        # Propagate the NIR wavefront
        self.wf_sci_focal_plane = self.propagator(self.deformable_mirror(self.layer(self.wf_sci)))
        # self.wf_sci_focal_plane_shack = self.propagator(self.deformable_mirror_shack(self.layer(self.wf_sci)))
        
        state = self.wf_wfs_after_foc_subsample.power
        strehl = get_strehl_from_focal(self.wf_sci_focal_plane.power, self.unaberrated_PSF * self.wf_wfs.total_power) * 100

        return state


#-----------------------------------------------------------------------------#
#--------------------------------- step --------------------------------------#
    def step(self, u):
        
        done = False
        if self.shack_count == 1:
            self.deformable_mirror.actuators = u
            
        else:
            self.deformable_mirror.actuators = u / (np.arange(self.num_modes) + 10)
            
            # instead of (np.std(self.deformable_mirror.surface)), average value of 0.0817 is used ...
            self.deformable_mirror.actuators *= 0.1 * self.wavelength_sci / (np.std(self.deformable_mirror.surface))
        
        
        self.timestep += 1
        self.timestep_render += 1
        
        self.layer.t = self.timestep_render * self.velocity_mult * self.delta_t
        self.phase_screen_phase = self.layer.phase_for(self.wavelength_wfs) # in radian
        self.phase_screen_opd = self.phase_screen_phase * (self.wavelength_wfs / (2 * np.pi)) * 1e6
        
        wf_wfs_after_atmos_fiber = self.layer(self.wf_wfs_fiber)
        wf_wfs_after_dm_fiber = self.deformable_mirror(wf_wfs_after_atmos_fiber)
        self.wf_wfs_after_foc = self.propagator_fiber(wf_wfs_after_dm_fiber)
        self.wf_wfs_after_foc_subsample = self.propagator_fiber_subsample(wf_wfs_after_dm_fiber)
    
        # Propagate the NIR wavefront
        self.wf_sci_focal_plane = self.propagator(self.deformable_mirror(self.layer(self.wf_sci)))
        
        next_state = self.wf_wfs_after_foc_subsample.power    
        
        strehl = get_strehl_from_focal(self.wf_sci_focal_plane.power, self.unaberrated_PSF * self.wf_wfs.total_power) * 100

        cost_strehl = - (100 - strehl)
        
        # cost_focal = -(self.wf_sci_focal_plane.total_power/self.wf_sci_focal_plane.power.max())
        
        cost = cost_strehl
        if self.timestep == self.max_steps:
            done = True
            
        return next_state, float(cost), done, float(strehl)


#-----------------------------------------------------------------------------#
    def shack_step(self):
        
        # print(i_so_far)
        burn_in_iterations = 0
        gain = 0.3
        leakage = 0.01
        self.shack_count = 1
        
            
        wf_wfs_after_atmos = self.layer(self.wf_wfs)
        wf_wfs_after_dm = self.deformable_mirror_shack(wf_wfs_after_atmos)
        wf_wfs_on_sh = self.shwfs(self.magnifier(wf_wfs_after_dm))
        
        # Read out WFS camera
        self.camera.integrate(wf_wfs_on_sh, self.delta_t)
        self.wfs_image = self.camera.read_out()
        self.wfs_image = large_poisson(self.wfs_image).astype('float')
        
        # Calculate slopes from WFS image
        slopes = self.shwfse.estimate([self.wfs_image + 1e-10])
        slopes -= self.slopes_ref
        slopes = slopes.ravel()
            
        self.deformable_mirror_shack.actuators = (1 - leakage) * self.deformable_mirror_shack.actuators - gain * self.reconstruction_matrix.dot(slopes)
            
        u = self.deformable_mirror_shack.actuators
        log_u = torch.tensor([1])    # this line is just to make shack compatible with other algorithms
        
        return u, log_u
    
#-----------------------------------------------------------------------------#
#---------------------------- plot rendering ---------------------------------#
    def render(self, mode='human', close=False):
        
        plt.clf()
        plt.suptitle('timestep %d / %d' % (self.timestep+1, self.max_steps))
        
        plt.subplot(2,2,1)
        plt.title('Atmosphere')
        imshow_field(self.phase_screen_opd, vmin=-6, vmax=6, cmap='RdBu')
        plt.colorbar()
        
        plt.subplot(2,2,3)
        # plt.title('Instantaneous PSF at 2.2$\\mu$m [log]')
        imshow_field(np.log10(self.wf_sci_focal_plane.power/ self.wf_sci_focal_plane.power.max()), vmin=-6, vmax=0, cmap='inferno') #
        plt.colorbar()
        
        plt.subplot(2,2,4)
        # plt.title('Focal plane')
        imshow_field(self.wf_wfs_after_foc.power)
        circ = plt.Circle((0, 0), self.multimode_fiber_core_radius, edgecolor='white', fill=False, linewidth=2, alpha=0.5)
        plt.gca().add_artist(circ)
        plt.xlabel('x (um)')
        plt.ylabel('y (um)')
        
        plt.show()
        plt.clf()
        plt.close()
            
            
#-----------------------------------------------------------------------------#

