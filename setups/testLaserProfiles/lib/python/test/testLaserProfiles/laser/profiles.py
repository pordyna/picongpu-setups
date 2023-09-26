import numpy as np
import scipy.constants as cs


class GaussianPulse:
    def __init__(self, tau, t_shift=0):
        self.tau = tau
        self.t_shift = t_shift

    def __call__(self, t, cut=True):
        t_old = t
        t = t + self.t_shift
        res =  np.exp(-2 * np.log(2) * (t / self.tau) **2)
        if cut:
            res[t_old<=0] = 0
        return res


def _extrapolate_expo(t1, a1, t2, a2, t):
    log1 = (t2 - t) * np.log(a1)
    log2 = (t - t1) * np.log(a2)
    return np.exp((log1 + log2) / (t2 - t1))


class ExpRampWithPrepulse:
    def __init__(self, int_ratio_prepulse,
                 int_ratio_point_1,
                 int_ratio_point_2,
                 int_ratio_point_3,
                 ramp_init,
                 time_1,
                 time_2,
                 time_3,
                 time_prepulse,
                 tau,
                 time_peakpulse=0,
                 laser_nofocus_constant=0):
        self.amp_prepulse = np.sqrt(int_ratio_prepulse)
        self.amp_1 = np.sqrt(int_ratio_point_1)
        self.amp_2 = np.sqrt(int_ratio_point_2)
        self.amp_3 = np.sqrt(int_ratio_point_3)
        self.time_start_init = time_1 - (0.5 * ramp_init * tau)
        self.time_1 = time_1
        self.time_2 = time_2
        self.time_3 = time_3
        self.time_prepulse = time_prepulse
        self.time_peakpulse = time_peakpulse
        self.gauss = GaussianPulse(tau)
        self.laser_nofocus_constant = laser_nofocus_constant

    def __call__(self, t):
        amplitude = np.zeros_like(t)
        mask_1 = t < self.time_start_init
        amplitude[mask_1] = 0.0
        mask_2 = ~mask_1 & (t < self.time_1)
        amplitude[mask_2] = self.amp_1 * self.gauss(t[mask_2] - self.time_1, cut=False)
        end_upramp = self.time_peakpulse - 0.5 * self.laser_nofocus_constant
        mask_3 = ~mask_1 & ~mask_2 & (t < end_upramp)
        ramp_when_peakpulse = _extrapolate_expo(self.time_2, self.amp_2,
                                                self.time_3, self.amp_3,
                                                end_upramp)
        amplitude[mask_3] = (1 - ramp_when_peakpulse) * self.gauss(t[mask_3] - end_upramp, cut=False)
        amplitude[mask_3] += self.amp_prepulse * self.gauss(t[mask_3] - self.time_prepulse, cut=False)
        mask_4 = mask_3 & (self.time_1 < t) & (t < self.time_2)
        amplitude[mask_4] += _extrapolate_expo(self.time_1, self.amp_1,
                                      self.time_2, self.amp_2, t[mask_4])
        mask_5 = ~mask_4 & mask_3
        amplitude[mask_5] += _extrapolate_expo(self.time_2, self.amp_2,
                                              self.time_3, self.amp_3, t[mask_5])
        mask_6 = ~mask_1 & ~mask_2 & ~mask_3 & (t < self.time_peakpulse + 0.5 * self.laser_nofocus_constant)
        amplitude[mask_6] = 1
        mask_7 = t > self.time_peakpulse + 0.5 * self.laser_nofocus_constant
        amplitude[mask_7] = self.gauss(
                t[mask_7] - self.time_peakpulse + 0.5 * self.laser_nofocus_constant, cut=False)
        return amplitude


class GaussianBeam:
    def __init__(self, w_0, focus_position, wavelength, temporal_shape,
                 speed_of_light=None):
        self.w_0 = w_0 / 1.17741
        self.focus_position = focus_position
        self.wavelength = wavelength
        self.rayleigh = np.pi * w_0 ** 2 / wavelength
        self.k = 2 * np.pi / wavelength
        self.temporal_shape = temporal_shape
        if speed_of_light is None:
            self.speed_of_light = cs.c * 1e6*1e-15
        else:
            self.speed_of_light = speed_of_light

    def _w(self, z):
        return self.w_0 * np.sqrt(1 + (z / self.rayleigh) ** 2)

    def _gouy_phase(self, z):
        return np.arctan(z / self.rayleigh)

    def _curvature_radius(self, z):
        return z * (1 + (self.rayleigh / z) ** 2)

    def _delta_z(self, r, z):
        return r ** 2 / 2 / self._curvature_radius(z)

    def _e_trans(self, r, z, t):
        w = self._w(z)
        res = self.w_0 / w * np.exp(-r ** 2 / w ** 2)

        phase = self.k * z + self.k * self._delta_z(r, z) - self._gouy_phase(z) -self.k * self.speed_of_light * t
        return res, np.cos(phase)

    def __call__(self, x1, x2, x3, t):
        # x1, x2, x3 are x, y, z in picongpu
        # laser propagates along y in pic and here the axis is called z
        # we just switch the two
        x1 = x1[:, None, None, None]
        x2 = x2[None, :, None, None]
        x3 = x3[None, None, :, None]
        t = t[None, None, None, :]
        r = np.sqrt((x1 - self.focus_position[0]) ** 2
                    + (x3 - self.focus_position[2]) ** 2)
        z = x2 - self.focus_position[1]
        t_axis = t - (z + self._delta_z(r, z)) / self.speed_of_light
        e_trans = self._e_trans(r, z, t)
        return e_trans[0] * self.temporal_shape(t_axis), e_trans[1]
