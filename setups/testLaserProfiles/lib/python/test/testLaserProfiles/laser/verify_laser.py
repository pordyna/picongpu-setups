import json

import openpmd_api as io
import numpy as np
from numpy.typing import NDArray
import scipy.constants as cs
from scipy.ndimage import maximum_filter
from profiles import GaussianPulse, GaussianBeam, ExpRampWithPrepulse
import inspect
from typing import Type
import matplotlib.pyplot as plt

from warnings import warn


def compare(a, b, atol, rtol):
    diff = np.abs(a - b)
    rel_test = diff <= rtol * np.maximum(np.abs(a), np.abs(b))
    abs_test = diff <= atol
    return np.logical_or(rel_test, abs_test)


class ReferenceCalc:
    def __init__(self, **kwargs):
        valid_temporal_shapes = dict(GaussianPulse=GaussianPulse,
                                     ExpRampWithPrepulse=ExpRampWithPrepulse)
        temporal_shape_class: Type = valid_temporal_shapes[
            kwargs['profile']['TemporalShape']]
        temporal_shape = temporal_shape_class(
            **kwargs['profile']['temporal_shape_param'])
        kwargs = kwargs | dict(temporal_shape=temporal_shape)
        self.beam_functor = GaussianBeam(temporal_shape=temporal_shape,
                                         **kwargs['profile']['beam_param'])
        self.t_0 = kwargs['simulation']['t0']

    def __call__(self, x, y, z, t) -> tuple[NDArray, NDArray]:
        """

        :param x: x field coordinates (x as in PIConGPU) in um
        :param y: y field coordinates (y as in PIConGPU) in um
        :param z: z field coordinates (z as in PIConGPU) in um
        :param t: t simulation time in fs (without t_0 shift)
        :return:  (envelope, wave)
            envelope are field values including only envelope,
             wave is the oscillating term. Full field is envelope * wave.
        """
        # probably should get reed of that confusion at some point

        t = t - self.t_0
        return self.beam_functor(x, y, z, t)


def _get_positions(mesh: io.Mesh, mrc: io.Mesh_Record_Component,
                   shape: tuple[int, ...],
                   slicing: tuple[slice, ...], axis: int):
    start = mesh.grid_global_offset[axis] + slicing[axis].start * \
            mesh.grid_spacing[axis]
    pos = np.arange(shape[axis], dtype=np.float64)
    pos += mrc.position[axis]
    pos *= mesh.grid_spacing[axis]
    pos += start

    return pos


def _compare_versions(sim_series_path, ref_series_path, wavelength_si=800e-9,
                      init_surfaces=((16, -16), (16, -16), (16, -16))):
    slicing_list = [slice(x[0] + 1, x[1] - 1) for x in init_surfaces]
    ref_series = io.Series(ref_series_path, io.Access_Type.read_linear)
    sim_series = io.Series(sim_series_path, io.Access_Type.read_linear)
    # we assume for now linear polarization along x
    # iteration loop
    ref_iteration: io.Iteration
    sim_read_iterations = iter(sim_series.read_iterations())
    for ref_iteration in ref_series.read_iterations():
        sim_iteration = next(sim_read_iterations)
        iteration_idx = ref_iteration.iteration_index
        assert sim_iteration.iteration_index == iteration_idx, \
            "Iterations in reference and tested simulation need to match."

        ref_iteration.open()
        sim_iteration.open()

        fields_to_compare = {'E': ['x', 'y', 'z'], 'B': ['x', 'y', 'z']}
        for mesh_name in fields_to_compare.keys():
            sim_mesh: io.Mesh = sim_iteration.meshes[mesh_name]
            ref_mesh: io.Mesh = ref_iteration.meshes[mesh_name]

            sim_time = (sim_iteration.time +
                        sim_mesh.time_offset) * sim_iteration.time_unit_SI
            ref_time = (ref_iteration.time +
                        ref_mesh.time_offset) * ref_iteration.time_unit_SI
            assert np.isclose(sim_time, ref_time, atol=0.0, rtol=1e-6)
            assert sim_mesh.axis_labels == ref_mesh.axis_labels
            grid = np.array(ref_mesh.grid_spacing) * ref_mesh.grid_unit_SI
            max_filter_size = tuple(np.ceil(wavelength_si / grid / 1))
            for component in fields_to_compare[mesh_name]:
                sim_mrc: io.Mesh_Record_Component = sim_mesh[component]
                ref_mrc: io.Mesh_Record_Component = ref_mesh[component]
                assert sim_mrc.ndim == ref_mrc.ndim
                slicing = tuple(slicing_list[:ref_mrc.ndim])
                ref_data = ref_mrc[slicing]
                sim_data = sim_mrc[slicing]
                assert np.isclose(sim_mrc.unit_SI, ref_mrc.unit_SI,
                                  atol=0.0, rtol=1e-6)
                sim_series.flush()
                ref_series.flush()
                epsilon = max(np.finfo(ref_mrc.dtype).eps,
                              np.finfo(sim_mrc.dtype).eps)
                ref_amplitude_estimate = maximum_filter(np.abs(ref_data),
                                                        size=max_filter_size,
                                                        mode='constant',
                                                        cval=0.0)
                test = compare(sim_data, ref_data, rtol=1 * epsilon,
                               atol=100 * ref_amplitude_estimate * epsilon)

                try:
                    assert np.sum(~test) < 1000, \
                    (f"it: {iteration_idx} at {ref_time * 1e15:.2f} fs."
                     f" {mesh_name}_{component} failed test!")
                except AssertionError as err:
                    raise err
                    print(err.args[0])
                print(f"it: {iteration_idx} at {ref_time * 1e15:.2f} fs."
                       f" {mesh_name}_{component} test passed!")


def _verify(sim_series_path, setup_dict):
    series = io.Series(sim_series_path, io.Access_Type.read_linear)
    reference_calc = ReferenceCalc(**setup_dict)
    # we assume for now linear polarization along x
    # iteration loop
    iteration: io.Iteration
    for iteration in series.read_iterations():
        iteration.open()
        time = iteration.time * iteration.time_unit_SI
        mesh: io.Mesh = iteration.meshes['E']
        time += mesh.time_offset * iteration.time_unit_SI
        mrc = mesh['x']
        # Some consistency checks:
        assert mrc.ndim == setup_dict['simulation']['ndim']
        if mrc.ndim == 3:
            assert mesh.axis_labels == ['z', 'y', 'x']
            x_idx, y_idx, z_idx = 2, 1, 0
        else:
            assert mesh.axis_labels == ['y', 'x']
            x_idx, y_idx = 1, 0

        slicing = tuple(
            [slice(a[0], a[1]) for a in setup_dict['simulation']['slicing']])
        Ex = mrc[slicing]
        if mrc.ndim == 3:
            z = _get_positions(mesh, mrc, Ex.shape, slicing, 0)
        else:
            z = np.array([0.0])
        x = _get_positions(mesh, mrc, Ex.shape, slicing, x_idx)
        y = _get_positions(mesh, mrc, Ex.shape, slicing, y_idx)
        # consider the laser origin at the init plane
        y -= mesh.grid_spacing[1] * (16 + 0.75)

        x *= mesh.grid_unit_SI * 1e6
        y *= mesh.grid_unit_SI * 1e6
        z *= mesh.grid_unit_SI * 1e6
        time *= 1e15
        y += setup_dict['profile']['beam_param']['focus_position'][1]

        a0_to_amplitude_si = (-2.0 * np.pi / 800e-9 * cs.electron_mass *
                              cs.c ** 2 / cs.elementary_charge)

        unit = mrc.unit_SI / (
                setup_dict['profile']['a_0'] * a0_to_amplitude_si)
        series.flush()
        sim_dtype = Ex.dtype
        Ex = Ex.astype(np.float64, copy=False)
        Ex *= unit
        time_step = iteration.dt * iteration.time_unit_SI * 1e15

        epsilon = np.finfo(sim_dtype).eps

        # For z>t*c there should be no field and everything should be equal 0
        # but there may be some missmatch in calculating this cutoff.
        # Here we also accept field values if they match reference from +- 1
        # time step.
        def check(t_offset):
            amplitude, phase = reference_calc(x, y, z, time + t_offset)
            if mrc.ndim == 2:
                amplitude = np.squeeze(amplitude)
                phase = np.squeeze(phase)
            full = amplitude * phase
            # transpose Ex to switch from z,y,x to x,y,z:
            result = compare(Ex.T, full, rtol=10 * epsilon,
                             atol=amplitude * epsilon)
            # if not result:
            #     print(iteration.time)
            #     f, (ax1, ax2, ax3) = plt.subplots(3)
            #     vmax = max(np.abs(full).max(), np.abs(Ex).max())
            #     ax1.imshow(full, vmax=vmax, vmin=-vmax)
            #     ax2.imshow(Ex.T, vmax=vmax, vmin=-vmax)
            #     ax3.plot(np.abs(full[122]))
            #     ax3.plot(np.abs(Ex.T[122]))
            #     plt.show()
            return result

        # print(check(0) or check(-1 * time_step) or check(+ 1 * time_step))

        amplitude, phase = reference_calc(x, y, z, time + np.linspace(-1, 1,
                                                                      1024) * time_step)
        amplitude_max = np.max(amplitude, axis=3)
        amplitude, phase = reference_calc(x, y, z, np.array([time]))
        full = amplitude * phase
        # full_max = np.squeeze(np.max(full, axis=3))
        # full_min = np.squeeze(np.min(full, axis=3))
        # if mrc.ndim == 2:
        #     amplitude_0 = np.squeeze(amplitude_0)
        #     amplitude_1 = np.squeeze(amplitude_1)
        #     phase_0 = np.squeeze(phase_0)
        #     phase_1 = np.squeeze(phase_1)
        # full_0 = amplitude_0 * phase_0
        # full_1 = amplitude_1 * phase_1
        # mask_inc = full_1 > full_0
        # check = np.zeros_like(Ex.T, dtype=bool)
        # check[mask_inc] = (Ex.T[mask_inc] >= full_0[mask_inc]) & (Ex.T[
        #      mask_inc] <= full_1[mask_inc])
        # check[~mask_inc] = (Ex.T[~mask_inc] >= full_1[~mask_inc]) & (Ex.T[
        #     ~mask_inc] <= full_0[~mask_inc])

        check = (Ex.T >= -(1 + 1e-1) * np.squeeze(amplitude_max)) & (
                Ex.T <= (1 + 1e-1) * np.squeeze(amplitude_max))
        check2 = compare(Ex.T, np.squeeze(full), rtol=10 * epsilon,
                         atol=10 * np.squeeze(amplitude_max) * epsilon)
        check = np.logical_or(check, check2)
        print(np.all(check))
        print(iteration.time)
        f, (ax1, ax2, ax3) = plt.subplots(3)
        # # vmax = max(np.abs(full).max(), np.abs(Ex).max())
        # # #ax1.imshow(full, vmax=vmax, vmin=-vmax)
        # #ax2.imshow(Ex.T, vmax=vmax, vmin=-vmax)
        im = ax1.imshow(np.squeeze(full.T) - Ex)
        plt.colorbar(im, ax=ax1)
        ax2.imshow(np.squeeze(check.T))
        plt.show()

        # ax3.scatter(y,np.abs(full[111,:, 0, 512]), s=1, c='black')
        # ax3.scatter(y,np.abs(full_min[111]), s=1 , c='green')
        # ax3.scatter(y,np.abs(full_max[111]), s=1, c= 'red')
        # ax3.scatter(y,np.abs(Ex.T[111]), s=1, c = 'blue')
        # plt.show()
        #     plt.show()
        # Ex.T >= full_0
        # transpose Ex to switch from z,y,x to x,y,z:
        #     result = compare(Ex.T, full, rtol=10 * epsilon,
        #                      atol=amplitude * epsilon)
        # if not result:
        #     print(iteration.time)
        #     f, (ax1, ax2, ax3) = plt.subplots(3)
        #     vmax = max(np.abs(full).max(), np.abs(Ex).max())
        #     ax1.imshow(full, vmax=vmax, vmin=-vmax)
        #     ax2.imshow(Ex.T, vmax=vmax, vmin=-vmax)
        #     ax3.plot(np.abs(full[122]))
        #     ax3.plot(np.abs(Ex.T[122]))
        #     plt.show()

        Ey = mesh['y'][slicing]
        Ez = mesh['z'][slicing]
        series.flush()
        # assert np.allclose(Ey, 0.0, atol=1 * epsilon, rtol=0)
        # assert np.allclose(Ez, 0.0, atol=1 * epsilon, rtol=0)


if __name__ == "__main__":
    path_new = "../../../../../simOutput_new/openPMD/simData_%T.bp"
    path_ref = "../../../../../simOutput_ref/openPMD/simData_%T.bp"
    _compare_versions(path_new, path_ref)
    # json_path = "../setups/00_gaussian_pulse_2D.json"
    # with open(json_path, "r") as j:
    #     _verify(path, json.load(j))
