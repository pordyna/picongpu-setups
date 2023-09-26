import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, Normalize
from laser import GaussianBeam, GaussianPulse, ExpRampWithPrepulse
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

def _normalize(x):
    one = np.max(np.abs(x))
    return x/one, one
if __name__ == '__main__':
    t = np.linspace(-1.2e-12, 200e-15, 1000)
    # temporal_shape = GaussianPulse(50e-15)
    temporal_shape = ExpRampWithPrepulse(
        int_ratio_prepulse=1.e-4,
        int_ratio_point_1=1.0e-8,
        int_ratio_point_2=1.0e-5,
        int_ratio_point_3=1.0e-3,
        ramp_init=16,
        time_1=-1.0e-12,
        time_2=-400.0e-15,
        time_3=-100e-15,
        time_prepulse=-950.0e-15,
        tau=30e-15,
    )

    beam = GaussianBeam(w_0=1.0e-6,
                        focus_position=(4e-6, 5e-6, 3e-6),
                        wavelength=800e-9, temporal_shape=temporal_shape)
    x= np.linspace(-10e-6,10e-6, 1024)
    z= np.array([3.0e-6])
    y= np.linspace(-20e-6,20e-6, 1024)
    X,Y = np.meshgrid(x,y,  sparse=False)
    for ii, norm in enumerate([Normalize(vmin=-1, vmax=1), SymLogNorm(linthresh=1e-10, vmin=-1, vmax=1)]):
        f = plt.figure()
        ax = f.add_subplot(2, 1, 1)
        ax2 = f.add_subplot(2, 1, 2)
        im = ax.pcolormesh(Y, X, np.zeros_like(X), norm=norm, cmap='bwr')
        f.colorbar(mappable=im, ax=ax)
        text =ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.8, 'pad':5},
                transform=ax.transAxes, ha="center")
        n_frames = 10
        ax2.set_yscale('log')
        ax2.plot(t, temporal_shape(t))
        line = ax2.axvline(t[0], color='r', lw=1)
        plt.tight_layout()
        with tqdm(total=n_frames) as pbar:
            def animate(i):
                tt = t[0] + (t[-1] - t[0])/n_frames * i
                (envelope, wave) = beam(x[:, None, None], y[None, : ,None], z[None, None, :], tt)
                amplitude = envelope * wave
                if ii == 0:
                    amplitude, one = _normalize(amplitude)
                    text.set_text(f"1={one:.2e}, t={tt*1e15:.2f}fs")
                else:
                    text.set_text(f"t={tt*1e15:.2f}fs")
                im.set_array(amplitude.T.ravel())
                line.set_xdata([tt,tt])
                pbar.update(1)
                return im, text, line
            an = FuncAnimation(f, animate, frames=n_frames, interval=1, blit=True,)
            an.save(f'sim_{ii}.gif', dpi=80, writer='imagemagick', fps=10)
    plt.close()
