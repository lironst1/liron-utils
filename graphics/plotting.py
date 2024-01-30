import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.collections
import matplotlib.colors
import matplotlib.style
import matplotlib.animation
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from .axes import AxesLironUpper
from ..uncertainties_math import to_numpy
from .utils.default_kwargs import update_kwargs, COLORS
from ..signal_processing.base import interp1


# TODO add standalone functions out of class gr.plot


class AxesLiron(AxesLironUpper):
	def plot(self,
	         x, y,
	         **plot_kw):

		@self._vectorize(cls=self, x=x, y=y)
		def _plot(ax: Axes,
		          x, y,
		          **plot_kw):
			"""
			2D plot of y=f(x)

			Args:
				ax:
				x:
				y:
				**plot_kw:

			Returns:

			"""

			plot_kw = update_kwargs(plot_kw=plot_kw)["plot_kw"]

			ax.plot(x, y, **plot_kw)

		return _plot(**plot_kw)

	def plot_errorbar(self,
	                  x, y, xerr=None, yerr=None,
	                  **errorbar_kw):

		@self._vectorize(cls=self, x=x, y=y, xerr=xerr, yerr=yerr)
		def _plot_errorbar(ax: Axes,
		                   x, y, xerr=None, yerr=None,
		                   **errorbar_kw):
			"""
			2D plot of y=f(x) with errorbars

			Parameters
			----------
			ax :
			x :
			y :
			xerr, yerr :        array_like, optional
				Deviations in 'x','y'. 'x','y' may also be sent as 'uncertainties' arrays, then 'xerr','yerr' are disregarded
			errorbar_kw :

			Returns
			-------

			"""

			errorbar_kw = update_kwargs(errorbar_kw=errorbar_kw)["errorbar_kw"]

			x, xerr = to_numpy(x, xerr)
			y, yerr = to_numpy(y, yerr)

			ax.errorbar(x, y, xerr=xerr, yerr=yerr, **errorbar_kw)

		return _plot_errorbar(**errorbar_kw)

	def plot_data_and_curve_fit(self,
	                            x, y, fit_fcn, xerr=None, yerr=None,
	                            p_opt=None, p_cov=None, n_std=2, interp_factor=20,
	                            curve_fit_plot_kw=None, **errorbar_kw):

		@self._vectorize(cls=self,
		                 x=x, y=y, fit_fcn=fit_fcn, xerr=xerr, yerr=yerr,
		                 p_opt=p_opt, p_cov=p_cov, n_std=n_std, interp_factor=interp_factor,
		                 curve_fit_plot_kw=curve_fit_plot_kw)
		def _plot_data_and_curve_fit(ax: Axes,
		                             x, y, fit_fcn, xerr=None, yerr=None,
		                             p_opt=None, p_cov=None, n_std=2, interp_factor=20,
		                             curve_fit_plot_kw=None, **errorbar_kw):
			"""
			2D scatter plot y=f(x) + curve fit

			Parameters
			----------
			ax :
			x, y :
			fit_fcn :
			xerr, yerr :        array_like, optional
				Deviations in 'x','y'. 'x','y' may also be sent as 'uncertainties' arrays, then 'xerr','yerr' are disregarded
			p_opt : Output parameter of scipy.optimize.curve_fit
			p_cov : Output parameter of scipy.optimize.curve_fit
			n_std : Number of standard deviations of confidence to be plotted with the fitted curve
			interp_factor :
			curve_fit_plot_kw :
			errorbar_kw :

			Returns
			-------

			Examples
			--------
				>>> import numpy as np
				>>> from liron_utils import graphics as gr
				>>> import scipy.optimize

				>>> N = 101
				>>> x = np.linspace(0, 10, N)
				>>> yerr = 5 * np.random.randn(N)
				>>> y = 2 * x ** 2 + 4 * x + 5 + yerr

				>>> def fit_fcn(x, a, b, c):
				>>>     return a * x ** 2 + b * x + c

				>>> (p_opt, p_cov) = scipy.optimize.curve_fit(fit_fcn, x, y)

				>>> ax = gr.AxesLiron()
				>>> ax.plot_data_and_curve_fit(x, y, fit_fcn, yerr=yerr, p_opt=p_opt, p_cov=p_cov)
				>>> ax.show_fig()

			"""

			errorbar_kw = {
				              "label":  'Data',
				              "zorder": -1,
			              } | errorbar_kw

			if curve_fit_plot_kw is None:
				curve_fit_plot_kw = dict()
			curve_fit_plot_kw = {
				                    "label": 'Curve fit'
			                    } | curve_fit_plot_kw

			# Data
			x, xerr = to_numpy(x, xerr)
			y, yerr = to_numpy(y, yerr)

			self.plot_errorbar(x, y, xerr=xerr, yerr=yerr, **errorbar_kw)

			# Curve fit
			x_interp = interp1(x, interp_factor * len(x))
			fit_mid = fit_fcn(x_interp, *p_opt)

			ax.plot(x_interp, fit_mid, **curve_fit_plot_kw)

			# Confidence fill
			if p_opt is not None and p_cov is not None:
				p_err = np.sqrt(np.diag(p_cov))
				fit_low = fit_fcn(x_interp, *(p_opt - n_std * p_err))
				fit_high = fit_fcn(x_interp, *(p_opt + n_std * p_err))

				ax.fill_between(x_interp, fit_low, fit_high, linestyle='-', color=COLORS.LIGHT_GREY, alpha=0.4,
				                label=f'{n_std} std confidence')

		return _plot_data_and_curve_fit(**errorbar_kw)

	def plot_data_and_lin_reg(self,
	                          x, y, reg=None, xerr=None, yerr=None,
	                          reg_plot_kw=None, **plot_errorbar_kw):

		@self._vectorize(cls=self,
		                 x=x, y=y, reg=reg, xerr=xerr, yerr=yerr,
		                 reg_plot_kw=reg_plot_kw)
		def _plot_data_and_lin_reg(ax: Axes,
		                           x, y, reg=None, xerr=None, yerr=None,
		                           reg_plot_kw=None, **plot_errorbar_kw):
			"""
			2D scatter plot y=f(x) + linear regression.

			Examples:
				>>> import numpy as np
				>>> from scipy.stats import linregress
				>>> from liron_utils import graphics as gr

				>>> N = 100
				>>> x = np.arange(N)
				>>> y = 2*x + np.random.randn(N)
				>>> reg = linregress(x, y)

				>>> ax = gr.AxesLiron()
				>>> ax.plot_data_and_lin_reg(x, y, reg)
				>>> ax.show_fig()

			Args:
				ax:
				x:
				y:                      f(x)
				reg:                    Output of scipy.stats.linregress
				xerr:                   Error in x
				yerr:                   Error in y
				reg_plot_kw:
				**plot_errorbar_kw:

			Returns:

			"""

			plot_errorbar_kw = {
				                   "label": "Data"
			                   } | plot_errorbar_kw

			if reg_plot_kw is None:
				reg_plot_kw = dict()
			reg_plot_kw = {
				              "label": fr"{plot_errorbar_kw['label']} linreg: slope={reg.slope:.3f}$\pm${reg.stderr:.3f}, $R^2$={reg.rvalue ** 2:.3f}"
			              } | reg_plot_kw

			x, xerr = to_numpy(x, xerr)
			y, yerr = to_numpy(y, yerr)

			self.plot_errorbar(x, y, xerr=xerr, yerr=yerr,
			                   **plot_errorbar_kw)  # TODO: need to change to _plot_errorbar

			if reg is not None:
				tmp = [reg.slope * i + reg.intercept for i in x]
				self.plot(x, tmp, **reg_plot_kw)

		return _plot_data_and_lin_reg(**plot_errorbar_kw)

	def plot_line_collection(self,
	                         x: np.ndarray, y: np.ndarray, arr: np.ndarray,
	                         **LineCollection_kwargs):

		@self._vectorize(cls=self, x=x, y=y, arr=arr)
		def _plot_line_collection(ax: Axes,
		                          x: np.ndarray, y: np.ndarray, arr: np.ndarray,
		                          **LineCollection_kwargs):
			"""

			Args:
				ax:
				x:
				y:
				arr:
				**LineCollection_kwargs:

			Returns:

			"""
			points = np.array([x, y]).T.reshape(-1, 1, 2)
			segments = np.concatenate([points[:-1], points[1:]], axis=1)

			norm = plt.Normalize(vmin=np.min(arr), vmax=np.max(arr))

			# cmap = matplotlib.colors.ListedColormap(['r', 'g', 'b'])
			lc = matplotlib.collections.LineCollection(segments, norm=norm, **LineCollection_kwargs)
			lc.set_array(arr)

			line = ax.add_collection(lc)

			ax.figure.colorbar(line, ax=ax)

			return line

		return _plot_line_collection(**LineCollection_kwargs)

	def plot_specgram(self,
	                  y: np.ndarray, fs: int,
	                  **specgram_kw):

		@self._vectorize(cls=self, y=y, fs=fs)
		def _plot_specgram(ax: Axes,
		                   y: np.ndarray, fs: int,
		                   **specgram_kw):
			"""
			Plot spectrogram

			Args:
				ax:
				y:                      Data, given as 1D array with respect to time
				fs:                     Sample rate
				**specgram_kw:

			Returns:

			"""

			specgram_kw = update_kwargs(specgram_kw=specgram_kw)["specgram_kw"]

			specgram_out = ax.specgram(y, Fs=fs, **specgram_kw)

			if ax.get_title() == '':
				ax.set_title('Spectrogram')
			if ax.get_label() == '':
				ax.set_xlabel('Time [sec]')
				ax.set_ylabel('Frequency [Hz]')

			ax.figure.colorbar(matplotlib.cm.ScalarMappable(), ax=ax)

			return specgram_out

		return _plot_specgram(**specgram_kw)

	def plot_surf(self,
	              x, y, z,
	              **plot_surface_kw):

		@self._vectorize(cls=self, x=x, y=y, z=z)
		def _plot_surf(ax: Axes,
		               x, y, z,
		               **plot_surface_kw):
			"""
			3D surf plot of z=f(x,y)

			Args:
				ax:
				x:          1D or 2D. If given as 1D, will automatically apply meshgrid and treat z as a lambda function
				y:          1D or 2D. If given as 1D, will automatically apply meshgrid and treat z as a lambda function
				z:          2D or lambda function

			Returns:

			"""

			assert hasattr(ax, "plot_surface"), "Axes does not have a plot_surface attribute. " \
			                                    "make sure that you created an axes with projection='3d'"

			plot_surface_kw = update_kwargs(plot_surface_kw=plot_surface_kw)["plot_surface_kw"]

			X, Y, Z = x, y, z
			if x.ndim == 1:
				X, Y = np.meshgrid(x, y)
			if callable(z):
				Z = z(X, Y)
			if np.all(Z.shape == np.flip(X.shape)):
				Z = Z.T

			ax.plot_surface(X, Y, Z, **plot_surface_kw)

		# ax.figure.colorbar(matplotlib.cm.ScalarMappable(), ax=ax)

		return _plot_surf(**plot_surface_kw)

	def plot_contour(self,
	                 x, y, z, contours,
	                 *args, **kwargs):

		@self._vectorize(cls=self, x=x, y=y, z=z, contours=contours)
		def _plot_contour(ax: Axes,
		                  x, y, z, contours,
		                  *args, **kwargs):
			"""
			Contour plot of scalar field z=f(x,y)

			Args:
				ax:
				x:
				y:
				z:
				contours:
				*args:
				**kwargs:

			Returns:

			"""

			cs = ax.contour(x, y, z, contours, *args, **kwargs)
			ax.clabel(cs, inline=True, fontsize=10)
			return cs

		return _plot_contour(*args, **kwargs)

	def plot_animation(self,
	                   images, titles=None,
	                   im_instance: AxesImage = None,
	                   *args, **kwargs):

		@self._vectorize(cls=self, images=images, titles=titles, im_instance=im_instance)
		def _plot_animation(ax: Axes,
		                    images, titles=None,
		                    im_instance: AxesImage = None,
		                    *args, **kwargs):
			"""
			Plot animation

			Examples:
				>>> import numpy as np
				>>> from liron_utils import graphics as gr

				>>> nimages = 10
				>>> images = np.random.random((nimages))
				>>> Ax = gr.AxesLiron()
				>>> Ax.plot_animation(images)
				>>> Ax.save_fig("test.gif")

			Parameters
			----------
			ax :
			images :            array_like
								The images to be plotted. Can be given as:
								- a 3D array of size [#images, x, y]
								- a list of 2D arrays of size [x, y]
								- a list of image handles. In this case, im_instance is ignored
			titles :            list or function handle, optional
									- List of changing titles
									- function handle whose input argument is iterable and outputs the title
			im_instance   :     AxesImage
								an image handle to use in case user wants some pre-defined properties
			args :              sent to matplotlib.animation.FuncAnimation
			kwargs :

			Returns
			-------

			"""

			if titles is not None:
				kwargs |= {"blit": False}

				if callable(titles):
					titles = [titles(i) for i in range(len(images))]

			if type(images[0]) is AxesImage:
				def update_image(i):
					im = images[i]
					if titles is not None:
						ax.set_title(titles[i])
					return im
			else:
				images = np.asarray(images)

				im = im_instance
				if im_instance is None:
					im = ax.imshow(images[0])

				def update_image(i):
					im.set_data(images[i])
					if titles is not None:
						ax.set_title(titles[i])
					return im

			self.func_animation = matplotlib.animation.FuncAnimation(fig=ax.figure,
			                                                         func=update_image, frames=images.shape[0],
			                                                         *args, **kwargs)

		return _plot_animation(*args, **kwargs)
