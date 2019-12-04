"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Loading data
"""

import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


SMALL_SIZE = 8.5
MEDIUM_SIZE = 14
BIGGER_SIZE = 20
plt.style.use('ggplot')
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE, titleweight='bold')     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE, labelweight='bold')    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rc('text', usetex=True)

class Plotter:

	@staticmethod
	def plot_pc(xs, filepath, savefig=True):
		# pre-process
		# mask = xs < 1
		# x = np.sum(mask, axis=1)
		# mask = (x == xs.shape[1])
		# xs = xs[mask]

		fig = plt.figure()
		if 3 == xs.shape[1]:
			ax = fig.add_subplot(111, projection='3d')
			ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2])
			ax.set_xlim([0, 1])
			ax.set_ylim([0, 1])
		else:
			plt.scatter(xs[:, 0], xs[:, 1])
		if savefig:
			plt.savefig(filepath, bbox__hnches='tight')
			plt.close(fig)
		else:
			plt.show()

	@staticmethod
	def plot_pc_color(xs, filepath, savefig=True):
		last_point = xs[-1, :]
		xs = xs[:-1, :]
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.set_xlim([0, 1])
		ax.set_ylim([0, 1])
		ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], color='g')
		ax.scatter(last_point[0], last_point[1], last_point[2], color='r')
		plt.savefig(filepath, bbox__hnches='tight')
		plt.close(fig)

	@staticmethod
	def save_images(samples, im_size, path, idx, n_fig_unit=2):
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(n_fig_unit, n_fig_unit)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(im_size, im_size), cmap='Greys_r')

		plt.savefig(path + '/{}.png'.format(str(idx).zfill(3)),
					bbox__hnches='tight')
		plt.close(fig)
		return fig

	@staticmethod
	def plot_sample(samples, real_data):
		x = np.mean(samples, axis=0)
		y = np.mean(real_data, axis=0)
		plt.scatter(range(100), x[:100])
		plt.scatter(range(100), y[:100])
		plt.show()

		sample_mu = np.mean(samples, axis=0)
		sample_std = np.std(samples, axis=0)
		data_mu = np.mean(real_data, axis=0)
		data_std = np.std(real_data, axis=0)
		print("Mu: ", np.linalg.norm(sample_mu - data_mu))
		print("std: ", np.linalg.norm(sample_std - data_std))


	@staticmethod
	def plot_hist_1(data, deg_vec, path, idx):
		x = data / deg_vec
		fig = plt.figure(figsize=(4, 4))
		plt.scatter(np.arange(len(x)), x)
		plt.savefig(path + '/{}.png'.format(str(idx).zfill(3)),
					bbox__hnches='tight')
		plt.close(fig)

	@staticmethod
	def plot_hist_2(data, deg_vec):
		fig = plt.gcf()
		fig.show()
		fig.canvas.draw()
		plt.title("Gaussian Histogram")
		plt.xlabel("Value")
		plt.ylabel("Frequency")
		for node in range(len(deg_vec)):
			try:
				x = data[:, node] / deg_vec[node]
				mu = np.mean(x)
				sig = np.std(x)
				print('Node {:d}: mean:{:.3f}, std: {:.3f}'.format(node, mu, sig))
				plt.hist(x, 20) # Hard-code
				fig.canvas.draw()
				input('Press to continue ...')
			except:
				# from IPython import embed; embed() #os._exit(1)
				print('Exception')
				break

	@staticmethod
	def plot_fig(lmse):
		x = np.arange(len(lmse))
		plt.figure()
		plt.plot(x, results[0], c='r')
		plt.plot(x, results[1], c='b')
		plt.plot(x, results[-1], c='g')

	@staticmethod
	def plot_tnse(fname):
		data = np.load(fname)
		tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
		result = tsne.fit_transform(data)
		vis_x = result[:, 0]
		vis_y = result[:, 1]
		plt.scatter(vis_x, vis_y, cmap=plt.cm.get_cmap("jet", 10))
		plt.title('Laplacian')
		plt.show()

	@staticmethod
	def plot_dist(dists, n_pixels, name):
		# Plot the distributions
		# from IPython import embed; embed()
		means = dists[:, 0]
		sds = dists[:, 1]
		hpd_025 = dists[:, 2]
		hpd_975 = dists[:, 3]

		fig = plt.figure(figsize=(20, 7))
		plt.errorbar(range(n_pixels), means, yerr=sds, fmt='-o', linestyle="None")
		plt.scatter(range(n_pixels), hpd_025, c='b')
		plt.scatter(range(n_pixels), hpd_975, c='b')
		fig.tight_layout()
		fig.savefig(name, bbox_inches='tight')
		plt.xlabel('Vertex')
		plt.title('Difference of means distribution')
		plt.show()

	@staticmethod
	def plot_signals(off_data, gan_name, cn):
		root_path = '../result'
		fig = plt.figure(figsize=(10, 7))
		name_data = offdata2name(off_data)
		if cn:
			cn_name = 'cn'
		else:
			cn_name = 'ad'
		plot_path = "{}/{}/plt_{}_{}.png".format(root_path, gan_name, name_data, cn_name)
		data_path = '../data/{}/data_{}_4k.mat'.format(name_data, name_data)
		_, real_signals = load_data(data_path, is_control=cn)
		for off_model in range(1, 4):
			name_model = offmodel2name(off_model)
			if off_model == 3:
				signals = real_signals
				if off_data == 3: # Simuln
					signals = signals[:1000, :]
			else:
				sample_path = "{}/{}/{}/{}".format(root_path, gan_name, name_data, name_model)
				signal_path = os.path.join(sample_path, "{}/samples/samples_1000.npy".format(cn_name))
				signals = np.load(signal_path)[:real_signals.shape[0], :]
			means = np.mean(signals, axis = 0)
			plt.scatter(range(len(means)), means, label=name_model)
		plt.legend()
		plt.xlabel('Node')
		plt.ylabel("Value")
		plt.title('{} signals on {} - {}'.format(cn_name, name_data, gan_name))
		plt.show()
		fig.tight_layout()
		fig.savefig(plot_path, bbox_inches='tight')

	@staticmethod
	def plot_eb(off_data, off_gan, alp, epsilon):
		fig = plt.figure(figsize=(10, 7))
		name_data = offdata2name(off_data)
		name_gan = offgan2name(off_gan)
		root_folder = "../result/eb-same/{}".format(name_gan)
		plot_path = "{}/plt_{}_{}_{}_{}.png".format(root_folder, name_data, name_gan, alp, epsilon)
		for off_model in range(1, 4):
			name_model = offmodel2name(off_model)
			fname = "eval_{}_{}_{}".format(name_data, name_gan, name_model)
			if off_model ==1:
				file_path = "{}/{}_{}_{}.npy".format(root_folder, fname, alp, epsilon)
			elif off_model == 2:
				file_path = "{}/{}_{}.npy".format(root_folder, fname, epsilon)
			else:
				sample_folder = '../result/eb-same/real'
				file_path = "{}/eval_{}_{}.npy".format(sample_folder, name_data,  epsilon)
			arr = np.load(file_path)
			# plt.plot(range(len(arr)), sorted(arr), label=name_model)
			plt.scatter(range(len(arr)), arr, label=name_model)
		plt.legend()
		plt.xlabel('Node')
		plt.ylabel("e-value")
		plt.title('{} data - e-value on {} - {} with epsilon = {}'.format(name_data, name_gan, alp, epsilon))
		plt.show()

	@staticmethod
	def plot_ttest_alp(off_data, off_gan, alp):
		name_data = offdata2name(off_data)
		name_gan = offgan2name(off_gan)
		root_path = '../result'
		output_path = "{}/ttest/{}".format(root_path, name_gan)
		plot_path = "{}/plt_{}_{}_{}.png".format(output_path, name_data, name_gan, alp)

		fig = plt.figure(figsize=(10, 7))
		if off_data == 4:
			model_list = [2, 3]
		else:
			model_list = [1, 2, 3]

		colors = ['y', 'b', 'r', 'g']
		for off_model in model_list:
			name_model = offmodel2name(off_model)
			if off_model == 1:
				file_path = "{}/{}_{}_{}_{}.npy".format(output_path, name_data, name_gan, name_model, alp)
			else:
				file_path = "{}/{}_{}_{}_0.01.npy".format(output_path, name_data, name_gan, name_model)
			arr = np.load(file_path)
			spio.savemat('/home/tuandinh/Documents/Project/glapGan/data/mesh_all/pvalue_{}_{}.mat'.format(name_model, name_data), {'pvalue_{}'.format(name_model):arr})
			return
			# from IPython import embed; embed()
			# idx = arr < 0.08
			# arr = arr[idx]
			plt.plot(range(len(arr)), sorted(arr), label=name_model,  c=colors[off_model])
			# plt.scatter(range(len(arr)), arr, label=name_model)

		plt.legend()
		plt.xlabel('Node')
		plt.ylabel("p-value")
		plt.title('{} data - BH correction with {} - alpha {}'.format(name_data, name_gan, alp))
		plt.show()
		fig.tight_layout()
		fig.savefig(plot_path, bbox_inches='tight')

	@staticmethod
	def plot_ttest_all(off_data, off_gan):
		name_data = offdata2name(off_data)
		name_gan = offgan2name(off_gan)
		root_path = '../result'
		output_path = "{}/ttest/{}".format(root_path, name_gan)
		plot_path = "{}/plt_{}_{}_all.png".format(output_path, name_data, name_gan)

		fig = plt.figure(figsize=(7, 4))
		if off_data == 4:
			model_list = [2, 3]
		else:
			model_list = [1, 2, 3]

		colors = ['y', 'b', 'r', 'g']
		for off_model in [2, 3]:
			name_model = offmodel2name(off_model)
			file_path = "{}/{}_{}_{}_0.01.npy".format(output_path, name_data, name_gan, name_model)
			arr = np.load(file_path)
			# idx = arr < 0.08
			# arr = arr[idx]
			plt.plot(range(len(arr)), sorted(arr), label=name_model, c=colors[off_model])

		linestyles = ['-', '--', '-.', ':']
		i = 0
		for alp in [0.03, 0.07, 0.11, 0.15]:
			name_model = offmodel2name(1)  # lapgan
			file_path = "{}/{}_{}_{}_{}.npy".format(output_path, name_data, name_gan, name_model, alp)
			arr = np.load(file_path)
			plt.plot(range(len(arr)), sorted(arr), label=name_model + ' ' r'$\alpha=$'+str(alp), linestyle=linestyles[i], c='b')
			i = i + 1

		plt.legend()
		plt.xlabel('Node')
		plt.ylabel("p-value")
		plt.title('{} data - BH correction with {}'.format(name_data, name_gan))
		plt.show()
		fig.tight_layout()
		fig.savefig(plot_path, bbox_inches='tight')

	@staticmethod
	def plot_ttest_zoom(off_data, off_gan):
		name_data = offdata2name(off_data)
		name_gan = offgan2name(off_gan)
		root_path = '../result'
		output_path = "{}/ttest/{}".format(root_path, name_gan)
		plot_path = "{}/plt_{}_{}_zoom.png".format(output_path, name_data, name_gan)

		# fig = plt.figure(figsize=(7, 4))
		fig, ax = plt.subplots() #
		x = range(4225)
		linestyles = ['-', '--', '-.', ':']
		colors = ['y', 'b', 'r', 'g']
		for off_model in [2, 3]:
			name_model = offmodel2name(off_model)
			file_path = "{}/{}_{}_{}_0.01.npy".format(output_path, name_data, name_gan, name_model)
			arr = np.load(file_path)
			ax.plot(x, sorted(arr), label=name_model, c=colors[off_model])

		i = 0
		for alp in [0.03, 0.07, 0.11, 0.15]:
			name_model = offmodel2name(1)  # lapgan
			file_path = "{}/{}_{}_{}_{}.npy".format(output_path, name_data, name_gan, name_model, alp)
			arr = np.load(file_path)
			ax.plot(x, sorted(arr), label=r'$\lambda_L=$'+str(alp), linestyle=linestyles[i], c='b')
			i = i + 1

		plt.legend(loc='lower right')
		plt.xlabel('Node (arbitrary order)')
		plt.ylabel("p-value")
		plt.title('Sorted p values after Benjamini-Hochberg correction')

		axins = zoomed_inset_axes(ax, 3, loc='upper left', bbox_to_anchor=(0.16, 0.9),bbox_transform=ax.figure.transFigure) # zoom-factor: 2.5,
		# axins = inset_axes(ax, 1,1 , loc=2,bbox_to_anchor=(0.2, 0.55))
		for off_model in [2, 3]:
			name_model = offmodel2name(off_model)
			file_path = "{}/{}_{}_{}_0.01.npy".format(output_path, name_data, name_gan, name_model)
			arr = np.load(file_path)
			axins.plot(x, sorted(arr), label=name_model, c=colors[off_model])

		i = 0
		for alp in [0.03, 0.07, 0.11, 0.15]:
			name_model = offmodel2name(1)  # lapgan
			file_path = "{}/{}_{}_{}_{}.npy".format(output_path, name_data, name_gan, name_model, alp)
			arr = np.load(file_path)
			axins.plot(x, sorted(arr), label=r'$\alpha=$'+str(alp), linestyle=linestyles[i], c='b')
			i = i + 1

		x1, x2, y1, y2 = 1100, 2100, 0, 0.08 # specify the limits
		axins.set_xlim(x1, x2) # apply the x-limits
		axins.set_ylim(y1, y2) # apply the y-limits
		axins.set_facecolor((1, 0.75, 0.75))
		mark_inset(ax, axins, loc1=1, loc2=3, linewidth=1, ec="0.5")
		# plt.yticks(visible=False)
		plt.xticks(visible=False)
		plt.grid(False)
		fig.tight_layout()
		fig.savefig(plot_path)
		plt.show()

	@staticmethod
	def get_fdr(r_pvalues, l_pvalues):
		n = 20
		t = np.linspace(0.01, 0.1, num=n)
		lines = np.zeros((n, 1))
		for k in range(n):
			threshold = t[k]
			l_pred = np.asarray(l_pvalues < threshold, dtype=int)
			r_pred = np.asarray(r_pvalues < threshold, dtype=int)
			l_v = 0
			for i in range(len(r_pred)):
				if r_pred[i] == 1:
					l_v += l_pred[i]
			# l_v = np.sum(abs(l_pred - r_pred))
			# b_v = np.sum(abs(b_pred - r_pred))
			lines[k] = l_v / np.sum(r_pred)
			# from IPython import embed; embed()
		return lines

	@staticmethod
	def plot_fdr(off_data, off_gan):
		name_data = offdata2name(off_data)
		name_gan = offgan2name(off_gan)
		root_path = '../result'
		output_path = "{}/ttest/{}".format(root_path, name_gan)
		plot_path = "{}/recall_{}_{}_all.png".format(output_path, name_data, name_gan)

		fig = plt.figure(figsize=(10, 7))
		b_pvalues = np.load("{}/{}_{}_{}_0.01.npy".format(output_path, name_data, name_gan, offmodel2name(2)))
		r_pvalues = np.load("{}/{}_{}_{}_0.01.npy".format(output_path, name_data, name_gan, offmodel2name(3)))

		b_line = get_fdr(r_pvalues, b_pvalues)
		plt.plot(t, b_line, label='WGAN (baseline)', c='r')
		linestyles = ['-', '--', '-.', ':']
		colors = ['y', 'b', 'r', 'g']
		i = 0
		for alp in [0.05, 0.1, 0.15]:
			name_model = offmodel2name(1)  # lapgan
			file_path = "{}/{}_{}_{}_{}.npy".format(output_path, name_data, name_gan, name_model, alp)
			l_pvalues = np.load(file_path)
			l_line = get_fdr(r_pvalues, l_pvalues)
			plt.plot(t, l_line, label=r'$\lambda_L=$'+str(alp), linestyle=linestyles[i], c='b')
			i = i + 1
			# from IPython import embed; embed()

		plt.legend()
		plt.xlabel('p-value threshold')
		plt.ylabel("Sensitivity")
		plt.grid(b=True)
		plt.title('Sensitivity of t-test with generated data');
		plt.show()
		fig.tight_layout()
		fig.savefig(plot_path, bbox_inches='tight')

if __name__ ==  '__main__':
	data = np.load('../../data/plane/res_pg.npy')
	for i in range(100):
		xs = data[0, :, :]
		Plotter.plot_pc(xs, '../../data/plane/img/{}.png'.format(i), savefig=True)
