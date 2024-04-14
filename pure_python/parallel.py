import sys
import multiprocessing
from tqdm import tqdm

NUM_CPUS = multiprocessing.cpu_count()


def parallel_map(func, iter, num_cpus=NUM_CPUS, *args, **kwargs):
	"""
	Run function func in parallel.
	See qutip.parallel.parallel_map for reference.

	Examples
	--------
	>>> def func(iter, x, y):
	>>>     return (x + y) ** iter
	>>> x = 1
	>>> y = 2
	>>> parallel_map(func, range(10), x=x, y=y)

	Parameters
	----------
	func :          The function to evaluate in parallel.
					The First argument should be the changing value of each iteration.
	iter :          First input argument for 'func'
	num_cpus :      number of CPUs to use for parallel computation.
	args :
	kwargs :

	Returns
	-------
	list of 'func' outputs, organized in the order of 'iter'.
	"""
	if sys.platform == 'darwin':
		Pool = multiprocessing.get_context('fork').Pool
	else:
		Pool = multiprocessing.Pool

	if num_cpus > NUM_CPUS:
		print(f"Requested number of CPUs {num_cpus} is larger than physical number {NUM_CPUS}.\n"
		      f"Reduce 'num_cpus' for better performance.")

	pool = Pool(processes=num_cpus)

	t = tqdm(range(len(iter)))

	def progress_bar(n):
		progress_bar.n += 1
		t.update(progress_bar.n)

	progress_bar.n = 0

	try:
		out_async = [pool.apply_async(func=func,
		                              args=(i,) + args,
		                              kwds=kwargs,
		                              callback=progress_bar)
		             for i in iter]

		while not all([ar.ready() for ar in out_async]):
			for out in out_async:
				out.wait(timeout=0.1)

	except KeyboardInterrupt as e:
		raise e

	finally:
		pool.terminate()
		pool.join()

	return [out.get() for out in out_async]

# =============================================================
# from joblib import Parallel, delayed
# p = Parallel(n_jobs=NUM_CPUS)
# tmp = p(delayed(func)(idx) for idx in tqdm(iterable))
