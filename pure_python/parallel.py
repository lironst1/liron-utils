import warnings
import sys
import functools
import multiprocessing as mp
from tqdm import tqdm

NUM_CPUS = mp.cpu_count()
NUM_PROCESSES_TO_USE = NUM_CPUS // 2


def parallel_map(func, iterable, num_processes=NUM_PROCESSES_TO_USE, **kwargs):
	"""
	Run function 'func' in parallel.
	See qutip.parallel.parallel_map for reference.

	Notes
	-----
	- 'func' must be a global function.
	- parallel_map uses 'spawn' [1,2] by default in Windows, which starts a Python child process from scratch.
	  This means that everything not under an 'if __name__==__main__' block will be executed multiple times.
	- In UNIX we use 'fork'.

	References
	----------
	[1] https://stackoverflow.com/questions/64095876/multiprocessing-fork-vs-spawn/66113051#66113051
	[2] https://stackoverflow.com/questions/72935231/statements-before-multiprocessing-main-executed-multiple-times-python
	[3] https://superfastpython.com/multiprocessing-pool-issue-tasks

	Examples
	--------
	>>> def func(iter, x, y):
	>>> 	time.sleep(1)
	>>> 	return (x + y) ** iter
	>>>
	>>> if __name__ == '__main__':
	>>> 	x = 1
	>>> 	y = 2
	>>> 	t0 = time.time()
	>>> 	out = parallel_map(func=func, iterable=range(100), num_processes=8, x=x, y=y)
	>>> 	print(out[:5])
	>>> 	print(f"time: {time.time() - t0}sec")

	Parameters
	----------
	func :          The function to evaluate in parallel. The first argument is the changing value of each iteration
	iterable :      First input argument for 'func'
	num_processes : number of processes to use
	args :          passed to func
	kwargs :        passed to func

	Returns
	-------
	list of 'func' outputs, organized by the order of 'iter'.
	"""
	if sys.platform == 'darwin':  # in UNIX 'fork' can be used (faster but more dangerous)
		Pool = mp.get_context('fork').Pool
	else:  # In Windows only 'spwan' is available
		Pool = mp.Pool

	if num_processes > NUM_CPUS:
		warnings.warn(f"Requested number of processes {num_processes} is larger than number of CPUs {NUM_CPUS}.\n"
		              f"For better performance, consider reducing 'num_processes'.", category=UserWarning)
	num_processes = min(num_processes, NUM_CPUS, len(iterable))

	with Pool(processes=num_processes) as pool:
		func_partial = functools.partial(func, **kwargs)

		out_async = [pool.apply_async(func=func_partial, args=(i,)) for i in iterable]

		out = []
		for out_async_i in tqdm(out_async, total=len(iterable)):
			try:
				out += [out_async_i.get()]

			except KeyboardInterrupt as e:
				raise e

			except Exception as e:
				warnings.warn(str(e))

	return out
