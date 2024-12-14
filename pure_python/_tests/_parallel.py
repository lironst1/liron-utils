import time
from liron_utils.pure_python import parallel


def foo(i, x, y):
	time.sleep(.1)
	return i


if __name__ == '__main__':
	for par_func in [parallel.parallel_map, parallel.parallel_threading]:
		t0 = time.time()
		out = par_func(func=foo, iterable=range(1000), x=1, y=2)
		print(out[:5])
		print(f"{par_func.__name__} time: {time.time() - t0} sec")
		time.sleep(0.1)

		"""
		Results:
		--------
		len(iterable)	                    | 100   | 1000  | 10000 |
		parallel_map                        | 4.0   | 10.2  | 72.7  | [sec]
		parallel_threading (10 threads)     | 1.1   | 11.1  | 111.2 | [sec]
		"""
	pass
