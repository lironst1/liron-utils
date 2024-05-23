import os
import platform
import subprocess
import shutil

# todo: copy/paste files and folders

move_file = os.rename
remove_file = os.remove


def mkdirs(dirs, *args, **kwargs):
	"""
	Create (possibly multiple) directories

	Parameters
	----------
	dirs :
	args :
	kwargs :

	Returns
	-------

	"""

	kwargs = {"exist_ok": True} | kwargs

	os.makedirs(dirs, *args, **kwargs)


def rmdir(dir):
	try:
		os.rmdir(dir)
	except FileNotFoundError:
		pass


def copy_file(src, dst, *args, **kwargs):
	"""
	Copy file or an entire directory

	Parameters
	----------
	src : 
	dst : 
	args : 
	kwargs : 

	Returns
	-------

	"""

	if os.path.isdir(src):
		shutil.copytree(src, dst, *args, **kwargs)
	else:
		shutil.copy2(src, dst, *args, **kwargs)


def open_file(file):
	if platform.system() == "Windows":
		os.startfile(file)
	elif platform.system() == "Darwin":
		subprocess.Popen(["open", file])
	else:
		subprocess.Popen(["xdg-open", file])
