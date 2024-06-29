import os
import platform
import subprocess
import shutil

import numpy as np

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


def copy(src, dst, *args, **kwargs):
	"""
	Copy file or an entire directory

	Parameters
	----------
	src :       str | list
				The source file(s) to be copied
	dst :       str | list
				- If 'src' is a string, 'dst' should be either the destination directory (and 'src' will preserve the
				same name), or as a file name.
				- If 'src' is an array, 'dst' is the destination directory and all files in 'src' will preserve the
				same name into 'dst'.
	args :      sent to shutil.copy2
	kwargs : 

	Returns
	-------

	"""

	isdir = lambda path: len(os.path.splitext(path)[-1]) == 0

	if type(src) is str: src = [src]
	if type(dst) is str: dst = [dst]

	src_file_name = [os.path.split(src[i])[-1] for i in range(len(src))]

	if len(dst) == 1 and len(src) > 1:  # copy multiple files/dirs into the same dir
		assert len(os.path.splitext(dst[0])[-1]) == 0, "'dst' should be a directory."
		dst *= len(src)

	assert len(src) == len(dst), "len(dst) must be either 1 or len(src)."

	# copy multiple files/dirs into multiple files/dirs
	for i in range(len(src)):
		if os.path.isdir(src[i]):  # copy dir
			dst[i] = os.path.join(dst[i], src_file_name[i])
			shutil.copytree(src[i], dst[i], *args, **kwargs)

		else:  # copy file
			if isdir(dst[i]):  # into dir (preserve file name)
				mkdirs(dst[i])
				dst[i] = os.path.join(dst[i], src_file_name[i])
			shutil.copy2(src[i], dst[i], *args, **kwargs)


def open_file(file):
	if platform.system() == "Windows":
		os.startfile(file)
	elif platform.system() == "Darwin":
		subprocess.Popen(["open", file])
	else:
		subprocess.Popen(["xdg-open", file])
