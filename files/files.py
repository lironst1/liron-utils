import os
import platform
import subprocess
import shutil
import natsort

import numpy as np

# todo: copy/paste files and folders

move_file = os.rename
remove_file = os.remove
natural_sort = natsort.natsorted


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


import os
import shutil


def mkdirs(path):
	"""Create directory if it doesn't exist."""
	os.makedirs(path, exist_ok=True)


def copy(src, dst, overwrite=True, symlink=False):
	"""
	Copy file(s) or directories, or create symbolic links.

	Parameters
	----------
	src : str | list[str]
		Source file(s) or directory(ies).
	dst : str | list[str]
		- If 'src' is a string, 'dst' can be a target path or directory.
		- If 'src' is a list, 'dst' must be either a directory or a list of same length.
	overwrite : bool, default=True
		Overwrite destination if it exists.
	symlink : bool, default=False
		If True, create symbolic links instead of copying files/directories.
	"""

	# Normalize src and dst to lists
	if isinstance(src, str):
		src = [src]
	if isinstance(dst, str):
		if len(src) > 1 and not os.path.splitext(dst)[1]:
			# If dst has no extension and src has multiple items, treat dst as a directory
			dst = [os.path.join(dst, os.path.basename(s)) for s in src]
		else:
			dst = [dst]

	if len(dst) == 1 and len(src) > 1:  # convert dst from directory to list of filenames
		dst = [os.path.join(dst[0], os.path.basename(s)) for s in src]

	assert len(src) == len(dst), "dst must be a single path or a list of same length as src."

	for s, d in zip(src, dst):
		if not overwrite and os.path.exists(d):  # Skip if not overwriting and destination exists
			continue

		if symlink:
			if os.path.exists(d):  # Remove existing destination if exists
				os.remove(d)
			os.symlink(s, d, target_is_directory=True)  # Create symlink to source directory
			continue

		mkdirs(os.path.dirname(d))  # Ensure the parent directory exists

		if os.path.isdir(s):  # copy directory
			shutil.copytree(s, d)  # Copy directory recursively (todo: check)

		elif os.path.islink(s):
			if os.path.exists(d):  # Remove existing destination if exists
				os.remove(d)
			s = os.readlink(s).replace("\\\\?\\", "")  # Normalize Windows symlink path
			os.symlink(s, d)  # Create symlink to source file

		else:  # copy file
			if not os.path.splitext(d)[1]:  # If destination is a directory (i.e., has no extension), append filename
				d = os.path.join(d, os.path.basename(s))
			shutil.copy2(s, d)  # Copy file with metadata


def open_file(file):
	if platform.system() == "Windows":
		os.startfile(file)
	elif platform.system() == "Darwin":
		subprocess.Popen(["open", file])
	else:
		subprocess.Popen(["xdg-open", file])
