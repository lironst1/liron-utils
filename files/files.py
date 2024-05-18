import os
import platform
import subprocess

# todo: copy/paste files and folders, make/edit Word documents

move_file = os.rename


def open_file(path):
	if platform.system() == "Windows":
		os.startfile(path)
	elif platform.system() == "Darwin":
		subprocess.Popen(["open", path])
	else:
		subprocess.Popen(["xdg-open", path])
