import os

dir = os.getcwd()
output = os.path.join(dir, "tracks")
playlist = ""

os.system("spotdl " + playlist + " --output " + output)