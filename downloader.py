import os

dir = os.getcwd()
output = os.path.join(dir, "tracks")
playlist_link = ""

os.system("spotdl " + playlist_link + " --output " + output)