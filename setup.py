import os

os.system('sudo -H pip install pipreqs')
os.system('pipreqs --force ./')
os.system('sudo -H pip install -r ./requirements.txt')
