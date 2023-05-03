import platform 
import os

if platform.system() == "Windows":
    os.add_dll_directory('C:\\Users\\circleci\\project\\conda\\Library\\bin')
