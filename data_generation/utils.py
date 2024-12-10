import os
from pathlib import Path

def check_or_make_folder(folder_path):
    """
    Helper function that (safely) checks if a dir exists; if not, it creates it
    """
    
    folder_path = Path(folder_path)

    try:
        folder_path.resolve(strict=True)
    except FileNotFoundError:
        print("{} dir not found, creating it".format(folder_path))
        os.mkdir(folder_path)