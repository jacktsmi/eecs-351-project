import pathlib
import os

def rename_data(dir_path, start_ind=1):
    """
    Renames all files in dir_path to "trainX" where X is a number.

    Inputs:
        dir_path : Path to the directory which has the files we wish to rename
        (For our purposes this will likely be the folder with either the train
        data or test data)

        start_ind : index to start the naming with before iterating
    """
    index = start_ind
    for path in pathlib.Path(dir_path).iterdir():
        if path.is_file():
            directory = path.parent
            old_extension = path.suffix
            new_name = "train" + str(index) + old_extension
        path.rename(pathlib.Path(directory, new_name))
        index = index + 1