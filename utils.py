import pathlib

def rename_data(dir_path):
    """
    Renames all files in dir_path to "trainX" where X is a number.

    Inputs:
        dir_path : Path to the directory which has the files we wish to rename
        (For our purposes this will likely be the folder with either the train
        data or test data)
    """
    index = 1
    for path in pathlib.Path(dit_path).iterdir():
        if path.is_file():
            directory = path.parent
            old_extension = path.suffix
            new_name = "train" + str(index) + old_extension
        path.rename(pathlib.Path(directory, new_name))
        index = index + 1