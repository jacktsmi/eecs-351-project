import pathlib

def rename_data():
    index = 1
    for path in pathlib.Path("train").iterdir():
        if path.is_file():
            directory = path.parent
            old_extension = path.suffix
            new_name = "train" + str(index) + old_extension
        path.rename(pathlib.Path(directory, new_name))
        index = index + 1