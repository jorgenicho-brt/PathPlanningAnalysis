import os
import sys
from pathlib import Path

# adding paths to PYTHON path does not work 
def add_pypl_path():
    planning_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/lib'))
    if planning_lib_path not in sys.path: 
        sys.path.insert(0, planning_lib_path)
        print('Added {} to python path'.format(planning_lib_path))
    
        # listing items in path
        dir_contents = [f for f in Path(planning_lib_path).iterdir()]
        for f  in dir_contents:
            #print('Found {} {}'.format( ('dir' if f.is_dir() else 'file') , str(f)))
            f_str = str(f)
            if f.suffix == '.so' and  f_str not in sys.path:
                sys.path.insert(0, f_str)
                print('Added library {} to python path'.format(f_str))

# Adding softlinks to PyPL library files
def setup_pypl():
    current_dir_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__))))
    pypl_slink_path = current_dir_path / 'pypl'
    pypl_library_path = Path(os.path.abspath(os.path.join(str(current_dir_path), '../build/lib')))
    if not pypl_slink_path.exists():    
        os.symlink(pypl_library_path, pypl_slink_path)
    dir_contents = [f for f in pypl_library_path.iterdir()]
    for f  in dir_contents:
            f_slink = current_dir_path / f.name
            if f.suffix == '.so' and not f_slink.exists():
                os.symlink(f, f_slink)
                print('Added soft link to {} to library'.format(str(f_slink), str(f)))

setup_pypl()
            