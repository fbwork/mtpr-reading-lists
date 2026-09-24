import os
import argparse
import logging

FLAG_DIR = 'completion_flags'
os.makedirs(FLAG_DIR, exist_ok=True) 

def get_flag_path(module_name):
    return os.path.join(FLAG_DIR, f"{module_name}_done.flag")

def is_module_done(module_name):
    return os.path.exists(get_flag_path(module_name))

def mark_module_done(module_name):
    with open(get_flag_path(module_name), 'w') as f:
        f.write('done')

def clear_module_flag(module_name):
    if os.path.exists(get_flag_path(module_name)):
        os.remove(get_flag_path(module_name))
