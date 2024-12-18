"""
This module contains functions to operate over file systems.
"""
from pathlib import Path
import subprocess
import shutil

from pyspark.dbutils import DBUtils


def package_and_move_wheel(source_path: str, target_volume_path: str, dbutils: DBUtils, overwrite: bool=False):
    """
    Packages the Python code at `source_path` and moves the resulting wheel file to `target_volume_path`.

    Args:
      source_path (str): The path to the Python code to package.
      target_volume_path (str): The path to the volume to move the wheel file to.
      dbutils (DBUtils): The DBUtils object to use for file system operations.
      overwrite (bool): Whether to overwrite the wheel file if it already exists in the target volume. Defaults to False.
    """
    # Step 1.1: Ensure temp directory on driver node is empty
    temp_dir = Path("/tmp/yolov5_package")
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)
    
    # Step 1.2: Copy files to temp directory on driver node
    dbutils.fs.cp(f"dbfs:{source_path}", f"file:{temp_dir}", recurse=True)

    # Confirm files are copied
    if not temp_dir.exists():
      raise FileNotFoundError(f"Temporary directory {temp_dir} not found after copy.")

    # Step 2: Build the wheel in the temporary directory
    subprocess.run(["python", "setup.py", "bdist_wheel"], check=True, cwd=temp_dir)

    # Step 3: Locate the wheel file
    dist_path = temp_dir / "dist"
    wheel_files = list(dist_path.glob("*.whl"))
    if not wheel_files:
      raise FileNotFoundError("No .whl file found in the dist directory.")
    
    wheel_name = wheel_files[0].name

    # Step 4: Check if wheel file already exists in target volume
    target_file = f"{target_volume_path}/{wheel_name}"
    if not overwrite and Path(target_file).exists():
      raise FileExistsError(f"Wheel file {wheel_name} already exists in target volume.")
    
    # Step 5: Move wheel file from local driver node to target volume
    dbutils.fs.cp(
        f"file:{dist_path}/{wheel_name}",
        f"dbfs:{target_volume_path}/{wheel_name}"
    )
