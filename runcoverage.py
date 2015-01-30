import subprocess


if __name__ == "__main__":
    # coverage options set in .coveragerc
    subprocess.call("coverage erase", shell=True)
    subprocess.call("coverage run runtests.py", shell=True)
    subprocess.call("coverage report -m", shell=True)
