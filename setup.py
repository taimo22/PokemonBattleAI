from setuptools import setup, find_packages



def setup_dependency():
    
    requirement_txt_path = "PPO\\requirements.txt"
    with open(requirement_txt_path) as requirements:
        packages = requirements.readlines()
        
    
    setup(name="requirements", install_requires=packages, packages=find_packages())
    


if __name__ == "__main__":

    setup_dependency()