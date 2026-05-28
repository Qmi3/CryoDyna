from setuptools import setup, find_packages
        
if __name__ == "__main__":
    # currently configuration with pyproject.toml is *BETA*
    setup(name='cryodyna',
          version='0.1.0',
          packages=find_packages(exclude=("projects", "assets")),
          include_package_data=True,
          entry_points=
          {'console_scripts': [
              'cstar_show_mrc_info=cryodyna.cli_tools.sak:show_mrc_info',
              'cstar_center_origin=cryodyna.cli_tools.sak:center_origin',
              'cstar_generate_gaussian_density=cryodyna.cli_tools.sak:generate_gaussian_density'
          ], },
          package_data={"cryodyna": ["martini/*.pkl"]},
          )
