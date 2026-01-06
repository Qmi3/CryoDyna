from setuptools import setup, find_packages
import subprocess
# copied from https://github.com/open-mmlab/mmdetection/blob/master/setup.py
def check_dssp():
    """检查并安装dssp"""
    try:
        subprocess.run(['which', 'dssp'], check=True, capture_output=True)
        print("✓ dssp已安装")
    except:
        print("正在安装dssp...")
        subprocess.run(['conda', 'install', '-c', 'ostrokach', 'dssp'], check=False)
        
def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.
    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs
    Returns:
        List[str]: list of requirements items
    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    # import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


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
          install_requires=parse_requirements('requirements.txt'),
          package_data={"cryodyna": ["martini/*.pkl"]},
          )
    check_dssp()
