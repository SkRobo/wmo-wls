# WMO-WLS: Windowed Multiscan Optimization using Weighted Least Squares

For scan-matching we use [CSM](https://github.com/AndreaCensi/csm). Precompiled
`libcsm.so` and `sm2` for 64-bit Linux located inside `csm` folder.
We are using slightly modified version of the CSM, so if you want to compile it
yourself it's recommended to apply patch `csm.patch` first.

## Dependencies installation
For modern Debian based distribution you can execute the following command:
```sh
sudo apt-get install libgsl2 python3 python3-numpy python3-scipy python3-matplotlib python3-cffi texlive texlive-latex-extra dvipng
```

Additionally you'll need progressbar2 Python library, you can install it using `pip3`:
```sh
sudo pip3 install progressbar2
```


## Usage
To begin download and unpack datasets archive:
```sh
wget https://github.com/SkRobo/wmo-wls/releases/download/0.1/datasets.tar.bz2
bzip2 -dc datasets.tar.bz2 | tar xv
```

If you only want to build figures or examine data you can download results using this command:
```sh
wget https://github.com/SkRobo/wmo-wls/releases/download/0.2/results.tar.bz2
bzip2 -dc results.tar.bz2 | tar xv
```

Otherwise execute the following commands. First perform matching:
```sh
./match.py
```
Results will be saved in the `./results/match/` folder.

Finally perform optimization:
```sh
./wls.py
```
Results in the form of trajectories will be saved in the
`./results/wls/` folder.

To calculate trajectories using keyframe apporach run:
```sh
./keyframes.py
```

To perfrom nonlinear optimization for set of alphas run:
```sh
./nonlinear.py
```
Here argument is the index of dataset to be optimized for.
Results in the form of trajectories will be saved in the
`./results/nonlinear/` folder.

Finally to plot figures used in the WMO-WLS paper run:
```sh
$ ./plot_figures.py
```
Figures will be saved in the `./figures/` folder.

## Perfomance
This code is created for scientific purposes only. It is not intended for
uses in practical applications. Thus various optimizations are applicable
which can significantly boost up computational and memory efficiency.
