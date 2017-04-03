# WMO-WLS: Windowed multiscan optimization using weighted least squares

For matching we use [CSM](https://github.com/AndreaCensi/csm). Precompiled
`libcsm.so` and `sm2` for 64-bit Linux located inside `csm` folder.
We use slightly modified version of the CSM, so if you want to compile it
yourself it's recommended to apply patch `csm.patch` first.

## Usage
Download datasets archive:
```sh
wget https://github.com/SkRobo/wmo-wls/releases/download/0.1/datasets.tar.bz2
```

And unpack it:
```sh
$ bzip2 -dc datasets.tar.bz2 | tar xv
```

Next perform matching:
```sh
$ ./match.py
```
Results will be saved in the `./results/match/` folder.

Finally perform optimization:
```sh
$ ./wls.py
```
Results in the form of trajectories will be saved in the
`./results/wls/` folder.

To perfrom nonlinear optimization for set of alphas run:
```sh
$ ./nonlinear.py 1
```
Here argument is the index of dataset to be optimized for.
Results in the form of trajectories will be saved in the
`./results/nonlinear/` folder.

If you don't want to wait for nonlinear optimization to end you can download
precomputed results:
```sh
wget https://github.com/SkRobo/wmo-wls/releases/download/0.1/L-BFGS-B.tar.bz2
```

To unpack them run:
```sh
$ bzip2 -dc L-BFGS-B.tar.bz2 | tar xv -C results/
```

To plot figures used in the paper run:
```sh
$ ./plot_figures.py
```
Figures will be saved in the `./figures/` folder.

## Perfomance
This code is created for scientific purposes only. It is not intended for
uses in practical applications. Thus various optimizations are applicable
which can significantly boost up computational and memory efficiency.
