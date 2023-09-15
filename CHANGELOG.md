# CHANGELOG



## v1.0.0 (2023-09-14)

### Breaking

* feat(kxx_calib): removed all polynomial interpolation funcs + others

BREAKING CHANGE: chebyDeriv and polynomial interpolations unavailable
after this ([`f953741`](https://github.com/vikramnn/python-data-analysis/commit/f953741682bf880879808d2b1a11096f503366f7))

### Documentation

* docs(kxx_calib): added docstrings to most functions ([`585db92`](https://github.com/vikramnn/python-data-analysis/commit/585db92ca11461030c1237aadcf1a75e183d6fd7))


## v0.2.0 (2023-09-15)

### Feature

* feat(kxx_calib): another test function ([`808437b`](https://github.com/vikramnn/python-data-analysis/commit/808437b906dba47841f949e0c0f4ecee156b803f))


## v0.1.0 (2023-09-15)

### Feature

* feat(kxx_calib.py): adding a test function to test semantic-release ([`3c446af`](https://github.com/vikramnn/python-data-analysis/commit/3c446af9103f3ca1c5a68370aaf7601294e02ee1))


## v0.0.0 (2023-09-15)

### Documentation

* docs(kxx_calib): Adding documentation to plotCalib ([`214a6ca`](https://github.com/vikramnn/python-data-analysis/commit/214a6ca74e08d437b5bf1dc91ca4c2f3b8a134d4))

### Unknown

* New directory structure from cookiecutter to formulate analysis code
into a package ([`e8994ef`](https://github.com/vikramnn/python-data-analysis/commit/e8994ef95cf0d1e795a4ee6a5caf5ff453a71d90))

* Added yaxis labels to minMaxinterpspline ([`8ce4c5c`](https://github.com/vikramnn/python-data-analysis/commit/8ce4c5c6ec08d2ef11c7f4fb24b6bf41b9984e2b))

* Adding .gitignore ([`229a8da`](https://github.com/vikramnn/python-data-analysis/commit/229a8da2bc9ebadd5e630fb1c7c4f65ef4c21b92))

* Minor changes. ([`15d1c54`](https://github.com/vikramnn/python-data-analysis/commit/15d1c54965f5dde9bbe406d45a21ae86c1c5d126))

* Changing base 10 logs to natural logs ([`9f7d42b`](https://github.com/vikramnn/python-data-analysis/commit/9f7d42b7487bac2672ecbf2407357e1f17b5d99c))

* Added thermometerMR function to inspect magnetoresistance of therms ([`0b63182`](https://github.com/vikramnn/python-data-analysis/commit/0b63182e004728360d03dc80a2356adcbcfacf6f))

* kxx_calib.py (minMax): Function for extracting min and max R-value ([`5b72043`](https://github.com/vikramnn/python-data-analysis/commit/5b720438de262eaeaf8cf4cb6e983df5eb814452))

* Added a couple of functions for plotting cheby and cheby deriv (still need to work on deriv). Changed base 10 logs to natural logs. ([`dfe95cc`](https://github.com/vikramnn/python-data-analysis/commit/dfe95cc4d78b43110c1e21eb36242cd00116b9b3))

* Changing functions so that thermometer data is not hard-coded in and can take a list of dataframe columns. This commit changes interpPolySpline1D. ([`50604aa`](https://github.com/vikramnn/python-data-analysis/commit/50604aa2cbd0630fa8279aebb81f49de3dcf7b38))

* Initial commit of my python data analysis files. So far, I only have general purpose data cleaning (data_cleaning.py) and specific functions for analyzing thermal conductivity data (kxx_calib.py). I am wondering how I should include my org-roam notes with babel blocks. ([`cbf12be`](https://github.com/vikramnn/python-data-analysis/commit/cbf12be73b9d0cf6fb6194f180e21ec25450404d))
