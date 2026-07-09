# Splitting of residuals with Ground Motion Model (GMM) Regression Functions

This repository contains implementations for deriving GMMs and residuals, such as between-event (δBe) and site-to-site residuals (δS2S), using linear mixed-effects regression (Bates et al., 2015; Stafford et al., 2014) to develop Ground Motion Models (GMM)

## Files

- **`gmm_functions.R`** - R implementation of GMM regression functions
- **`gmpe_run.R`** - Example script demonstrating how to run the regression workflow in R
- **`gmm_functions.py`** - Python implementation (translated from R, uses pymer4 for mixed models)

- **`py_requirements.txt`** - List of Python packages requirements 

## Key Features

### Regression Models
- Linear mixed-effects models with event and site random effects
- Support for both standard (`lmer`) and robust (`rlmer`) regression algorithms (note that `rlmer` is only avaliable in the R-codes and computionally heavy)
- Handles Spectral Acceleration (SA) and Fourier Amplitude Spectra (FAS) domains

### GMM functional forms
The models estimate ground motion scaling with:
- **Magnitude scaling**: Piecewise linear with hinge at Mh (reference magnitude)
- **Distance attenuation**: Geometric spreading and anelastic attenuation
- **Site effects**: Station-to-station residuals (δS2S)
- **Event terms**: Between-event residuals (δBe)

### Supported regions and networks

##### A region specific GMM should always be used. This repository supports the following two regions:
- **Europe and Mediterranean region**:
    When using the ESM flatfile, or ground motions from Europe and Mediterranean region in general, the functional form based on Kotha et al (2020) for SA and Kotha et al (2022) for FAS should be used.
    - **ESM (European Strong-Motion)**
        - Download from: https://esm-db.eu/#/products/flat_file
        - Use the `read_select_ESM()` function in **`gmm_functions.py`** to read and select records of interest from the original ESM flatfile.
        - Note: the ESM flatfile stores PGA in **cm/s²**; the linearity threshold is therefore `PGA < 0.05 × 981 cm/s²`.

- **Japan**: 
    When using Japanese records, for example from the KiK-net or K-Net strong motion networks (NIED), use a functional form based on Kotha et al (2018) and modified by Loviknes et al (2021)
    - **KiK-net and K-NET flatfile (Loviknes et al., 2026)**:
        - Download from https://doi.org/10.5880/GFZ.LKUT.2026.003.
        - Use the `read_select_KiK_Knet()` function in **`gmm_functions.py`** to read and select records of interest.
        - Note: the KiK-net/K-NET flatfile stores PGA in **g**; the linearity threshold is `PGA < 0.05 g`.
    
## Main Functions

### Regression Functions
- `gmm_new()` (R and Python) - Fit new GMM coefficients from data; residuals are relative to the mean across all sites/events. Accepts `do_plot` and `savefigto` to save diagnostic figures.
- `calculate_residuals()` (R and Python) - Split total residuals (relative to GMM prediction) into δBe, δS2S, and δWSes. Accepts `do_plot` and `savefigto`. Use when loading an existing coefficient table instead of fitting a new model.
- `gmm_eval()` / `evaluate_residuals()` (R / Python) - Generate magnitude–distance scenario predictions and scaling plots.

### Helper Functions
- `read_select_ESM()` / `read_select_KiK_Knet()` - Reads and seelect the ground motion of interest
- `initialize_coeff_table()` - Initialize coefficient table structure
- `prepare_esm_predictors()` / `prepare_other_predictors()` - Prepare predictor variables
- `fit_mixed_model()` - Fit mixed-effects model (Python uses pymer4)

## Outputs

- **Coefficient tables**: GMM model coefficients
- **Residual tables**: Between-event (δBe) and station (δS2S) terms
- **Diagnostic plots**: Residual distributions, magnitude/distance scaling
- **Scenario predictions**: Ground motion predictions for magnitude-distance scenarios


## Requirements

The functions have been developed on Python 3.13 (using pymer4, se `py_requirements.txt`) and R 4.5.3


## References
 - Bates, D., et al. (2015). Fitting Linear Mixed-Effects Models Using lme4. Journal of Statistical Software, 67(1), 1-48.
 - Kotha, S. R., F. Cotton, and D. Bindi (2018). A new approach to site classification: Mixed-effects ground motion prediction equation with spectral clustering of site amplification functions, Soil Dynam. Earthq. Eng. 110, 318–329.
 - Kotha, S. R., Weatherill, G., Bindi, D., & Cotton, F. (2020). A regionally-adaptable ground-motion model for shallow crustal earthquakes in Europe. Bulletin of Earthquake Engineering, 18(9), 4091-4125.
 - Kotha, S. R., Bindi, D., & Cotton, F. (2022). A regionally adaptable ground-motion model for fourier amplitude spectra of shallow crustal earthquakes in Europe. Bulletin of Earthquake Engineering, 20(2), 711-740.
 - Lanzano G., Luzi L., Cauzzi C., Bienkowski J., Bindi D., Clinton J., Cocco M., D’Amico M., Douglas J., Faenza L., Felicetta C., Gallovic F., Giardini D., Ktenidou O., Lauciani V., Manakou M., Marmureanu A., Maufroy E., Michelini A., Özener H., Puglia R., Rupakhety R., Russo E., Shahvar M., Sleeman R., Theodoulidis N.; Accessing European Strong‐Motion Data: An Update on ORFEUS Coordinated Services. Seismological Research Letters 2021;; 92 (3): 1642–1658. https://doi.org/10.1785/0220200398 
 - Loviknes, K., et al. (2021). Testing nonlinear amplification factors of ground-motion models. BSSA, 111(5), 2121-2137.
 - Loviknes, K., Cotton, F., & Weatherill, G. (2024). Exploring inferred geomorphological sediment thickness as a new site proxy to predict ground-shaking amplification at regional scale: application to Europe and eastern Türkiye. Natural Hazards and Earth System Sciences, 24(4), 1223-1247.
 - Loviknes, Karina; von Specht, Sebastian; Lilienkamp, Henning; Händel, Annabel; Zhu, Chuanbin; Cotton, Fabrice (2026): KiK-net and K-NET flatfile with automatically processed ground motions and associated metadata. V. 0002. GFZ Data Services. https://doi.org/10.5880/GFZ.LKUT.2026.003
 - Luzi L., Puglia R., Russo E., D'Amico M., Felicetta C., Pacor F., Lanzano G., Çeken U., Clinton J., Costa G., Duni L., Farzanegan E., Gueguen P., Ionescu C., Kalogeras I., Özener H., Pesaresi D., Sleeman R., Strollo A., Zare M. (2016). The Engineering Strong‐Motion Database: A Platform to Access Pan‐European Accelerometric Data. Seismological Research Letters ; 87 (4): 987–997. doi:https://doi.org/10.1785/0220150278
 - National Research Institute for Earth Science and Disaster Resilience (2019) NIED K-NET, KiK-net, National Research Institute for Earth Science and Disaster Resilience. https://www.doi.org/10.17598/NIED.0004
 - Stafford, P. J. (2014). Crossed and nested mixed-effects approaches for enhanced model development and removal of the ergodic assumption in empirical ground-motion models. BSSA, 104(2), 702-719.
