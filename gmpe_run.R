# =============================================================================
# Ground Motion Model (GMM) Regression Pipeline
# =============================================================================
# Workflow:
#   1. gmm_new():        fit a mixed-effects GMM via lme4::lmer (or robustlmm::rlmer)
#                         and write coefficient table + split residuals to CSV
#   2. calculate_residuals():  compute total, between-event (dBe), site-to-site (dS2S),
#                         and within-event residuals relative to the fitted GMM
#   3. gmm_eval():       generate scenario predictions and diagnostic plots
#
# Supported networks : ESM/Europe (uses Kotha et al. 2020 functional form)
#                      Other/Japan/KiK-net/K-net (uses Loviknes 2020 / YA15 functional form)
# Supported domains  : 'SA' :  spectral acceleration
#                      'FAS':  Fourier amplitude spectrum
#
# =============================================================================

# Load required libraries
required_packages <- c(
  "lme4", "Matrix", "ggplot2", "ggthemes", "scales",
  "plyr", "dplyr", "stringr", "tidyr", "viridis",
  "gtable", "gridExtra", "grid", "robustlmm"
)

missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

invisible(lapply(required_packages, library, character.only = TRUE))


######### LOAD DATA #########
getwd()
#setwd()

source("gmm_functions.R")

network <- 'ESM'

# Set domain: 'SA' for spectral acceleration, 'FAS' for Fourier amplitude spectrum.
domain <- 'SA'
# domain <- 'FAS'

# Input CSV naming convention: selected_data_<network><domain>.csv
selected_data <- read.csv(paste("selected_data_",network,domain,".csv", sep = ""))
print(nrow(selected_data))

#### Magnitude binning (for plotting####
selected_data[["Mbin"]] <- ifelse(selected_data$MAG < 4.5, "M<4.5",
                                  ifelse(selected_data$MAG < 5.5, "4.5\u2264M<5.5",
                                         ifelse(selected_data$MAG < 6.5, "5.5\u2264M<6.5", "6.5\u2264M")))

selected_data[["Mbin"]] <- factor(selected_data$Mbin, c("M<4.5", "4.5\u2264M<5.5", "5.5\u2264M<6.5", "6.5\u2264M"))


#### Select regression periods / frequencies ####
if (domain == "FAS") {
  # Fourier frequencies in Hz
  periods <- c("0.550", "0.603", "0.725", "0.851", "1.000", "1.660",
               "2.042", "2.455", "3.091", "3.390", "4.075", "4.572", "5.014",
               "5.130", "6.028", "7.416", "8.131", "8.713", "9.776", "10.004",
               "11.225", "12.028", "12.888", "13.810", "14.461", "15.142",
               "16.225", "17.386", "18.630", "19.508", "20.427")
} else {
  # Spectral periods in seconds; "PGA" is treated as T=0.01 s internally
  periods <- c("PGA", 0.010, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060, 0.075,
               0.090, 0.100, 0.120, 0.150, 0.170, 0.200, 0.300, 0.400, 0.500,
               0.600, 0.750, 1.000, 1.200, 1.500, 1.600, 1.700, 1.800, 2.000,
               2.500, 3.000, 4.000, 5.000, 7.500, 10.000)
  periods <- c("PGA", 0.100, 1.000) # For quick testing; comment out for full regression
}

##### Remove potential nonlinear records ######
# ESM flatfile stores PGA in cm/s; Japan networks store it in g.
if (network == "ESM") {
  lin_data <- subset(selected_data, selected_data$PGA < 0.05 * 981)
} else {
  lin_data <- subset(selected_data, selected_data$PGA < 0.05)
}
print(nrow(lin_data))

########## REGRESSION ############
# Alg options: "lmer" (standard REML via lme4) 
#  or "rlmm" (robust via robustlmm; computationally expensive)
Alg <- "lmer"
savefigto <- paste("gmm_", network, domain, Alg, sep = "")

new_gmm = TRUE  # Set to FALSE to skip regression and load existing coefficient table
gmpe <- "Gmm"
if (new_gmm) {
  print("Fitting GMM and generating coefficient table")
  coeff_table <- gmm_new(lin_data, periods, domain, network, Alg,
                        min_no_event = 3, min_no_site = 3,
                        do_plot = TRUE, savefigto = savefigto)
  # For scenario plotting
  gmpe_res <- read.csv(paste("gmm_data_",network,domain,Alg,".csv", sep=""))
} else {
  print("Loading existing coefficient table and split total residuals into between-event, site-to-site, and within-event components")
  coeff_table <- read.csv(paste("coeff_table_", network, domain, ".csv", sep = ""))
  #Split total residuals into between-event, site-to-site, and within-event components
  gmpe_res <- calculate_residuals(lin_data, periods, gmpe, coeff_table, domain, network, Alg,
                           min_no_event = 3, min_no_site = 3,
                           do_plot = TRUE, savefigto = savefigto)
}

# Generate scenario predictions and evaluation plots
scenarios_allM <- gmpe_eval(gmpe_res, coeff_table, periods, domain, network,
                            gmpe = gmpe, savefigto = savefigto)
write.csv(scenarios_allM, paste("gmm_", network, "_scenario_", domain, ".csv", sep = ""))
