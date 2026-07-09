"""
Ground Motion Model (GMM) functions for deriving and evaluating GMMs 
and residuals, such as between-event (δBe) and site-to-site 
residuals (δS2S), using linear mixed effect regression (lmer). 
"""
#%%
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import os
import geopandas as gpd
from shapely.geometry import Point

from scipy.signal import windows
from scipy.constants import g
from typing import Tuple, List, Union

# pymer4 fits lme4 models via rpy2 under the hood, but works natively with
# polars DataFrames instead of pandas, so model inputs/outputs are converted
# at the fit_mixed_model/fit_residuals boundary.
from pymer4.models import lmer
from pymer4.tidystats.broom import tidy


def initialize_coeff_table(periods: list, network: str) -> pd.DataFrame:
    """Initialize the coefficient table based on network type."""
    
    if network == "ESM":
        coeff_table = pd.DataFrame({
            'period': periods,
            'a': 0.0, #called 'e1' in the Kotha 2020 model
            'b1': 0.0,
            'Mref': 4.5,
            'b2': 0.0,
            'Mh': 5.7, # 6.2 in orginal Kotha 2020 GMM, but later adjusted to 5.7
            'b3': 0.0,
            'Rref': 30.0,
            'c1': 0.0,
            'h_D10': 4.0,
            'h_10D20': 8.0,
            'h_20D': 12.0,
            'c2': 0.0,
            'c3': 0.0,
            'phis2s': 0.0,
            'tau': 0.0,
            'phi0': 0.0,
            'sigma': 0.0,
            'rock_adjustement': 0.0
        })
    else:
        coeff_table = pd.DataFrame({
            'period': periods,
            'a': 0.0,
            'b1': 0.0,
            'Mref': 4.5,
            'b2': 0.0,
            'Mh': 0.0,
            'b3': 0.0,
            'c1': 0.0,
            'Rs': 100.0,
            'c2': 0.0,
            'c3': 0.0,
            'phis2s': 0.0,
            'tau': 0.0,
            'phi0': 0.0,
            'sigma': 0.0,
            'rock_adjustement': 0.0
        })
    
    return coeff_table


def prepare_esm_predictors(ff_ss: pd.DataFrame, t: str, coeff_table: pd.DataFrame) -> pd.DataFrame:
    """Prepare predictors for ESM specification."""
    period_idx = coeff_table['period'] == t
    
    Mref = coeff_table.loc[period_idx, 'Mref'].iloc[0]
    Mh = coeff_table.loc[period_idx, 'Mh'].iloc[0]
    
    ff_ss['Mref'] = Mref
    ff_ss['Mh'] = Mh
    
    # Magnitude scaling coefficients
    ff_ss['b1'] = np.where(ff_ss['MAG'] <= ff_ss['Mh'], 
                           ff_ss['MAG'] - ff_ss['Mh'], 0)
    ff_ss['b2'] = np.where(ff_ss['MAG'] <= ff_ss['Mh'], 
                           (ff_ss['MAG'] - ff_ss['Mh'])**2, 0)
    ff_ss['b3'] = np.where(ff_ss['MAG'] <= ff_ss['Mh'], 
                           0, ff_ss['MAG'] - ff_ss['Mh'])
    
    # Distance scaling
    Rref = coeff_table.loc[period_idx, 'Rref'].iloc[0]
    ff_ss['Rref'] = Rref
    
    # Depth binning
    ff_ss['Dbin'] = pd.cut(ff_ss['ev_depth_km'],
                           bins=[-np.inf, 10, 20, np.inf],
                           labels=['D<10km', '10km≤D<20km', '20km≤D'],
                           right=False)
    
    # Depth-dependent h parameter
    h_D10 = coeff_table.loc[period_idx, 'h_D10'].iloc[0]
    h_10D20 = coeff_table.loc[period_idx, 'h_10D20'].iloc[0]
    h_20D = coeff_table.loc[period_idx, 'h_20D'].iloc[0]
    
    ff_ss['h'] = ff_ss['Dbin'].map({
        'D<10km': h_D10,
        '10km≤D<20km': h_10D20,
        '20km≤D': h_20D
    }).astype(float)
    
    # Distance coefficients
    ff_ss['c1'] = np.log(np.sqrt(ff_ss['RJB']**2 + ff_ss['h']**2) / 
                        np.sqrt(ff_ss['Rref']**2 + ff_ss['h']**2))  
    ff_ss['c2'] = (ff_ss['MAG'] - ff_ss['Mref']) * \
                  np.log(np.sqrt(ff_ss['RJB']**2 + ff_ss['h']**2) / 
                        np.sqrt(ff_ss['Rref']**2 + ff_ss['h']**2))
    ff_ss['c3'] = (np.sqrt(ff_ss['RJB']**2 + ff_ss['h']**2) - 
                  np.sqrt(ff_ss['Rref']**2 + ff_ss['h']**2)) / 100
    
    return ff_ss


def prepare_other_predictors(ff_ss: pd.DataFrame, t: str, coeff_table: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Prepare predictors for other specifications."""
    
    period_idx = coeff_table['period'] == t
    
    Rs = coeff_table.loc[period_idx, 'Rs'].iloc[0]
    ff_ss['Rs'] = Rs
    
    # Calculate h parameter (Youngs et al. 1995 style)
    ff_ss['h'] = np.exp(2.303 * np.maximum(
        -0.05 + 0.15 * ff_ss['MAG'],
        -1.72 + 0.43 * ff_ss['MAG']
    ))
    
    # Distance coefficients
    ff_ss['c1'] = np.where(
        ff_ss['RJB'] < ff_ss['Rs'],
        np.log(np.sqrt(ff_ss['RJB']**2 + ff_ss['h']**2)),
        np.log(np.sqrt(ff_ss['Rs']**2 + ff_ss['h']**2))
    )
    
    ff_ss['c2'] = np.where(
        ff_ss['RJB'] >= ff_ss['Rs'],
        np.log(ff_ss['RJB'] / ff_ss['Rs']),
        0
    )
    
    ff_ss['c3'] = np.where(
        ff_ss['RJB'] >= ff_ss['Rs'],
        ff_ss['RJB'] - ff_ss['Rs'],
        0
    )
    
    # Magnitude scaling
    if t == "PGA":
        tt = 0.01 # This is only for the Mh selection below, the value does not have a meaning
    else:
        tt = float(t)
        
    # Update Mh based on domain and period
    if domain == "FAS":
        Mh = 5.7 if tt > 10 else 5.5
    else:
        Mh = 5.5 if tt < 0.1 else 5.7
    
    coeff_table.loc[period_idx, 'Mh'] = Mh
    ff_ss['Mh'] = Mh
    
    Mref = coeff_table.loc[period_idx, 'Mref'].iloc[0]
    ff_ss['Mref'] = Mref
    
    # Magnitude coefficients
    ff_ss['b1'] = np.where(ff_ss['MAG'] < ff_ss['Mref'], 
                           ff_ss['MAG'] - ff_ss['Mref'], 0)
    
    ff_ss['b2'] = np.where(
        ff_ss['MAG'] > ff_ss['Mh'],
        ff_ss['Mh'] - ff_ss['Mref'],
        np.where(ff_ss['MAG'] >= ff_ss['Mref'], 
                ff_ss['MAG'] - ff_ss['Mref'], 0)
    )
    
    ff_ss['b3'] = np.where(ff_ss['MAG'] >= ff_ss['Mh'], 
                           ff_ss['MAG'] - ff_ss['Mh'], 0)
    
    return ff_ss


def _random_effects_dict(m: lmer) -> dict:
    """Build {group_name: DataFrame(index=level, columns=['(Intercept)', 'condsd'])}
    from a fitted pymer4 lmer model, mirroring lme4::ranef(model, condVar=TRUE)."""
    rv = tidy(m.r_model, effects='ran_vals', conf_int=True).to_pandas()

    random_effects_dict = {}
    for group_name, group_df in rv.groupby('group'):
        re_df = group_df.set_index('level')[['estimate', 'std_error']].copy()
        re_df.columns = ['(Intercept)', 'condsd']
        random_effects_dict[group_name] = re_df

    return random_effects_dict


def _residuals_series(m: lmer) -> pd.Series:
    """Per-record residuals, indexed by the 'record_id' column that was attached
    to the model data (the original DataFrame's index, as a string)."""
    data_out = m.data.to_pandas()
    return pd.Series(data_out['resid'].values, index=data_out['record_id'].values)


def fit_mixed_model(ff_ss: pd.DataFrame):
    """Fit the mixed effects model using pymer4."""

    model_data = ff_ss[['IM_values', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3',
                         'EQ_Code', 'StationCode']].copy()
    model_data['record_id'] = ff_ss.index.astype(str)

    m = lmer(
        "IM_values ~ 1 + b1 + b2 + b3 + c1 + c2 + c3 + (1|EQ_Code) + (1|StationCode)",
        data=pl.from_pandas(model_data)
    )
    m.fit()

    # Fixed effects
    fixed_effects_df = (
        m.result_fit
        .select(['term', 'estimate'])
        .rename({'term': 'coefficient', 'estimate': 'Estimate'})
        .to_pandas()
    )

    # Variance components (equivalent to lme4::VarCorr as_data_frame, 'sdcor' column)
    varCorr = (
        m.ranef_var
        .select(['group', 'estimate'])
        .rename({'group': 'grp', 'estimate': 'sdcor'})
        .to_pandas()
    )

    random_effects_dict = _random_effects_dict(m)
    dwses = _residuals_series(m)

    return fixed_effects_df, random_effects_dict, dwses, varCorr


def extract_coefficients(fixed_effects, varCorr, t: str, rock_adjustment: float, IM: str, coeff_table: pd.DataFrame):
    """Extract and store coefficients from fitted model."""
    period_idx = coeff_table['period'] == t
    
    fixed_effects = fixed_effects.set_index('coefficient')
    coeff_table.loc[period_idx, 'a'] = fixed_effects.loc['(Intercept)', 'Estimate']
    coeff_table.loc[period_idx, 'b1'] = fixed_effects.loc['b1', 'Estimate']
    coeff_table.loc[period_idx, 'b2'] = fixed_effects.loc['b2', 'Estimate']
    coeff_table.loc[period_idx, 'b3'] = fixed_effects.loc['b3', 'Estimate']
    coeff_table.loc[period_idx, 'c1'] = fixed_effects.loc['c1', 'Estimate']
    coeff_table.loc[period_idx, 'c2'] = fixed_effects.loc['c2', 'Estimate']
    coeff_table.loc[period_idx, 'c3'] = fixed_effects.loc['c3', 'Estimate']
    
    # Extract variance components
    varCorr = varCorr.set_index('grp')
    coeff_table.loc[period_idx, 'tau'] = varCorr.loc['EQ_Code','sdcor']
    coeff_table.loc[period_idx, 'phis2s'] = varCorr.loc['StationCode','sdcor']
    coeff_table.loc[period_idx, 'phi0'] = varCorr.loc['Residual','sdcor']
    coeff_table.loc[period_idx, 'sigma'] = np.sqrt(varCorr.loc['EQ_Code','sdcor']**2 +
                                               varCorr.loc['StationCode','sdcor']**2 + 
                                               varCorr.loc['Residual','sdcor']**2
                                               )
    
    # Add rock adjustment
    coeff_table.loc[period_idx, 'rock_adjustement'] = rock_adjustment
    
    return coeff_table


def extract_random_effects(random_effects_dict, dwses, 
                            IM: str, tt: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract random effects (residuals) from fitted model."""
    
    # Extract between-event residuals
    dbe_table = pd.DataFrame({
        'IM': IM,
        't': tt,
        'EQ_Code': random_effects_dict['EQ_Code'].index.values,
        'dbe': random_effects_dict['EQ_Code']['(Intercept)'].values,
        'se_dbe': random_effects_dict['EQ_Code']['condsd'].values
    })
    
    # Extract site-to-site residuals
    ds2s_table = pd.DataFrame({
        'IM': IM,
        't': tt,
        'StationCode': random_effects_dict['StationCode'].index.values,
        'ds2s': random_effects_dict['StationCode']['(Intercept)'].values,
        'se_ds2s': random_effects_dict['StationCode']['condsd'].values
    })
    
    # Extract left-over residuals
    dwses_table = pd.DataFrame({
        'IM': IM,
        't': tt,
        'record_id': dwses.index.values,
        'dwses': dwses.values
    })
    
    return dbe_table, ds2s_table, dwses_table


def update_main_tables(dbe_table: pd.DataFrame, ds2s_table: pd.DataFrame,
                       dwses_table: pd.DataFrame, dbe_table0: pd.DataFrame,
                       ds2s_table0: pd.DataFrame, selected_data: pd.DataFrame, 
                       IM: str, t: str):
    """Update main residual tables and data."""

    # dbe_table/ds2s_table are keyed by EQ_Code/StationCode as strings
    # (process_period casts them to str for lmer), so cast the merge keys
    # here too to avoid dtype mismatches silently producing all-NaN columns.
    dbe_by_eq = dbe_table.set_index(dbe_table['EQ_Code'].astype(str))['dbe']
    se_dbe_by_eq = dbe_table.set_index(dbe_table['EQ_Code'].astype(str))['se_dbe']
    ds2s_by_station = ds2s_table.set_index(ds2s_table['StationCode'].astype(str))['ds2s']
    se_ds2s_by_station = ds2s_table.set_index(ds2s_table['StationCode'].astype(str))['se_ds2s']

    # Update dbe_table0
    dbe_table0[f'dbe_{t}'] = dbe_table0['EQ_Code'].astype(str).map(dbe_by_eq)
    dbe_table0[f'se_dbe_{t}'] = dbe_table0['EQ_Code'].astype(str).map(se_dbe_by_eq)
    # Update ds2s_table0
    ds2s_table0[f'ds2s_{t}'] = ds2s_table0['StationCode'].astype(str).map(ds2s_by_station)
    ds2s_table0[f'se_ds2s_{t}'] = ds2s_table0['StationCode'].astype(str).map(se_ds2s_by_station)

    # Update selected_data
    selected_data[f'dbe_{IM}'] = selected_data['EQ_Code'].astype(str).map(dbe_by_eq)
    selected_data[f'ds2s_{IM}'] = selected_data['StationCode'].astype(str).map(ds2s_by_station)
    
    # Create address for matching dwses
    address_to_dwses = dwses_table.set_index('record_id')['dwses']
    selected_data_address = selected_data.index.astype(str)
    selected_data[f'dwses_{IM}'] = selected_data_address.map(address_to_dwses)
    
    # Calculate corrected values
    selected_data[f'site_corrected_{IM}'] = np.exp(
        np.log(selected_data[IM]) - selected_data[f'ds2s_{IM}']
    )
    selected_data[f'event_corrected_{IM}'] = np.exp(
        np.log(selected_data[IM]) - selected_data[f'dbe_{IM}']
    )
    selected_data[f'event_and_site_corrected_{IM}'] = np.exp(
        np.log(selected_data[IM]) -
        selected_data[f'dbe_{IM}'] -
        selected_data[f'ds2s_{IM}']
    )


def generate_plots(ff_ss: pd.DataFrame, IM: str, t: str, tu: str, savefigto: str, network: str):
    """Generate diagnostic plots."""
    os.makedirs(savefigto, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(6, 9))

    # Plot 1: dbe vs Magnitude
    ax1 = axes[0]
    dbe_col = f'dbe_{IM}'

    ff_unique_eq = ff_ss.drop_duplicates(subset=['EQ_Code'])
    if dbe_col in ff_unique_eq.columns:
        ax1.scatter(ff_unique_eq['MAG'], ff_unique_eq[dbe_col],
                   facecolors='none', edgecolors='black', s=15)

        # Binned statistics
        bns = np.logspace(np.log10(np.nanmin(ff_unique_eq['MAG'].astype(float))),
                                  np.log10(np.nanmax(ff_unique_eq['MAG'].astype(float))),6)
        Dpos, Dsd, b0pos = bin_plot(ff_unique_eq['MAG'].astype(float),
                                            ff_unique_eq[dbe_col].astype(float), bns)
        ax1.errorbar(b0pos, Dpos, yerr=Dsd, linestyle='-',
                   fmt='s', color = 'r', ms=8, mew=1.5, capsize=3, elinewidth=1.5)

    ax1.set_title(f'T = {t}{tu}', fontsize=15)
    ax1.set_xlabel('$M_W$', fontsize=12)
    ax1.set_ylabel('$\\delta B_e$', fontsize=12)
    ax1.set_xlim(2.5, 7.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.grid(True, alpha=0.3)

    # Plot 2: ds2s vs VS30
    ax2 = axes[1]
    ff_unique_sites = ff_ss.drop_duplicates(subset=['StationCode'])
    ds2s_col = f'ds2s_{IM}'

    if ds2s_col in ff_unique_sites.columns:
        ax2.scatter(ff_unique_sites['VS30'], ff_unique_sites[ds2s_col],
                   facecolors='none', edgecolors='black', s=15)

        bns = np.logspace(np.log10(np.nanmin(ff_unique_sites['VS30'].astype(float))),
                                  np.log10(np.nanmax(ff_unique_sites['VS30'].astype(float))),6)
        Dpos, Dsd, b0pos = bin_plot(ff_unique_sites['VS30'].astype(float),
                                            ff_unique_sites[ds2s_col].astype(float), bns)
        ax2.errorbar(b0pos, Dpos, yerr=Dsd, linestyle='-',
                   fmt='s', color = 'r', ms=8, mew=1.5, capsize=3, elinewidth=1.5)

    ax2.set_xlabel('$V_{s30}$ (m/s)', fontsize=12)
    ax2.set_ylabel('$\\delta S2S_s$', fontsize=12)
    ax2.set_xscale('log')
    ax2.set_xlim(100, 2000)
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_xticks([180, 360, 760, 1500])
    ax2.set_xticklabels(['180', '360', '760', '1500'])
    ax2.grid(True, alpha=0.3)

    # Plot 3: dwses vs Distance
    ax3 = axes[2]
    dwses_col = f'dwses_{IM}'

    if dwses_col in ff_ss.columns:
        ax3.scatter(ff_ss['RJB'], ff_ss[dwses_col],
                   facecolors='none', edgecolors='black', s=15)

        # Binned statistics
        ff_ss0 = ff_ss[ff_ss['RJB']>0].copy()
        bns = np.logspace(np.log10(np.nanmin(ff_ss0['RJB'].astype(float))),
                                  np.log10(np.nanmax(ff_ss0['RJB'].astype(float))),6)
        Dpos, Dsd, b0pos = bin_plot(ff_ss0['RJB'].astype(float),
                                            ff_ss0[dwses_col].astype(float), bns)
        ax3.errorbar(b0pos, Dpos, yerr=Dsd, linestyle='-',
                   fmt='s', color = 'r', ms=8, mew=1.5, capsize=3, elinewidth=1.5)

    ax3.set_xlabel('$R_{JB}$ (km)', fontsize=12)
    ax3.set_ylabel('$\\delta WS_{es}$', fontsize=12)
    ax3.set_xscale('log')
    ax3.set_xlim(0.5, 600)
    ax3.set_ylim(-2.5, 2.5)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{savefigto}/res_{t}_{network}.jpeg', dpi=300, bbox_inches='tight')
    plt.close()


def fit_residuals(ff_ss: pd.DataFrame):
    """Fit the mixed effects model using pymer4."""

    model_data = ff_ss[['res', 'EQ_Code', 'StationCode']].copy()
    model_data['record_id'] = ff_ss.index.astype(str)

    m = lmer(
        "res ~ 1 + (1|StationCode) + (1|EQ_Code)",
        data=pl.from_pandas(model_data)
    )
    m.fit()

    random_effects_dict = _random_effects_dict(m)
    dwses = _residuals_series(m)

    return random_effects_dict, dwses

def process_period(t: str, selected_data: pd.DataFrame,
                   coeff_table: pd.DataFrame, domain: str, 
                   network: str, 
                   min_no_event: int = 3):
    
    """Process a single period for regression."""
    
    # Determine IM name and numeric period
    if t == "PGA":
        IM = 'PGA'
        tt = 0.01 # This is only for the frequency selection below, the value does not have a meaning
    else:
        IM = f"X{float(t):.3f}"
        tt = float(t)
    
    
    cols_of_interest = ['Address', 'EQ_Code', 'StationCode', 'RJB', 'MAG',
                        'VS30', 'ev_depth_km', 'tHigh', 'fLow', 'fHigh', IM]
    ff_ss = selected_data[cols_of_interest].copy()
    
    # Set EQ_Code to string to ensure the lmer goes smoothly
    ff_ss['EQ_Code'] = ff_ss['EQ_Code'].astype(str)
    ff_ss['StationCode'] = ff_ss['StationCode'].astype(str)

    # Define usable subset for the period
    if domain == "FAS":
        ff_ss = ff_ss[(ff_ss[IM] > 0) & (ff_ss['fLow'] <= tt) & 
                                        (ff_ss['fHigh'] >= tt)]
    else:
        ff_ss = ff_ss[(ff_ss[IM] > 0) & (ff_ss['tHigh'] >= tt)]
    
    print(f"Records after filtering: {len(ff_ss)}")
    
    # Remove events with less than min_no_event records
    event_counts = ff_ss['EQ_Code'].value_counts()
    valid_events = event_counts[event_counts >= min_no_event].index
    ff_ss = ff_ss[ff_ss['EQ_Code'].isin(valid_events)]
    
    print(f"Records after event filtering: {len(ff_ss)}")
    
    # Prepare predictors based on network type
    if network == "ESM":
        ff_ss = prepare_esm_predictors(ff_ss, t, coeff_table)
    else:
        ff_ss = prepare_other_predictors(ff_ss, t, coeff_table, domain)
    
    # Add log IM values
    ff_ss['IM_values'] = np.log(ff_ss[IM])
    
    fixed_effects_df, random_effects_dict, dwses, varCorr = fit_mixed_model(ff_ss)

    # Extract random effects
    dbe_table, ds2s_table, dwses_table = extract_random_effects(random_effects_dict, dwses, IM, tt)
    
    # Calculate rock adjustment as the mean ds2s of rock stations
    ds2s_table['VS30'] = ds2s_table['StationCode'].map(
        ff_ss.drop_duplicates('StationCode').set_index('StationCode')['VS30'])
    rock_adjustment = ds2s_table[ds2s_table['VS30'] > 760]['ds2s'].mean()
        
    # Extract coefficients and update coefficient table
    coeff_table = extract_coefficients(fixed_effects_df, varCorr, t, rock_adjustment, IM, coeff_table)
    
    return coeff_table, dbe_table, ds2s_table, dwses_table


def split_period(t: str, selected_data: pd.DataFrame,
                   coeff_table: pd.DataFrame, domain: str, network: str, 
                   min_no_event: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    """Process a single period for regression."""
    
    # Determine IM name and numeric period
    if t == "PGA":
        IM = 'PGA'
        tt = 0.01 # This is only for the frequency selection below, the value does not have a meaning
    else:
        IM = f"X{float(t):.3f}"
        tt = float(t)
    
    
    cols_of_interest = ['Address', 'EQ_Code', 'StationCode', 'RJB', 'MAG',
                        'VS30', 'ev_depth_km', 'tHigh', 'fLow', 'fHigh', IM]
    ff_ss = selected_data[cols_of_interest].copy()
    
    # Define usable subset for the period
    if domain == "FAS":
        ff_ss = ff_ss[(ff_ss[IM] > 0) & (ff_ss['fLow'] <= tt) & (ff_ss['fHigh'] >= tt)]
    else:
        ff_ss = ff_ss[(ff_ss[IM] > 0) & (ff_ss['tHigh'] >= tt)]
    
    print(f"Records after frequency criteria: {len(ff_ss)}")
    
    # Remove events with less than min_no_event records
    event_counts = ff_ss['EQ_Code'].value_counts()
    valid_events = event_counts[event_counts >= min_no_event].index
    ff_ss = ff_ss[ff_ss['EQ_Code'].isin(valid_events)]
    
    print(f"Records after event criteria: {len(ff_ss)}")
    
    rock_adj = float(coeff_table.loc[coeff_table['period'] == t, 'rock_adjustement'].iloc[0])
    
    # Predict ground motion
    if network == "ESM":
        pred0 = kothaetal2020_epe(t, ff_ss['MAG'].values, 
                                ff_ss['RJB'].values,
                                ff_ss['ev_depth_km'].values,
                                coeff_table, domain)
        pred = np.exp(np.log(pred0) + rock_adj)
    else:
        pred0 = gmpe_allM_YA15_Mh_Mref(t, ff_ss['MAG'].values,
                           ff_ss['RJB'].values, coeff_table)
        pred = np.exp(np.log(pred0) + rock_adj)
    
    ff_ss["res"] = np.log(ff_ss[IM]) - np.log(pred) 
    ff_ss = ff_ss[np.isfinite(ff_ss["res"])].copy()   
    
    random_effects_dict, dwses = fit_residuals(ff_ss)

    # Extract random effects
    dbe_table, ds2s_table, dwses_table = extract_random_effects(random_effects_dict, dwses, IM, tt)
    
    return dbe_table, ds2s_table, dwses_table


def gmm_new(selected_data: pd.DataFrame,
         periods: List, domain: str, 
         network: str, alg: str = 'lmer', min_no_event: int = 3, 
         do_plot: bool = False, 
         savefigto: str = "GMM_figures") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform the GMM regression for all periods.
    
    Parameters:
    -----------
    selected_data : pd.DataFrame
        Input data containing earthquake records
    periods : list
        List of periods for analysis
    domain : str
        'FAS' for Fourier Amplitude Spectra or 'SA' for Spectral Acceleration
    network : str
        Specification type ('ESM' or network name)
    alg : str
        Algorithm to use ('lmer' or 'rlmm' for robust)
    min_no_event : int
        Minimum number of records per event
    do_plot : bool
        Whether to generate plots
    savefigto : str
        Directory to save figures
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Coefficient table, between-event residuals, site-to-site residuals, and updated data
    """

    coeff_table = initialize_coeff_table(periods, network)
    
    # Create tables for site-to-site and between-event residuals
    ds2s_table0 = pd.DataFrame({
        'StationCode': selected_data['StationCode'].unique()
    })
    ds2s_table0['VS30'] = ds2s_table0['StationCode'].map(
        selected_data.drop_duplicates('StationCode').set_index('StationCode')['VS30']
    )
    
    dbe_table0 = pd.DataFrame({
        'EQ_Code': selected_data['EQ_Code'].unique()
    })
    dbe_table0['MAG'] = dbe_table0['EQ_Code'].map(
        selected_data.drop_duplicates('EQ_Code').set_index('EQ_Code')['MAG']
    )
    
    # Determine time unit
    if domain == "FAS":
        tu = 'Hz'
    else:
        tu = 's'
    
    # Loop through periods
    for t in periods:
        print(f"Processing period: {t}")
        
        # Process period
        coeff_table, dbe_table, ds2s_table, dwses_table = process_period(
            t, selected_data.copy(), coeff_table, domain, network, min_no_event
        )
        
        # Determine IM name
        if t == "PGA":
            IM = 'PGA'
        else:
            IM = f"X{float(t):.3f}"
        
        # Update main tables
        update_main_tables(dbe_table, ds2s_table, dwses_table, 
                            dbe_table0, ds2s_table0, selected_data, IM, t)
    
        # Generate plots if requested
        if do_plot:
            generate_plots(selected_data, IM, t, tu, savefigto, network)
    
    return coeff_table, dbe_table0, ds2s_table0, selected_data


def calculate_residuals(selected_data: pd.DataFrame,
         periods: List, domain: str, 
         network: str, coeff_table: pd.DataFrame,
         alg: str = 'lmer', 
         min_no_event: int = 3, 
         do_plot: bool = False, 
         savefigto: str = "GMM_figures") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate residuals for a given GMM.
    
    Parameters:
    -----------
    selected_data : pd.DataFrame
        Data for residual calculation
    periods : list
        List of periods
    domain : str
        'FAS' or 'SA'
    network : str
        Specification type
    coeff_table : pd.DataFrame
        Pre-fitted coefficient table
    alg : str
        Algorithm type
    min_no_event : int
        Minimum records per event
    do_plot : bool
        Whether to generate plots
    savefigto : str
        Directory to save figures
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Between-event residuals, site-to-site residuals, and updated data
    """
    
    # Create tables for site-to-site and between-event residuals
    ds2s_table0 = pd.DataFrame({
        'StationCode': selected_data['StationCode'].unique()
    })
    ds2s_table0['VS30'] = ds2s_table0['StationCode'].map(
        selected_data.drop_duplicates('StationCode').set_index('StationCode')['VS30']
    )
    
    dbe_table0 = pd.DataFrame({
        'EQ_Code': selected_data['EQ_Code'].unique()
    })
    dbe_table0['MAG'] = dbe_table0['EQ_Code'].map(
        selected_data.drop_duplicates('EQ_Code').set_index('EQ_Code')['MAG']
    )
    
    # Determine time unit for plotting
    if domain == "FAS":
        tu = 'Hz'
    else:
        tu = 's'
    
    # Loop through periods
    for t in periods:
        print(f"Processing period: {t}")
        
        # Determine IM name and numeric period
        if t == "PGA":
            IM = 'PGA'
            tu=''
        else:
            IM = f"X{float(t):.3f}"
        
        # Process period
        dbe_table, ds2s_table, dwses_table = split_period(
            t, selected_data.copy(), coeff_table, domain, network, min_no_event
        )
        
        # Update main tables
        update_main_tables(dbe_table, ds2s_table, dwses_table, 
                            dbe_table0, ds2s_table0, selected_data, IM, t)
    
        # Generate plots if requested
        if do_plot:
            generate_plots(selected_data, IM, t, tu, savefigto, network)
    
    return dbe_table0, ds2s_table0, selected_data


def kothaetal2020_epe(period: Union[str, float], M: np.ndarray, R: np.ndarray, 
                      D: np.ndarray, coeff_table: pd.DataFrame, domain: str) -> np.ndarray:
    """
    Kotha et al. (2020) GMM functional form for ESM.
    
    Parameters:
    -----------
    period : str or float
        Period of interest
    M : array-like
        Magnitude values
    R : array-like
        Distance values (RJB)
    D : array-like
        Depth values
    coeff_table : pd.DataFrame
        Coefficient table
    domain : str
        'FAS' or 'SA'
    
    Returns:
    --------
    np.ndarray
        Predicted ground motion values
    """
    # Convert to arrays
    M = np.atleast_1d(M)
    R = np.atleast_1d(R)
    D = np.atleast_1d(D)
    
    # Get coefficients
    period_data = coeff_table[coeff_table['period'] == period].iloc[0]
    
    if 'a' in period_data:
        a = period_data['a']
    else:
        a = period_data.get('e1', period_data['a'])
    
    b1 = period_data['b1']
    Mref = period_data['Mref']
    b2 = period_data['b2']
    Mh = period_data['Mh']
    b3 = period_data['b3']
    
    c1 = period_data['c1']
    c2 = period_data['c2']
    c3 = period_data['c3']
    
    Rref = period_data['Rref']
    h_D10 = period_data['h_D10']
    h_10D20 = period_data['h_10D20']
    h_D20 = period_data['h_20D']
    
    # Depth-dependent h
    h = np.where(D < 10, h_D10,
                np.where(D < 20, h_10D20, h_D20))
    
    # Magnitude scaling
    exprM1 = M - Mh
    exprM2 = exprM1**2
    exprM3 = b1 * exprM1 + b2 * exprM2
    exprM4 = b3 * exprM1
    valueFM = np.where(M <= Mh, exprM3, exprM4)
    
    # Distance scaling
    exprD1 = M - Mref
    exprD2 = c1 + c2 * exprD1
    exprD3 = R**2 + h**2
    exprD4 = np.sqrt(exprD3)
    exprD5 = exprD4 / np.sqrt(Rref**2 + h**2)
    exprD6 = np.log(exprD5)
    exprD7 = (exprD4 - np.sqrt(Rref**2 + h**2)) / 100
    
    if domain == "FAS":
        valueFD = c1 * exprD6 + c3 * exprD7
    else:
        valueFD = exprD2 * exprD6 + c3 * exprD7
    
    # Prediction
    value = np.exp(a + valueFD + valueFM)
    
    return value


def gmpe_allM_YA15_Mh_Mref(period: Union[str, float], M: np.ndarray, Distance: np.ndarray,
                coeff_table: pd.DataFrame) -> np.ndarray:
    """
    GMM functional form (Youngs et al. 1995 style) for KiK-net.
    
    Parameters:
    -----------
    period : str or float
        Period of interest
    M : array-like
        Magnitude values
    Distance : array-like
        Distance values (RJB)
    coeff_table : pd.DataFrame
        Coefficient table
    
    Returns:
    --------
    np.ndarray
        Predicted ground motion values
    """
    # Convert to arrays
    M = np.atleast_1d(M)
    Distance = np.atleast_1d(Distance)
    
    # Get coefficients
    period_data = coeff_table[coeff_table['period'] == period].iloc[0]
    
    a = period_data['a']
    b1 = period_data['b1']
    Mref = period_data['Mref']
    b2 = period_data['b2']
    Mh = period_data['Mh']
    b3 = period_data['b3']
    c1 = period_data['c1']
    Rs = period_data['Rs']
    c2 = period_data['c2']
    c3 = period_data['c3']
    
    # Magnitude scaling
    exprM1 = M - Mref
    exprM2 = b1 * exprM1
    exprM3 = b2 * exprM1
    exprM4 = M - Mh
    exprM5 = b2 * (Mh - Mref) + b3 * exprM4
    
    FM = np.where(M < Mref, exprM2,
                 np.where(M < Mh, exprM3, exprM5))
    
    # Distance scaling
    h = np.exp(2.303 * np.maximum(
        -0.05 + 0.15 * M,
        -1.72 + 0.43 * M
    ))
    
    Rh = np.sqrt(Distance**2 + h**2)
    expr_R1 = c1 * np.log(Rh)
    expr_R2 = c1 * np.log(np.sqrt(Rs**2 + h**2)) + c2 * np.log(Distance / Rs) + c3 * (Distance - Rs)
    
    FD = np.where(Distance < Rs, expr_R1, expr_R2)
    
    # Prediction
    value = np.exp(a + FM + FD)
    
    return value

def plot_selected_data(selected_data: pd.DataFrame, IM: str, domain: str, savefigto: str, network: str):
    # RJB and MW with PREDICTION plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    im = selected_data[IM]

    # Distance PGA plot
    scatter = ax1.scatter(selected_data['RJB'], selected_data[IM],
                            c=selected_data['MAG'], cmap='magma_r', s=30)
    ax1.set_title(f'T = {IM}', fontsize=15)
    ax1.set_xlabel('$R_{JB}$ (km)', fontsize=15)
    ax1.set_ylabel(f'${domain}$ (g)', fontsize=15)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(0.5, 600)
    ax1.grid(True, alpha=0.3)

    m_plot = [3.5, 4.5, 5.5, 6.5]
    cbar1 = plt.colorbar(scatter, ax=ax1, ticks=m_plot)
    cbar1.ax.set_yticklabels([f'M{m}' for m in m_plot])
    
    # Magnitude PGA plot
    d_plot = [5, 10, 25, 50, 100, 200, 400]
    
    scatter2 = ax2.scatter(selected_data['MAG'], selected_data[IM],
                            c=selected_data['RJB'], cmap='viridis', 
                            norm=plt.matplotlib.colors.LogNorm(), s=30)
    ax2.set_title(f'T = {IM}', fontsize=15)
    ax2.set_xlabel('$M_W$', fontsize=15)
    ax2.set_ylabel(f'${domain}$ (g)', fontsize=15)
    ax2.set_xlim(2.5, 8.5)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('$R_{JB}$', fontsize=15)
    
    plt.tight_layout()
    plt.savefig(f'{savefigto}/selected_mag_dist_{network}{domain}.jpeg',
                dpi=300, bbox_inches='tight')
    plt.close()



def read_select_ESM(filename: str, periods: List, 
                    domain: str = 'SA', ASC: bool = True) -> pd.DataFrame:
    """
    Read and process ESM flatfile data for Europe and Mediterranean region.
    
    Parameters:
    -----------
    filename : str
        Path to the ESM flatfile CSV
    periods : list
        List of periods/frequencies of interest
    domain : str
        Domain type ('SA' for Spectral Acceleration or 'FAS' for Fourier Amplitude Spectrum)
    ASC : bool
        If True, filter for active shallow crustal events only
        
    Returns:
    --------
    pd.DataFrame
        Processed flatfile with selected columns and filtering applied
    """
    # Read flatfile
    esm_flatfile = pd.read_csv(filename, sep=';', low_memory=False)
    
    # Create unique address identifier
    esm_flatfile['Address'] = (
        esm_flatfile['event_id'] + '-' + 
        esm_flatfile['network_code'] + '-' + 
        esm_flatfile['station_code'] + '-' + 
        esm_flatfile['location_code'].astype(str)
    )
    
    # Combine magnitude columns (prioritize EMEC_Mw over Mw)
    esm_flatfile['MAG'] = esm_flatfile['EMEC_Mw'].where(
        esm_flatfile['EMEC_Mw'] >= 0, 
        esm_flatfile['Mw']
    )
    # Add MAG_type column
    esm_flatfile['MAG_type'] = np.where(
        esm_flatfile['EMEC_Mw'] >= 0, 
        'EMEC_Mw', 
        'Mw'
    )
     
    # Combine distance columns
    esm_flatfile['RJB'] = esm_flatfile.apply(
        lambda row: row['JB_dist'] if row['JB_dist'] > 0 
        else (row['epi_dist'] if row['MAG'] <= 5.5 else row['JB_dist']),
        axis=1
    )
    # Add R_type column
    esm_flatfile['R_type'] = esm_flatfile.apply(
        lambda row: 'JB_dist' if row['JB_dist'] > 0
        else ('epi_dist' if row['MAG'] <= 5.5 else 'JB_dist'),
        axis=1
    )
    
    # Flag events with minimum recording distance > 80 km
    dist_flag = np.full(len(esm_flatfile), "Near- and far-source", dtype=object)
    for event_id in esm_flatfile['event_id'].unique():
        event_mask = esm_flatfile['event_id'] == event_id
        min_dist = esm_flatfile.loc[event_mask, 'JB_dist'].min()
        if min_dist > 80:
            dist_flag[event_mask] = "Primarily far-source"
    esm_flatfile['Dist_flag'] = dist_flag
    
    # Calculate frequency and time parameters
    esm_flatfile['fHigh0'] = np.sqrt(esm_flatfile['U_lp'] * esm_flatfile['V_lp'])
    esm_flatfile['fLow0'] = np.sqrt(esm_flatfile['U_hp'] * esm_flatfile['V_hp'])
    esm_flatfile['tHigh'] = 1 / (0.8 * esm_flatfile['fLow0'])
    esm_flatfile['fHigh'] = esm_flatfile['fHigh0'] / 0.8
    esm_flatfile['fLow'] = 0.8 * esm_flatfile['fLow0']
    
    # Apply selection criteria
    criteria = (
        esm_flatfile['MAG'].notna() & 
        (esm_flatfile['late_triggered_flag_01'] == 0) & 
        esm_flatfile['RJB'].notna()
    )
    
    # Apply tectonic filtering
    if ASC:
        if 'tectonic_class' in esm_flatfile.columns:
            criteria = criteria & (esm_flatfile['tectonic_class'] == "nonsubduction")
        else:
            print('No tectonic class available; using event depth as criteria instead')
            criteria = criteria & (esm_flatfile['ev_depth_km'] <= 35)
    
    # Filter data
    df_new = esm_flatfile[criteria].copy().reset_index(drop=True)
    
    # Rename standard columns
    column_mapping = {
        'event_id': 'EQ_Code',
        'station_code': 'StationCode',
        'vs30_m_sec': 'VS30',
        'fm_type_code': 'FM',
        'rotD50_pga': 'PGA',
        'rotD50_pgv': 'PGV_rotd50'
    }
    df_new.rename(columns=column_mapping, inplace=True)
    
#    df_new['PGA'] = df_new['rotD50_pga']/(100*g)
    
    # Define columns of interest
    cols_of_interest = [
        'Address', 'EQ_Code', 'StationCode', 'RJB', 'R_type', 'MAG', 'MAG_type', 'VS30',
        'ev_depth_km', 'tHigh', 'fLow', 'fHigh', 'Dist_flag', 'FM', 'PGA'
    ]
    
    # Process domain-specific columns
    if domain == 'FAS':
        # Extract FAS columns
        fas_cols = [col for col in df_new.columns if col.startswith('U_F')]
        
        for col in fas_cols:
            freq_str = col[3:]  # Remove 'U_F' prefix
            freq_val = float(freq_str.replace('_', '.'))
            if freq_val in periods:
                # Calculate geometric mean of FAS 
                u_col = f'U_F{freq_str}'
                v_col = f'V_F{freq_str}'
                col_name = f"X{freq_val:.3f}"
                df_new[col_name] = np.sqrt(df_new[u_col] * df_new[v_col])#/(100*g) # convert from cm/s2 to g
                cols_of_interest.append(col_name)
                
    elif domain == 'SA':
        
        # Define SA columns
        sa_cols = [col for col in df_new.columns if col.startswith('rotD50_T')]
        
        for col in sa_cols:
            period = float(col.replace('rotD50_T', '').replace('_', '.'))
            if period in periods:
                col_name = f"X{period:.3f}"
                df_new[col_name] = df_new[col]#/(100*g) # convert from cm/s2 to g
                cols_of_interest.append(col_name)
    
    # Select final columns
    df_new = df_new[cols_of_interest].copy()
    
    return df_new


def parzen_smoother(y, window_width_hz, sampling_frequency):
    """
    Apply Parzen window smoothing to a signal.
    
    Parameters:
    -----------
    y : array-like
        Input signal
    window_width_hz : float
        Window width in Hz
    sampling_frequency : float
        Sampling frequency in Hz
        
    Returns:
    --------
    numpy.ndarray
        Smoothed signal
    """
    window_width_samples = int((1 / window_width_hz) * sampling_frequency)
    w = windows.parzen(window_width_samples)
    
    half_window = (len(w) - 1) // 2
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y_padded = np.concatenate((firstvals, y, lastvals))
    
    return np.convolve(w[::-1] / w[::-1].sum(), y_padded, mode='valid')
 

def read_select_KiK_Knet(filename: str, periods: List, 
                         domain: str = 'SA', ASC: bool = True) -> pd.DataFrame:
    """
    Read and process Kik-net or Knet flatfile data from Japan.
    
    Parameters:
    -----------
    filename : str
        Path to the KiK-net or Knet flatfile CSV
    periods : list
        List of periods/frequencies of interest
    domain : str
        Domain type ('SA' for Spectral Acceleration or 'FAS' for Fourier Amplitude Spectrum)
    ASC : bool
        If True, use active shallow crustal events only
        
    Returns:
    --------
    pd.DataFrame
        Processed flatfile with selected columns and filtering applied
    """
    # Read flatfile
    kikknet_flatfile = pd.read_csv(filename.replace('SA','META'), sep=',', low_memory=False)
    kikknet_IMs = pd.read_csv(filename, sep=',', low_memory=False)

    # Merge IMs with metadata
    kikknet_flatfile = pd.merge(kikknet_flatfile, kikknet_IMs, on=['Address'], how='left', suffixes=('', '_IM'))

    # Only consider events that are matched with the FNET catalog
    # (To ensure reliable moment magnitude (Mw), depth and RJB distance values)
    kikknet_flatfile = kikknet_flatfile[kikknet_flatfile['Fnet_match'] == True].copy()

    # Rename standard columns
    column_mapping = {'fnet_MT_Magnitude(Mw)': 'MAG', 
                      'fnet_MT_Depth(km)': 'ev_depth_km', 
                      'Focal_mechanism_BA': 'FM'}
    kikknet_flatfile.rename(columns=column_mapping, inplace=True)

    kikknet_flatfile['MAG_type'] = 'fnet_MT_Magnitude(Mw)'
    
    # Combine distance columns
    kikknet_flatfile["RJB"] = np.sqrt(kikknet_flatfile['RJB_0']*kikknet_flatfile['RJB_1'])
    kikknet_flatfile["R_type"] = 'rjb'

    # Calculate frequency and time parameters
    kikknet_flatfile['tHigh'] = 1 / (1.25 * kikknet_flatfile['fc0'])
    kikknet_flatfile['fHigh'] = kikknet_flatfile['fc1'] / 1.25
    kikknet_flatfile['fLow'] = 1.25 * kikknet_flatfile['fc0']
 
    # Flag events that have only been recorded at far distance
    dist_flag = np.full_like(kikknet_flatfile['EQ_Code'], "Near- and far-source", dtype='U20')
    offshore_flag = np.full_like(kikknet_flatfile['EQ_Code'], "Land", dtype='U10')
    for evU in kikknet_flatfile['EQ_Code'].unique():
        dists = kikknet_flatfile[kikknet_flatfile['EQ_Code']==evU].RJB  
        if dists.min() > 100:    
            dist_flag[kikknet_flatfile['EQ_Code']==evU] = "Primarily far-source"
        
        # Remove very off-shore events
        eq_lat = float(kikknet_flatfile[kikknet_flatfile['EQ_Code']==evU]['evLat._Meta'].values[0])
        eq_lon = float(kikknet_flatfile[kikknet_flatfile['EQ_Code']==evU]['evLong._Meta'].values[0])
        if ((eq_lat < 42.2) & (eq_lon>142.2)): 
            offshore_flag[kikknet_flatfile['EQ_Code']==evU] = "Offshore"
        elif ((eq_lat < 33.5) & (eq_lon>136)): 
            offshore_flag[kikknet_flatfile['EQ_Code']==evU] = "Offshore"
        elif ((eq_lat < 33) & (eq_lon>133.5)): 
            offshore_flag[kikknet_flatfile['EQ_Code']==evU] = "Offshore"
        
    kikknet_flatfile["Offshore_flag"] = offshore_flag     
    kikknet_flatfile["Dist_flag"] = dist_flag
    
    # Add Site info
    if 'VS30' not in kikknet_flatfile.columns:
        site_database = pd.read_csv('../Site_Database_of_KNET_and_KiKnet_StrongMotionStations_in_Japan_v1_0_0/Site_Database.csv')
    
        kikknet_flatfile['coordinates'] = list(zip(kikknet_flatfile['StationLong.'], kikknet_flatfile['StationLat.']))
        kikknet_flatfile.coordinates = kikknet_flatfile.coordinates.apply(Point)
        kikknet_flatfile_gpd = gpd.GeoDataFrame(kikknet_flatfile, geometry='coordinates')
        
        site_database['coordinates'] = list(zip(site_database['Longitude'], site_database['Latitude']))
        site_database.coordinates = site_database.coordinates.apply(Point)
        site_database_gpd = gpd.GeoDataFrame(site_database, geometry='coordinates')
        site_database_gpd = site_database_gpd.drop_duplicates(subset=['Site Code'], ignore_index=True)
        
        sjoin0 = gpd.sjoin(kikknet_flatfile_gpd, site_database_gpd, how='left')
        kikknet_flatfile = pd.DataFrame(sjoin0)
    
    # Apply selection criteria
    criteria = (
        (kikknet_flatfile['energy_ratioSignal']>0.8) & 
        (kikknet_flatfile['freq_range']>0.6) & 
        (kikknet_flatfile['energy_ratioNoise']<0.01) & 
        (kikknet_flatfile['MAG']>3.5) & 
        (kikknet_flatfile['RJB']>0) & (kikknet_flatfile['RJB']<600) & 
        (kikknet_flatfile['Dist_flag']=="Near- and far-source")
    )
    
    # Active shallow events filtering
    if ASC:
        criteria = (criteria & (kikknet_flatfile["Offshore_flag"]=="Land") &
                      (kikknet_flatfile['ev_depth_km']<=35))
    
    # Filter data
    df_new = kikknet_flatfile[criteria].copy().reset_index(drop=True)
    df_new['PGA'] = df_new['PGA_rotd50']/g # Convert from m/s2 to g

    # Define columns of interest
    cols_of_interest = [
        'Address', 'EQ_Code', 'StationCode', 'RJB', 'R_type', 'MAG', 'MAG_type', 'VS30',
        'ev_depth_km', 'tHigh', 'fLow', 'fHigh', 'Dist_flag', 'FM', 'PGA',
    ]
    
    # Process domain-specific columns
    if domain == 'FAS':
        # Extract FAS columns
        fas_cols = [col for col in df_new.columns if col.startswith('EW')]
        
        for col in fas_cols:
            freq_str = col[2:]  
            freq_val = float(freq_str.replace('_', '.'))
            if freq_val in periods:
                # Calculate geometric mean of FAS
                ew_col = f'EW{freq_str}'
                ns_col = f'NS{freq_str}'
                few = df_new[ew_col] / g
                fns = df_new[ns_col] / g
                
                col_name = f"X{freq_val:.3f}"
                df_new[col_name] = parzen_smoother(np.sqrt((few**2+fns**2)/2), 0.2, 1)
                cols_of_interest.append(col_name)
                
    elif domain == 'SA':
        # Define SA columns
        sa_cols = [col for col in df_new.columns if col.startswith('RotD50')]
        
        for col in sa_cols:
            period_str = col.replace('RotD50', '').replace('_', '.')
            #try:
            period = float(period_str)
            if period in periods:  
                col_name = f"X{period:.3f}"
                df_new[col_name] = df_new[col]/g
                cols_of_interest.append(col_name)
            #except ValueError:
            #    continue
    
    # Select final columns
    df_new = df_new[cols_of_interest].copy()
    
    return df_new
   
#Plot the residuals with bins and errorbars
def bin_plot(x, y, bins):
    Dpos = []
    Dsd = []
    b0pos = []
    for i, (bi1, bi2) in enumerate(zip(bins[:-1], bins[1:])):
        # First bin includes its left edge so the minimum-valued point
        # (which equals bins[0]) isn't dropped from every bin.
        if i == 0:
            b0int_mask = (x>=bi1) & (x<=bi2)
        else:
            b0int_mask = (x>bi1) & (x<=bi2)
        D = y[b0int_mask].values
        if np.count_nonzero(~np.isnan(D)) == 0:
            Dpos.append(np.nan)
            Dsd.append(np.nan)
        else:
            Dpos.append(np.nanmean(D))
            Dsd.append(np.nanstd(D))
        b0pos.append(np.nanmean([bi1,bi2]))
    return Dpos, Dsd, b0pos

def gmm_eval(residual_data: pd.DataFrame, coeff_table: pd.DataFrame, 
                 periods: List, domain: str, network: str, 
                 do_plot: bool = True, savefigto: str = "GMM_eval") -> pd.DataFrame:
    """
    Evaluate residuals with magnitude, distance and Vs30.
    
    Parameters:
    -----------
    residual_data : pd.DataFrame
        gmm results data
    coeff_table : pd.DataFrame
        Coefficient table for gmm calculations
    periods : list
        List of periods to evaluate
    domain : str
        Domain type ('FAS' or 'SA')
    network : str
        Species/type ('ESM' or other)
    do_plot : bool
        Whether to generate plots
    savefigto : str
        Directory path to save figures
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions for all scenarios
    """
    
    # Generate magnitude, distance, response spectra scaling plots
    M_values = [3.25, 3.5, 3.75, 4, 4.25, 4.5,
                4.6, 4.7, 4.8, 4.9, 5,
                5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6,
                6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7,
                7.1, 7.2, 7.3, 7.4, 7.5]
    
    D_values = [0.5, 1, 2, 3, 4, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 35, 40, 45, 50, 
                60, 70, 80, 90, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600]
    
    # Create all combinations
    scenarios_allM = pd.DataFrame([
        {'periods': p, 'M': m, 'D': d, 'depth': 0.0, 'Pred': 0.0}
        for p in periods
        for m in M_values
        for d in D_values
    ])
    
    # Determine time unit
    if domain == "FAS":
        tu = 'Hz'
    else:
        tu = 's'
     
    # Magnitude binning
    residual_data['Mbin'] = pd.cut(
        residual_data['MAG'],
        bins=[-np.inf, 4.5, 5.5, 6.5, np.inf],
        labels=['M<4.5', '4.5≤M<5.5', '5.5≤M<6.5', '6.5≤M']
    )
    
    # Loop through periods
    for t in periods:
        print(f"Evaluating period: {t}")
        
        # Get predictions based on network type
        if network == "ESM":
            period_mask = scenarios_allM['periods'] == t
            scenarios_allM.loc[period_mask, 'Pred'] = kothaetal2020_epe(
                t,
                scenarios_allM.loc[period_mask, 'M'].values,
                scenarios_allM.loc[period_mask, 'D'].values,
                scenarios_allM.loc[period_mask, 'depth'].values,
                coeff_table,
                domain
            )
            gmm = 'Kotha2020'
        else:
            period_mask = scenarios_allM['periods'] == t
            scenarios_allM.loc[period_mask, 'Pred'] = gmpe_allM_YA15_Mh_Mref(
                t,
                scenarios_allM.loc[period_mask, 'M'].values,
                scenarios_allM.loc[period_mask, 'D'].values,
                coeff_table
            )
            gmm = 'gmpe_allM_YA15_Mh_Mref'
        
        # Determine IM column name
        if t == "PGA":
            IM = 'PGA'
            gmm_col = 'event_and_site_corrected_PGA'
            tu=''
        else:
            IM = f"X{float(t):.3f}"
            gmm_col = f'event_and_site_corrected_{IM}'
        
        # PLOTS
        if do_plot:
            os.makedirs(savefigto, exist_ok=True)

            # Check if gmm column exists
            if gmm_col not in residual_data.columns:
                print(f"Warning: {gmm_col} not found in residual_data. Skipping magnitude/distance plots.")
#                continue
            
            # Magnitude and distance scaling with prediction lines
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Distance scaling with prediction: 7 magnitude bins colored with a
            # colorblind-safe gradient, matching gmpe_dist_scaling() in gmm_functions.R
            m_plot = [3.5, 4.5, 5.5, 6, 6.5, 7, 7.5]
            cbb_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
                'cbb', ['#000000', '#E69F00', '#56B4E9', '#009E73',
                        '#F0E442', '#0072B2', '#D55E00', '#CC79A7'])
            m_norm = plt.matplotlib.colors.Normalize(vmin=min(m_plot), vmax=max(m_plot))

            for m in m_plot:
                cl = cbb_cmap(m_norm(m))
                mask = (residual_data['MAG'] >= m-0.1) & (residual_data['MAG'] <= m+0.1)
                ax1.scatter(residual_data.loc[mask, 'RJB'],
                           residual_data.loc[mask, gmm_col],
                           s=30, alpha=0.5, color=cl)

                pred_mask = (scenarios_allM['M'] == m) & (scenarios_allM['periods'] == t)
                pred_data = scenarios_allM[pred_mask].sort_values('D')
                ax1.plot(pred_data['D'], pred_data['Pred'], 'k-', linewidth=2)
                ax1.plot(pred_data['D'], pred_data['Pred'], linewidth=1.5, color=cl, label=f'M{m}')

            ax1.set_title(f'T = {t}{tu}', fontsize=15)
            ax1.set_xlabel('$R_{JB}$ (km)', fontsize=15)
            ax1.set_ylabel(f'{domain}$[\\delta B_e + \\delta S2S_s]$ (g)', fontsize=15)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_xlim(0.5, 600)
            ax1.legend(loc='lower left')
            ax1.grid(True, alpha=0.3)

            # Magnitude scaling plot
            d_plot = [5, 10, 25, 50, 100, 200, 400]

            rjb_norm = plt.matplotlib.colors.LogNorm(
                vmin=min(residual_data['RJB'].min(), min(d_plot)),
                vmax=max(residual_data['RJB'].max(), max(d_plot)))
            scatter3 = ax2.scatter(residual_data['MAG'], residual_data[gmm_col],
                                  c=residual_data['RJB'], cmap='viridis',
                                  norm=rjb_norm, s=30, alpha=0.5)

            for d in d_plot:
                cl = plt.cm.viridis(rjb_norm(d))
                pred_mask = (scenarios_allM['D'] == d) & (scenarios_allM['periods'] == t)
                pred_data = scenarios_allM[pred_mask].sort_values('M')
                ax2.plot(pred_data['M'], pred_data['Pred'], 'k-', linewidth=2)
                ax2.plot(pred_data['M'], pred_data['Pred'], linewidth=1.5, color=cl, label=f'{d} km')
            
            ax2.set_title(f'T = {t}{tu}', fontsize=15)
            ax2.set_xlabel('$M_W$', fontsize=15)
            ax2.set_ylabel(f'{domain}$[\\delta B_e + \\delta S2S_s]$ (g)', fontsize=15)
            ax2.set_xlim(2.5, 8.5)
            ax2.set_yscale('log')
            ax2.legend(loc='lower right')
            ax2.grid(True, alpha=0.3)
            
            cbar3 = plt.colorbar(scatter3, ax=ax2)
            cbar3.set_label('$R_{JB}$', fontsize=15)
            
            plt.tight_layout()
            plt.savefig(f'{savefigto}/mag_dist_scaling_{t}_{gmm}_{network}.jpeg',
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    # Save scenarios to CSV
    scenarios_allM.to_csv(f'scenario_{network}_{domain}.csv', index=False)
    
    return scenarios_allM


#%%
if __name__ == "__main__":
    # 0) SELECT PARAMETERS
    network = 'ESM' # Can be 'ESM' (Europe) or other/Japan (e.g. KiKnet or Knet)
    domain = 'SA'  # Can be 'FAS' or 'SA'
    
    # Set algorithm
    alg = "lmer"
    savefigto = f'gmm_{network}{domain}{alg}'
    os.makedirs(savefigto, exist_ok=True)

    new_gmm = True
    new_data = True

    # 1) READ AND SELECT DATA based on domain and network
    if network == 'ESM':
        data_file = f'ESM_flatfile_2018/ESM_flatfile_{domain}.csv'
    else:
        #data_file = f'{network}_flatfile_{domain}.csv'
        data_file = '/home/karinlo/GFZ/2026-003_Loviknes/2026-003_Loviknes-et-al_1997_2025_kik_SA.csv'
    
    if not os.path.exists(data_file):
        print(f"ERROR: Data file {data_file} not found!")
        print("Please update the path to your data file.")
        import sys
        sys.exit(1)
    
    # Select regression time periods
    if domain == "FAS":
        periods = [0.550, 0.603, 0.725, 0.851, 1.000, 1.660,
                  2.042, 2.455, 3.091, 3.390, 4.075, 4.572, 5.014,
                  5.130, 6.028, 7.416, 8.131, 8.713, 9.776, 10.004, 
                  11.225, 12.028, 12.888, 13.810, 14.461, 15.142, 
                  16.225, 17.386, 18.630, 19.508, 20.427]
    else:
        periods = ["PGA", 0.010, 0.020, 0.025, 0.030, 0.040, 0.050, 
                  0.060, 0.075, 0.090, 0.100, 0.120, 0.150, 0.170, 
                  0.200, 0.300, 0.400, 0.500, 0.600, 0.750, 1.000, 
                  1.200, 1.500, 1.600, 1.700, 1.800, 2.000, 2.500, 
                  3.000, 4.000, 5.000, 7.500, 10.000]
        periods = ["PGA", 0.100, 1.000]
    
    if network == 'ESM':
        if new_data:
            selected_data = read_select_ESM(data_file, periods, domain)
            selected_data.to_csv(f"selected_data_{network}{domain}.csv", index=False)
        else:
            selected_data = pd.read_csv(f"selected_data_{network}{domain}.csv")
        # Remove potential nonlinear records (PGA < 0.05g)
        if 'PGA' in selected_data.columns:
            lin_data = selected_data[selected_data['PGA'] < 0.05*(100*g)].copy() #cm/s2
            print(f"Linear records (PGA < 0.05g): {len(lin_data)}")
        else:
            lin_data = selected_data.copy()
            print("PGA column not found, using all records")
    else:
        if new_data:
            selected_data = read_select_KiK_Knet(data_file, periods, domain)
            selected_data.to_csv(f"selected_data_{network}{domain}.csv", index=False)
        else:
            selected_data = pd.read_csv(f"selected_data_{network}{domain}.csv")

        # Remove potential nonlinear records (PGA < 0.05g)
        if 'PGA' in selected_data.columns:
            lin_data = selected_data[selected_data['PGA'] < 0.05].copy() # g
            print(f"Linear records (PGA < 0.05g): {len(lin_data)}")
        else:
            lin_data = selected_data.copy()
            print("PGA column not found, using all records")
    
    
    plot_selected_data(selected_data, 'PGA', domain, savefigto, network)

    # 3) RUN REGRESSION
    if new_gmm:
        coeff_table, dbe_table, ds2s_table, residual_data = gmm_new(
            selected_data=lin_data,
            periods=periods,
            domain=domain,
            network=network,
            alg=alg,
            min_no_event=3,
            do_plot=True,
            savefigto=savefigto
        )
        dbe_table.to_csv(f"dbe_table_{network}{domain}{alg}.csv")
        ds2s_table.to_csv(f"ds2s_table_{network}{domain}{alg}.csv")
        residual_data.to_csv(f"residual_data_{network}{domain}{alg}.csv")
        coeff_table.to_csv(f"coeff_table_{network}{domain}{alg}.csv")
    else:
        # Load pre-fitted coefficients
        coeff_table = pd.read_csv(f"coeff_table_{network}{domain}{alg}.csv")
        
        dbe_table, ds2s_table, residual_data = calculate_residuals(
            selected_data=lin_data,
            periods=periods,
            domain=domain,
            network=network,
            coeff_table=coeff_table,
            alg=alg,
            min_no_event=3,
            do_plot=True,
            savefigto=savefigto
        )
        dbe_table.to_csv(f"dbe_table_{network}{domain}.csv")
        ds2s_table.to_csv(f"ds2s_table_{network}{domain}.csv")
        residual_data.to_csv(f"residual_data_{network}{domain}.csv")
        
    # Evaluate GMM and residuals
    scenarios_allM = gmm_eval(
        residual_data=residual_data,
        coeff_table=coeff_table,
        periods=periods,
        domain=domain,
        network=network,
        do_plot=True,
        savefigto=savefigto
    )
    
    print("Processing complete!")
