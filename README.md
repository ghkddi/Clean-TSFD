# Repository of the paper: Benchmark Datasets Under the Microscope: Diagnosing Data Quality and Temporal Splits for Multivariate Time Series Forecasting
Submited to: ECMLPKDD 2026 Applied Data Science Track

## Content

### Electricity Load Diagrams
 * [x] Code to analyze different version of the dataset
   * Raw: 370 clients /// 15-minute resolution
   * ECL: 321 clients /// 1-hour resolution 
 * [x] Code to plot client with unusual patterns and create new dataset version
   * PELD_1H_3Y_308: 308 clients /// 1-hour resolution
 * [x] Cycle-inclusive splits dataloader
 * [x] CSV file:
   * PELD_1H_3Y_308.csv
 * [x] Various plots of the data
 * [x] Markdown files with experiment results

### Local Climatological Data
 * [x] Code to analyze version from Informer paper and identify inconsistencies
   * Weather: 12 indicators /// 1-hour resolution
 * [x] Code to correct inconsistencies
   * LCDWf_1H_4Y_USUNK: 11 indicators + WetBulbCelsius + 6 error identifiers
   * LCDWi_1H_4Y_USUNK: 11 indicators + WetBulbCelsiusInt + 6 error identifiers
   * LCDWr_1H_4Y_USUNK: 11 indicators + RealWetBulbCelsius + 6 error identifiers
 * [x] Cycle-inclusive splits dataloader
 * [x] CSV files:
   * LCDWf_1H_4Y_USUNK.csv
   * LCDWi_1H_4Y_USUNK.csv
   * LCDWr_1H_4Y_USUNK.csv
 * [x] Various plots of the data
 * [x] Markdown files with experiment results

### Max-Planck Institute for Biogeochemistry 
 * [x] Code to analyze different version of the dataset
   * Raw: 21 indicators /// 10-minute resolution
   * Weather: 21 indicators /// 10-minute resolution
 * [x] Code to identify and correct inconsistencies
   * MPIW_10T_1Y_R: 21 indicators + 5 identifiers /// 10-minute resolution
 * [x] Code to generate 4-year period dataset
   * MPIW_10T_4Y_R: 21 indicators + 6 identifiers /// 10-minute resolution
 * [x] Code to create 4-year hourly version
   * MPIW_1H_4Y_R: 21 indicators + 6 identifiers /// 1-hour resolution
 * [x] Cycle-inclusive splits dataloader
 * [x] CSV files:
   * MPIW_10T_1Y_R.csv
   * MPIW_10T_4Y_R.csv
   * MPIW_1H_4Y_R.csv
 * [x] Various plots of the data 
 * [x] Markdown files with experiment results
