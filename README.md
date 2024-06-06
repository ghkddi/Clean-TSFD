# Repository of the paper: Ensuring Fair Comparisons in Time Series Forecasting: Addressing Quality Issues in Three Benchmark Datasets
Submited to: NeurISP 2024 Dataset and Benchmark Track

## Content

### Electricity Load Diagrams
 * [ ] Code to analyze different version of the dataset
   * Raw: 370 clients /// 15-minute resolution
   * ECL: 321 clients /// 1-hour resolution 
 * [ ] Code to plot client with unusual patterns and create new dataset version
   * PELD_1H_3Y_308: 308 clients /// 1-hour resolution
 * [x] Cycle-inclusive splits dataloader
 * [ ] CSV file:
   * PELD_1H_3Y_308.csv
 * [ ] Various plots of the data
 * [x] Markdown files with experiment results

### Local Climatological Data
 * [ ] Code to analyze version from Informer paper
   * Weather: 12 indicators /// 1-hour resolution
 * [ ] Code to identify and correct inconsistencies
   * LCDW_1H_4Y_USUNK: 12 indicators + 2 revised indicators + 6 identifiers /// 1-hour resolution
   * LCDWf_1H_4Y_USUNK: 11 indicators + WetBulbCelsius + 6 indicators
   * LCDWi_1H_4Y_USUNK: 11 indicators + WetBulbCelsiusInt + 6 indicators
   * LCDWr_1H_4Y_USUNK: 11 indicators + RealWetBulbCelsius + 6 indicators
 * [x] Cycle-inclusive splits dataloader
 * [ ] CSV files:
   * LCDWf_1H_4Y_USUNK.csv
   * LCDWi_1H_4Y_USUNK.csv
   * LCDWr_1H_4Y_USUNK.csv
 * [ ] Various plots of the data
 * [ ] Markdown files with experiment results

### Max-Planck Institute for Biogeochemistry 
 * [ ] Code to analyze different version of the dataset
   * Raw: 21 indicators /// 10-minute resolution
   * Weather: 21 indicators /// 10-minute resolution
 * [ ] Code to identify and correct inconsistencies
   * MPIW_10T_1Y_R: 21 indicators + 3 identifiers /// 10-minute resolution
 * [ ] Code to create hourly version as well as generate 4-year period dataset
   * MPIW_10T_4Y_R: 21 indicators + 6 identifiers /// 10-minute resolution
   * MPIW_1H_4Y_R: 21 indicators + 6 identifiers /// 1-hour resolution
 * [x] Cycle-inclusive splits dataloader
 * [ ] CSV files:
   * MPIW_10T_1Y_R.csv
   * MPIW_10T_4Y_R.csv
   * MPIW_1H_4Y_R.csv
   * Various plots of the data 
 * [ ] Markdown files with experiment results