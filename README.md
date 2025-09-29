# Emerging Markets Bond Analysis

Interactive visualization tool for analyzing sovereign bond yields versus credit ratings across emerging market ETFs.

## Overview

This tool provides comparative analysis of three major emerging market bond ETFs:
- **EMBI** - iShares JP Morgan USD Emerging Markets Bond ETF
- **CEMBI** - iShares Emerging Markets Corporate Bond ETF  
- **GBI** - iShares Emerging Markets Local Currency Bond ETF

The analysis combines:
- Real-time sovereign credit ratings from Trading Economics
- Current bond holdings and yields from iShares ETF data
- Comparative spread analysis vs rating peers

## Features

- 📊 **Interactive dropdown menu** to switch between ETFs
- 📈 **Scatter plot** showing credit rating vs yield to worst
- 📉 **Bar chart** displaying yield spreads vs rating peers (basis points)
- 🔄 **Auto-updates daily** via GitHub Actions
- 💾 **Exports summary CSV files** for each ETF

## Live Demo

View the interactive analysis: [em_bonds_analysis.html](https://boquin.xyz/em_bonds_analysis.html)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the analysis:

```bash
python em_bonds_analysis.py
```

This generates:
- `em_bonds_analysis.html` - Interactive visualization with dropdown
- `EMBI_summary.csv` - Summary statistics for EMBI
- `CEMBI_summary.csv` - Summary statistics for CEMBI
- `GBI_summary.csv` - Summary statistics for GBI

## Automated Updates

The analysis updates automatically every morning at 9 AM UTC via GitHub Actions. 

Manual trigger: Go to Actions tab → "Update EM Bonds Analysis" → "Run workflow"

## Data Sources

- **Credit Ratings**: Trading Economics sovereign ratings
- **Bond Data**: iShares ETF holdings (updated daily)

## Methodology

1. **Scrape** current sovereign credit ratings
2. **Load** ETF bond holdings and yields
3. **Merge** data by country
4. **Calculate** yield spreads vs rating peer averages
5. **Visualize** with interactive Plotly charts

## Interpretation

### Scatter Plot
Shows relationship between credit quality and bond yields. Higher ratings (lower numbers) typically have lower yields.

### Spread Chart
- **Negative spreads** (green): Country yields are lower than peers = trading "expensive"
- **Positive spreads** (red): Country yields are higher than peers = trading "cheap"

Cheap bonds may represent value opportunities or reflect country-specific risks not captured by ratings.

## Requirements

- Python 3.8+
- pandas
- numpy
- requests
- beautifulsoup4
- plotly
- lxml

## Files

```
├── em_bonds_analysis.py          # Main analysis script
├── requirements.txt               # Python dependencies
├── .github/
│   └── workflows/
│       └── update_bonds.yml      # GitHub Action for daily updates
├── em_bonds_analysis.html        # Output: Interactive visualization
├── EMBI_summary.csv              # Output: EMBI statistics
├── CEMBI_summary.csv             # Output: CEMBI statistics
└── GBI_summary.csv               # Output: GBI statistics
```

## License

MIT

## Author

Created for [boquin.xyz](https://boquin.xyz)

## Disclaimer

This tool is for informational and educational purposes only. Not financial advice. Credit ratings and bond yields are subject to change. Always conduct your own research before making investment decisions.
