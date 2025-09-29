import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# ETF Data URLs
CEMBI = 'https://www.ishares.com/us/products/239525/ishares-emerging-markets-corporate-bond-etf/1467271812596.ajax?fileType=csv&fileName=CEMB_holdings&dataType=fund'
EMBI = 'https://www.ishares.com/us/products/239572/ishares-jp-morgan-usd-emerging-markets-bond-etf/1467271812596.ajax?fileType=csv&fileName=EMB_holdings&dataType=fund'
GBI = 'https://www.ishares.com/us/products/239528/ishares-emerging-markets-local-currency-bond-etf/1467271812596.ajax?fileType=csv&fileName=LEMB_holdings&dataType=fund'

# Rating mappings
ratings_dict = {
    "AAA+": 1, "AAA": 2, "AAA-": 3,
    "AA+": 4, "AA": 5, "AA-": 6,
    "A+": 7, "A": 8, "A-": 9,
    "BBB+": 10, "BBB": 11, "BBB-": 12,
    "BB+": 13, "BB": 14, "BB-": 15,
    "B+": 16, "B": 17, "B-": 18,
    "CCC": 19, "CCC+": 20, "CCC-": 21,
    "CC": 22, "C": 23, "SD": 23, "WD": 23, "RD": 23
}

ratings_dict_small = {
    "AAA+": 1, "AAA": 1, "AAA-": 1,
    "AA+": 2, "AA": 2, "AA-": 2,
    "A+": 3, "A": 3, "A-": 3,
    "BBB+": 4, "BBB": 4, "BBB-": 4,
    "BB+": 5, "BB": 5, "BB-": 5,
    "B+": 6, "B": 6, "B-": 6,
    "CCC": 7, "CCC+": 7, "SD": 7, "WD": 7, "RD": 7
}

def scrape_tradingeconomics_ratings():
    """Scrape sovereign credit ratings from Trading Economics"""
    url = "https://tradingeconomics.com/country-list/rating"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    print(f"Fetching credit ratings from Trading Economics...")
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        
        if not tables:
            raise ValueError("No tables found")
        
        df = tables[0]
        df = df.dropna(how='all')
        df.columns = df.columns.str.strip()
        
        print(f"✓ Scraped {len(df)} countries")
        return df
        
    except Exception as e:
        print(f"Error scraping ratings: {e}")
        return pd.DataFrame()

def get_year(x):
    """Extract year from maturity string"""
    x = str(x)
    try: 
        return int(x[-4:])
    except:
        return None

def process_ratings_data(ratings_df):
    """Process ratings data to match expected format"""
    if ratings_df.empty:
        return pd.DataFrame()
    
    processed_df = ratings_df.copy()
    
    # Identify columns
    country_col = None
    rating_col = None
    
    for col in processed_df.columns:
        if any(word in col.lower() for word in ['country', 'name', 'territory']):
            country_col = col
            break
    
    for col in processed_df.columns:
        if any(word in col.lower() for word in ['rating', 'grade']) and 'number' not in col.lower():
            rating_col = col
            break
    
    if not country_col:
        country_col = processed_df.columns[0]
    
    if not rating_col:
        for col in processed_df.columns:
            if col != country_col and processed_df[col].dtype == 'object':
                rating_col = col
                break
    
    if not rating_col:
        return pd.DataFrame()
    
    # Process ratings
    if 'Rating' not in processed_df.columns:
        processed_df['Rating'] = processed_df[rating_col]
    
    processed_df['Rating'] = processed_df['Rating'].str.replace("−", "-", regex=False)
    processed_df['Rating'] = processed_df['Rating'].str.strip()
    
    processed_df["Rating numbers"] = processed_df["Rating"].replace(ratings_dict)
    processed_df["Rating bucket"] = processed_df["Rating"].replace(ratings_dict_small)
    
    if "Outlook" not in processed_df.columns:
        processed_df["Outlook"] = "Stable"
    
    if country_col != processed_df.index.name:
        processed_df = processed_df.set_index(country_col)
    
    # Filter valid ratings
    if processed_df["Rating numbers"].dtype == 'object':
        valid_ratings = processed_df["Rating numbers"].apply(lambda x: isinstance(x, (int, float)))
        processed_df = processed_df[valid_ratings]
    
    print(f"✓ Processed {len(processed_df)} country ratings")
    return processed_df

def load_etf_data(indexchoice):
    """Load and process ETF holdings data"""
    chartnames = {EMBI: "EMBI", CEMBI: "CEMBI", GBI: "GBI"}
    
    print(f"Loading ETF data...")
    df = pd.read_csv(indexchoice, header=9)
    datalist = ["YTM (%)", "Maturity", "Mod. Duration", "Yield to Worst (%)", "Location"]
    df1 = df[datalist]
    df1 = df1.set_index("Location")
    
    df1["year"] = df1["Maturity"].apply(get_year)
    df1 = df1[df1["year"] < 2035]
    df1 = df1[df1["year"] > 2027]
    df1["YTW"] = pd.to_numeric(df1["Yield to Worst (%)"], errors="coerce")
    
    for col in df1.columns:
        if df1[col].dtype == 'object':
            df1[col] = pd.to_numeric(df1[col], errors='ignore')
    
    numeric_cols = df1.select_dtypes(include='number').columns
    df_mean = df1.groupby("Location")[numeric_cols].mean()
    
    print(f"✓ Loaded {len(df_mean)} countries with bond data")
    return df_mean, chartnames.get(indexchoice, "Unknown")

def calculate_log_regression(x_data, y_data):
    """Calculate logarithmic regression line"""
    # Remove any NaN values
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    
    if len(x_clean) < 2:
        return None, None
    
    # Transform x to log space
    x_log = np.log(x_clean).reshape(-1, 1)
    
    # Fit linear regression on log-transformed x
    model = LinearRegression()
    model.fit(x_log, y_clean)
    
    # Generate smooth line for plotting
    x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
    x_range_log = np.log(x_range).reshape(-1, 1)
    y_pred = model.predict(x_range_log)
    
    return x_range, y_pred

def create_analysis_for_etf(etf_choice, ratings_df):
    """Create analysis for a single ETF"""
    etf_mapping = {
        "EMBI": EMBI,
        "CEMBI": CEMBI, 
        "GBI": GBI
    }
    
    indexchoice = etf_mapping.get(etf_choice, EMBI)
    
    # Get ETF data
    try:
        df_mean, etf_name = load_etf_data(indexchoice)
    except Exception as e:
        print(f"Error loading {etf_choice}: {e}")
        return None
    
    # Merge
    new_df = ratings_df.join(df_mean, how='inner')
    new_df = new_df[new_df["YTW"].notnull()]
    
    if len(new_df) == 0:
        print(f"No matching data for {etf_choice}")
        return None
    
    print(f"✓ {etf_name}: {len(new_df)} countries with complete data")
    
    # Prepare visualization data
    df_scat = new_df.reset_index()
    df_scat["Outlook"].fillna("Stable", inplace=True)
    df_scat["Rating numbers"] = df_scat["Rating numbers"].apply(pd.to_numeric, errors="coerce")
    df_scat = df_scat[df_scat["Rating numbers"] < 17]
    
    # Calculate spreads
    new_df["Rating numbers"] = pd.to_numeric(new_df["Rating numbers"], errors="coerce")
    mean_ytw = new_df.groupby("Rating numbers")["YTW"].mean().to_dict()
    new_df["mean_ytw"] = new_df["Rating numbers"].map(mean_ytw)
    new_df["spread_to_mean"] = (new_df["YTW"] - new_df["mean_ytw"]) * 100
    
    new_df_mod = new_df[new_df["Rating numbers"] < 18]
    n_countries = 20
    
    new_table = pd.concat([
        new_df_mod["spread_to_mean"].sort_values().head(n_countries),
        new_df_mod["spread_to_mean"].sort_values().tail(n_countries)
    ])
    
    return {
        'name': etf_name,
        'scatter_data': df_scat,
        'spread_data': new_table.reset_index(),
        'summary': new_df
    }

def create_combined_html_with_dropdown():
    """Create HTML file with dropdown menu for all three ETFs"""
    
    print("=" * 60)
    print("EMERGING MARKETS BOND ANALYSIS - INTERACTIVE HTML")
    print("=" * 60)
    
    # Scrape ratings once
    print("\n1. Scraping credit ratings...")
    ratings_df_raw = scrape_tradingeconomics_ratings()
    
    if ratings_df_raw.empty:
        print("Error: Could not fetch ratings data")
        return
    
    ratings_df = process_ratings_data(ratings_df_raw)
    
    if ratings_df.empty:
        print("Error: Could not process ratings data")
        return
    
    # Process all three ETFs
    print("\n2. Processing ETF data...")
    etfs = ['EMBI', 'CEMBI', 'GBI']
    etf_data = {}
    
    for etf in etfs:
        print(f"\n   Processing {etf}...")
        data = create_analysis_for_etf(etf, ratings_df)
        if data:
            etf_data[etf] = data
    
    if not etf_data:
        print("Error: No ETF data could be processed")
        return
    
    print(f"\n3. Creating interactive HTML visualization...")
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Credit Rating vs Yield to Worst', 'Yield Spread vs Rating Peers (basis points)'),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )
    
    # Color mapping for outlooks
    color_map = {'Positive': '#2471A3', 'Stable': '#ABB2B9', 'Negative': '#C0392B'}
    
    # Add traces for each ETF
    for i, (etf_key, data) in enumerate(etf_data.items()):
        visible = True if i == 0 else False
        
        # Scatter plot
        df_scat = data['scatter_data']
        country_col = df_scat.columns[0]
        
        for outlook in df_scat['Outlook'].unique():
            df_outlook = df_scat[df_scat['Outlook'] == outlook]
            
            fig.add_trace(
                go.Scatter(
                    x=df_outlook["Rating numbers"],
                    y=df_outlook["YTW"],
                    mode='markers+text',
                    name=outlook,
                    text=df_outlook[country_col],
                    textposition='top center',
                    marker=dict(size=12, color=color_map.get(outlook, '#ABB2B9')),
                    visible=visible,
                    legendgroup=etf_key,
                    showlegend=(i == 0)
                ),
                row=1, col=1
            )
        
        # Calculate and add logarithmic regression line
        x_data = df_scat["Rating numbers"].values
        y_data = df_scat["YTW"].values
        x_range, y_pred = calculate_log_regression(x_data, y_data)
        
        if x_range is not None:
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_pred,
                    mode='lines',
                    name='Log Regression',
                    line=dict(color='red', width=2, dash='dash'),
                    visible=visible,
                    legendgroup=etf_key,
                    showlegend=(i == 0)
                ),
                row=1, col=1
            )
        
        # Bar chart - all bars same blue color
        spread_data = data['spread_data']
        
        fig.add_trace(
            go.Bar(
                x=spread_data.iloc[:, 0],
                y=spread_data['spread_to_mean'],
                name='Spread',
                marker=dict(color='#2471A3'),  # Single blue color
                visible=visible,
                legendgroup=etf_key,
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Create dropdown buttons
    buttons = []
    traces_per_etf = {}
    trace_idx = 0
    
    for etf_key, data in etf_data.items():
        n_outlooks = len(data['scatter_data']['Outlook'].unique())
        n_traces = n_outlooks + 1 + 1  # scatter traces + regression line + 1 bar chart
        traces_per_etf[etf_key] = (trace_idx, n_traces)
        trace_idx += n_traces
    
    for etf_key, data in etf_data.items():
        start_idx, n_traces = traces_per_etf[etf_key]
        
        # Create visibility list
        visible_list = [False] * len(fig.data)
        for i in range(start_idx, start_idx + n_traces):
            visible_list[i] = True
        
        buttons.append(
            dict(
                label=data['name'],
                method='update',
                args=[
                    {'visible': visible_list},
                    {'title.text': f"{data['name']}: Credit Rating vs Yield Analysis"}
                ]
            )
        )
    
    # Update layout with dropdown ABOVE the chart
    fig.update_layout(
        title={
            'text': f"{etf_data['EMBI']['name']}: Credit Rating vs Yield Analysis",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=1100,
        showlegend=True,
        plot_bgcolor='#F8F9F9',
        margin=dict(t=150, b=50, l=50, r=50),  # More top margin for dropdown
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.5,
                xanchor="center",
                y=1.08,  # Position above chart
                yanchor="top",
                bgcolor='white',
                bordercolor='gray',
                borderwidth=1
            )
        ],
        annotations=[
            dict(
                text="Select ETF:",
                showarrow=False,
                x=0.5,
                y=1.12,  # Position above dropdown
                xref="paper",
                yref="paper",
                xanchor="center",
                font=dict(size=14, color="black", family="Arial, sans-serif")
            )
        ]
    )
    
    # Update axes
    fig.update_xaxes(
        title_text="Credit Rating",
        tickmode="array",
        tickvals=list(range(6, 21, 2)),
        ticktext=["AA-", "A", "BBB+", "BBB-", "BB", "B+", "B-", "CCC-", "SD"],
        row=1, col=1
    )
    fig.update_yaxes(title_text="Yield to Worst (%)", row=1, col=1)
    
    fig.update_xaxes(title_text="Country", tickangle=-45, row=2, col=1)
    fig.update_yaxes(title_text="Spread (bp)", row=2, col=1)
    
    # Save as HTML with include_plotlyjs set to 'cdn' for better compatibility
    output_file = "em_bonds_analysis.html"
    fig.write_html(
        output_file,
        include_plotlyjs='cdn',
        config={'displayModeBar': True, 'displaylogo': False}
    )
    
    print(f"\n✓ Success! Interactive HTML saved to: {output_file}")
    print(f"✓ ETFs included: {', '.join([d['name'] for d in etf_data.values()])}")
    print(f"\nTo view locally: Just double-click {output_file} or open in browser")
    
    # Also save summary statistics
    print("\n4. Saving summary statistics...")
    for etf_key, data in etf_data.items():
        summary_df = data['summary']
        filename = f"{data['name']}_summary.csv"
        summary_df.to_csv(filename)
        print(f"   ✓ {filename}")
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Upload '{output_file}' to your website")
    print("The dropdown menu lets users switch between ETFs")

if __name__ == "__main__":
    create_combined_html_with_dropdown()
