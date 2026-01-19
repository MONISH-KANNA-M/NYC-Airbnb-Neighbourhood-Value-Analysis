import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="NYC Airbnb Value Dashboard", layout="wide", initial_sidebar_state="expanded")

# Apply premium custom styling
st.markdown("""
    <style>
        /* Main Background */
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 12px;
            color: white;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            text-align: center;
            font-weight: bold;
        }
        
        /* Insight Cards */
        .insight-card {
            background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
            padding: 25px;
            border-radius: 15px;
            border-left: 8px solid #667eea;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            margin: 15px 0;
        }
        
        .insight-card h3 {
            color: #1f1f2e;
            font-size: 1.4em;
            margin-bottom: 10px;
        }
        
        /* Undervalued Card */
        .undervalued-card {
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            padding: 25px;
            border-radius: 15px;
            border-left: 8px solid #2ecc71;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #1f1f2e;
            font-weight: bold;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Data frame styling */
        .dataframe {
            border-radius: 10px !important;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        
        /* Section divider */
        .section-divider {
            margin: 40px 0;
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #667eea, transparent);
        }
        
        /* Ranking badge */
        .ranking-badge {
            display: inline-block;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
            margin: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("AB_NYC_2019.csv")
    df = df[df["price"] > 0].drop_duplicates()
    return df

df = load_data()

# Feature Engineering - Enhanced Value Score
neigh_df = df.groupby(
    ["neighbourhood_group", "neighbourhood"]
).agg(
    avg_price=("price", "mean"),
    min_price=("price", "min"),
    max_price=("price", "max"),
    avg_availability=("availability_365", "mean"),
    total_reviews=("number_of_reviews", "sum"),
    listings=("id", "count"),
    room_type_diversity=("room_type", "nunique"),
    avg_minimum_nights=("minimum_nights", "mean")
).reset_index()

neigh_df["reviews_per_listing"] = (
    neigh_df["total_reviews"] / neigh_df["listings"]
).fillna(0)

# Enhanced Value Score Formula
neigh_df["value_score"] = (
    (neigh_df["avg_availability"] / 365) *  # Availability ratio
    (neigh_df["reviews_per_listing"] / neigh_df["reviews_per_listing"].max()) *  # Review popularity
    (1000 / neigh_df["avg_price"])  # Inverse price weight
) * 100

# Calculate percentile for ranking
neigh_df["value_percentile"] = neigh_df["value_score"].rank(pct=True) * 100

# Dashboard Title
st.title("🏙️ NYC Airbnb Neighbourhood Value Dashboard")
st.markdown(
    "📈 Data-driven evaluation of **undervalued vs overpriced neighbourhoods** "
    "using availability, demand, and pricing analytics."
)

# Sidebar Navigation
st.sidebar.title("📍 Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        "📊 Dataset Summary",
        "🧮 Value Score Computation",
        "🏆 Neighbourhood Rankings",
        "🗺️ Interactive Borough Explorer",
        "💬 Smart Insights"
    ]
)

# Sidebar Filters
st.sidebar.markdown("---")
st.sidebar.title("🔍 Filters")
borough = st.sidebar.selectbox(
    "Select Borough",
    ["All"] + sorted(neigh_df["neighbourhood_group"].unique())
)

if borough != "All":
    filtered = neigh_df[neigh_df["neighbourhood_group"] == borough].copy()
else:
    filtered = neigh_df.copy()

# ==================== PAGE 1: DATASET SUMMARY ====================
if page == "📊 Dataset Summary":
    st.header("📊 Dataset Overview & Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="font-size: 2.5em; margin: 0;">📍</h3>
        <p style="font-size: 1.2em; margin: 10px 0 0 0;">{len(df):,}</p>
        <p style="font-size: 0.9em; margin: 5px 0 0 0;">Total Listings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="font-size: 2.5em; margin: 0;">🏘️</h3>
        <p style="font-size: 1.2em; margin: 10px 0 0 0;">{df["neighbourhood"].nunique()}</p>
        <p style="font-size: 0.9em; margin: 5px 0 0 0;">Neighbourhoods</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="font-size: 2.5em; margin: 0;">🏙️</h3>
        <p style="font-size: 1.2em; margin: 10px 0 0 0;">{df["neighbourhood_group"].nunique()}</p>
        <p style="font-size: 0.9em; margin: 5px 0 0 0;">Boroughs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="font-size: 2.5em; margin: 0;">💰</h3>
        <p style="font-size: 1.2em; margin: 10px 0 0 0;">${df['price'].mean():.0f}</p>
        <p style="font-size: 0.9em; margin: 5px 0 0 0;">Avg Nightly Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Dataset Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Price Statistics")
        fig, ax = plt.subplots(figsize=(10, 5))
        price_data = [df['price'].min(), df['price'].quantile(0.25), 
                     df['price'].median(), df['price'].quantile(0.75), df['price'].max()]
        labels = ['Min', 'Q1', 'Median', 'Q3', 'Max']
        colors = ['#ff6b6b', '#ffa94d', '#51cf66', '#4d96ff', '#667eea']
        bars = ax.bar(labels, price_data, color=colors, edgecolor='black', linewidth=2)
        ax.set_ylabel("Price ($)", fontsize=12, fontweight='bold')
        ax.set_title("Price Distribution", fontsize=14, fontweight='bold')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${int(height)}', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("📅 Key Metrics")
        metrics_data = {
            "🔢 Std Dev Price": f"${df['price'].std():.2f}",
            "📈 IQR": f"${df['price'].quantile(0.75) - df['price'].quantile(0.25):.2f}",
            "⭐ Avg Availability": f"{df['availability_365'].mean():.0f} days",
            "👥 Avg Reviews": f"{(df['number_of_reviews'].sum() / len(df)):.1f}/listing",
            "🏠 Most Common Room": df['room_type'].mode()[0],
            "📊 Total Reviews": f"{df['number_of_reviews'].sum():,}"
        }
        for key, value in metrics_data.items():
            st.markdown(f"**{key}:** `{value}`")
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Borough Breakdown with visualization
    st.subheader("🏙️ Borough Deep Dive")
    borough_stats = df.groupby("neighbourhood_group").agg({
        "id": "count",
        "price": ["mean", "min", "max"],
        "availability_365": "mean",
        "number_of_reviews": "sum"
    }).round(2)
    borough_stats.columns = ["Listings", "Avg Price", "Min Price", "Max Price", "Avg Availability", "Total Reviews"]
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.dataframe(borough_stats, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        borough_colors = ['#667eea', '#764ba2', '#f093fb', '#4158d0', '#c471ed']
        ax.pie(borough_stats['Listings'], labels=borough_stats.index, autopct='%1.1f%%',
              colors=borough_colors, explode=[0.05]*len(borough_stats), startangle=90)
        ax.set_title("Listings Distribution", fontsize=12, fontweight='bold')
        st.pyplot(fig)

# ==================== PAGE 2: VALUE SCORE COMPUTATION ====================
elif page == "🧮 Value Score Computation":
    st.header("🧮 Value Score Formula & Methodology")
    
    st.markdown("""
    ### How We Calculate Value Score
    
    The **Value Score** is a composite metric that identifies undervalued neighbourhoods by combining:
    
    **Formula:**
    ```
    Value Score = (Availability Ratio × Popularity Score × Price Efficiency) × 100
    ```
    
    Where:
    - **Availability Ratio** = Average Days Available / 365
    - **Popularity Score** = Reviews per Listing / Maximum Reviews per Listing
    - **Price Efficiency** = 1000 / Average Price
    
    ✅ **Higher Score** = Better value (high availability + high demand + low price)
    ❌ **Lower Score** = Premium pricing (may be overvalued)
    """)
    
    st.markdown("---")
    
    # Interactive Demo
    st.subheader("🎯 Interactive Value Score Demo")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        demo_avail = st.slider("Avg Availability (days)", 0, 365, 180)
    with col2:
        demo_reviews = st.slider("Reviews per Listing", 0, 100, 50)
    with col3:
        demo_price = st.slider("Avg Nightly Price ($)", 50, 500, 150)
    
    # Calculate demo score
    demo_score = (
        (demo_avail / 365) *
        (demo_reviews / neigh_df["reviews_per_listing"].max()) *
        (1000 / demo_price)
    ) * 100
    
    st.metric("Calculated Value Score", f"{demo_score:.2f}")
    
    # Show real examples
    st.markdown("---")
    st.subheader("📈 Real Examples from Data")
    top_examples = filtered.nlargest(5, "value_score")[
        ["neighbourhood", "avg_price", "avg_availability", "reviews_per_listing", "value_score"]
    ].reset_index(drop=True)
    top_examples.columns = ["Neighbourhood", "Avg Price ($)", "Availability (days)", "Reviews/Listing", "Value Score"]
    st.dataframe(top_examples, use_container_width=True)

# ==================== PAGE 3: NEIGHBOURHOOD RANKINGS ====================
elif page == "🏆 Neighbourhood Rankings":
    st.header("🏆 Neighbourhood Rankings & Analysis")
    
    # KPIs with custom styling
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="font-size: 2em; margin: 0;">💵</h3>
        <p style="font-size: 1.3em; margin: 10px 0 0 0;">${filtered['avg_price'].mean():.0f}</p>
        <p style="font-size: 0.85em; margin: 5px 0 0 0;">Avg Price</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="font-size: 2em; margin: 0;">📅</h3>
        <p style="font-size: 1.3em; margin: 10px 0 0 0;">{filtered['avg_availability'].mean():.0f}</p>
        <p style="font-size: 0.85em; margin: 5px 0 0 0;">Avg Availability</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="font-size: 2em; margin: 0;">⭐</h3>
        <p style="font-size: 1.3em; margin: 10px 0 0 0;">{filtered['value_score'].mean():.2f}</p>
        <p style="font-size: 0.85em; margin: 5px 0 0 0;">Avg Value Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="font-size: 2em; margin: 0;">📍</h3>
        <p style="font-size: 1.3em; margin: 10px 0 0 0;">{filtered['listings'].sum():.0f}</p>
        <p style="font-size: 0.85em; margin: 5px 0 0 0;">Total Listings</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Complete Value Score Rankings")
        ranking_table = filtered.sort_values("value_score", ascending=False)[
            ["neighbourhood", "listings", "avg_price", "avg_availability", "reviews_per_listing", "value_score"]
        ].reset_index(drop=True)
        ranking_table.columns = ["Neighbourhood", "Listings", "Avg Price ($)", "Availability (days)", "Reviews/Listing", "Value Score"]
        ranking_table.index = ranking_table.index + 1
        
        st.dataframe(ranking_table, use_container_width=True, height=400)
    
    with col2:
        st.subheader("📈 Score Distribution")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(filtered["value_score"], bins=20, color='#667eea', edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Value Score", fontweight='bold')
        ax.set_ylabel("Count", fontweight='bold')
        ax.set_title("Distribution", fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Top Undervalued with enhanced design
    st.subheader("🏆 Top 10 Most Undervalued Neighbourhoods")
    top10 = filtered.nlargest(10, "value_score")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 7))
        y_pos = np.arange(len(top10))
        colors_gradient = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top10)))
        bars = ax.barh(y_pos, top10["value_score"].values, color=colors_gradient, edgecolor='black', linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top10["neighbourhood"].values, fontweight='bold')
        ax.set_xlabel("Value Score", fontweight='bold', fontsize=12)
        ax.set_title("🏆 Top 10 Undervalued Neighbourhoods", fontweight='bold', fontsize=14)
        ax.invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}', ha='left', va='center', fontweight='bold')
        
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("💎 Detailed Metrics")
        detailed = top10[["neighbourhood", "avg_price", "listings", "value_percentile"]].copy()
        detailed.columns = ["Neighbourhood", "Avg Price", "Listings", "Percentile"]
        detailed["Avg Price"] = detailed["Avg Price"].apply(lambda x: f"${x:.0f}")
        detailed["Percentile"] = detailed["Percentile"].apply(lambda x: f"{x:.1f}%")
        detailed.index = [f"🥇" if i == 0 else f"🥈" if i == 1 else f"🥉" if i == 2 else f"{i+1}." 
                         for i in range(len(detailed))]
        st.dataframe(detailed, use_container_width=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Enhanced scatter plot
    st.subheader("💎 Price vs Value Analysis")
    fig, ax = plt.subplots(figsize=(14, 7))
    scatter = ax.scatter(
        filtered["avg_price"],
        filtered["value_score"],
        s=filtered["listings"]*3,
        c=filtered["reviews_per_listing"],
        cmap='RdYlGn',
        alpha=0.6,
        edgecolors='black',
        linewidth=1.5
    )
    ax.set_xlabel("Average Price ($)", fontweight='bold', fontsize=12)
    ax.set_ylabel("Value Score", fontweight='bold', fontsize=12)
    ax.set_title("Price vs Value Score (Bubble size = Listings, Color = Reviews/Listing)", 
                fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Reviews per Listing", fontweight='bold')
    st.pyplot(fig)

# ==================== PAGE 4: INTERACTIVE BOROUGH EXPLORER ====================
elif page == "🗺️ Interactive Borough Explorer":
    st.header("🗺️ Interactive Borough Insights Explorer")
    
    # Borough Selection
    selected_borough = st.selectbox(
        "📍 Select a Borough to Explore",
        sorted(neigh_df["neighbourhood_group"].unique()),
        key="borough_selector"
    )
    
    borough_data = neigh_df[neigh_df["neighbourhood_group"] == selected_borough]
    
    # Borough KPIs with custom styling
    st.markdown(f"## 🏙️ {selected_borough} - Neighbourhood Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="font-size: 2em; margin: 0;">🏘️</h3>
        <p style="font-size: 1.3em; margin: 10px 0 0 0;">{len(borough_data)}</p>
        <p style="font-size: 0.85em; margin: 5px 0 0 0;">Neighbourhoods</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="font-size: 2em; margin: 0;">📍</h3>
        <p style="font-size: 1.3em; margin: 10px 0 0 0;">{int(borough_data['listings'].sum())}</p>
        <p style="font-size: 0.85em; margin: 5px 0 0 0;">Total Listings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="font-size: 2em; margin: 0;">💰</h3>
        <p style="font-size: 1.3em; margin: 10px 0 0 0;">${borough_data['avg_price'].mean():.0f}</p>
        <p style="font-size: 0.85em; margin: 5px 0 0 0;">Avg Price</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="font-size: 2em; margin: 0;">⭐</h3>
        <p style="font-size: 1.3em; margin: 10px 0 0 0;">{borough_data['value_score'].mean():.2f}</p>
        <p style="font-size: 0.85em; margin: 5px 0 0 0;">Value Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌟 Top Neighbourhoods by Value")
        top_neigh = borough_data.nlargest(8, "value_score")[
            ["neighbourhood", "avg_price", "value_score", "listings"]
        ].reset_index(drop=True)
        top_neigh.index = top_neigh.index + 1
        top_neigh.columns = ["Neighbourhood", "Avg Price ($)", "Value Score", "Listings"]
        st.dataframe(top_neigh, use_container_width=True)
    
    with col2:
        st.subheader("💰 Price Range by Listing Volume")
        fig, ax = plt.subplots(figsize=(10, 6))
        top_by_listings = borough_data.nlargest(8, "listings")
        colors = plt.cm.Spectral(np.linspace(0, 1, len(top_by_listings)))
        bars = ax.barh(top_by_listings["neighbourhood"], top_by_listings["avg_price"], 
                      color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xlabel("Average Price ($)", fontweight='bold')
        ax.set_title(f"{selected_borough} - Top 8 by Volume", fontweight='bold', fontsize=12)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 5, bar.get_y() + bar.get_height()/2, 
                   f'${int(width)}', ha='left', va='center', fontweight='bold')
        
        st.pyplot(fig)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Heatmap View
    st.subheader("🔥 Metrics Performance Heatmap")
    heatmap_data = borough_data.nlargest(10, "listings")[
        ["neighbourhood", "avg_price", "avg_availability", "reviews_per_listing", "value_score"]
    ].set_index("neighbourhood")
    
    # Normalize for heatmap
    heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(heatmap_normalized.T, annot=True, fmt='.2f', cmap='RdYlGn', 
                cbar_kws={'label': 'Normalized Value (0-1)'}, ax=ax, 
                linewidths=2, linecolor='white', cbar=True)
    ax.set_title(f"📊 Metrics Heatmap - {selected_borough} Top Neighbourhoods", 
                fontweight='bold', fontsize=14)
    ax.set_ylabel("Metrics", fontweight='bold')
    st.pyplot(fig)
    
    st.markdown("")
    
    # Additional Insights
    st.subheader("💡 Borough Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        best_value = borough_data.loc[borough_data["value_score"].idxmax()]
        st.markdown(f"""
        <div class="insight-card">
        
        ### 🏆 Best Value Neighbourhood
        
        **{best_value['neighbourhood']}**
        
        - Value Score: **{best_value['value_score']:.2f}**
        - Price: ${best_value['avg_price']:.2f}
        - Listings: {best_value['listings']:.0f}
        - Availability: {best_value['avg_availability']:.0f} days
        
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        most_expensive = borough_data.loc[borough_data["avg_price"].idxmax()]
        st.markdown(f"""
        <div class="insight-card">
        
        ### 💎 Premium Neighbourhood
        
        **{most_expensive['neighbourhood']}**
        
        - Price: **${most_expensive['avg_price']:.2f}**
        - Value Score: {most_expensive['value_score']:.2f}
        - Listings: {most_expensive['listings']:.0f}
        - Reviews/Listing: {most_expensive['reviews_per_listing']:.1f}
        
        </div>
        """, unsafe_allow_html=True)

# ==================== PAGE 5: SMART INSIGHTS ====================
else:  # page == "💬 Smart Insights"
    st.header("💬 Smart Insights & AI-Powered Recommendations")
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    st.subheader("🎯 Real-Time Market Analysis")
    
    # Calculate insights
    overall_best = filtered.loc[filtered["value_score"].idxmax()]
    overall_worst = filtered.loc[filtered["value_score"].idxmin()]
    price_range = filtered["avg_price"].max() - filtered["avg_price"].min()
    
    # Insight 1: Top Undervalued
    st.markdown(f"""
    <div class="undervalued-card">
    
    ### 🔝 #1 MOST UNDERVALUED NEIGHBOURHOOD
    
    **{overall_best['neighbourhood']}** • {overall_best['neighbourhood_group']}
    
    **Value Score:** <span style="font-size: 1.5em; color: #2ecc71; font-weight: bold;">{overall_best['value_score']:.2f}</span> 
    (Top {100-overall_best['value_percentile']:.1f}%)
    
    | Metric | Value |
    |--------|-------|
    | 💰 Nightly Rate | ${overall_best['avg_price']:.2f} |
    | 📅 Availability | {overall_best['avg_availability']:.0f} days/year |
    | 👥 Reviews/Listing | {overall_best['reviews_per_listing']:.1f} |
    | 🏠 Listings | {overall_best['listings']:.0f} |
    
    **💡 Key Insight:** This neighbourhood offers exceptional value! High guest availability combined with strong demand and competitive pricing makes it ideal for investors.
    
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Insight 2: Price Analysis
    high_price_neigh = filtered.loc[filtered["avg_price"].idxmax()]
    low_price_neigh = filtered.loc[filtered["avg_price"].idxmin()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="insight-card">
        
        ### 💎 Premium Market
        
        **Most Expensive:** {high_price_neigh['neighbourhood']}
        
        💰 **${high_price_neigh['avg_price']:.2f}** per night
        
        📊 Value Score: {high_price_neigh['value_score']:.2f}
        
        📅 Availability: {high_price_neigh['avg_availability']:.0f} days
        
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="insight-card">
        
        ### 🏷️ Budget-Friendly
        
        **Most Affordable:** {low_price_neigh['neighbourhood']}
        
        💰 **${low_price_neigh['avg_price']:.2f}** per night
        
        📊 Value Score: {low_price_neigh['value_score']:.2f}
        
        📅 Availability: {low_price_neigh['avg_availability']:.0f} days
        
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Insight 3: Market Trends
    high_demand = filtered.nlargest(3, "reviews_per_listing")
    
    st.markdown(f"""
    <div class="insight-card">
    
    ### 📈 HIGH DEMAND NEIGHBOURHOODS
    
    Top performers by guest interest and booking frequency:
    
    🥇 **{high_demand.iloc[0]['neighbourhood']}** 
    - {high_demand.iloc[0]['reviews_per_listing']:.1f} reviews/listing
    - ${high_demand.iloc[0]['avg_price']:.2f}/night
    - Value Score: {high_demand.iloc[0]['value_score']:.2f}
    
    🥈 **{high_demand.iloc[1]['neighbourhood']}** 
    - {high_demand.iloc[1]['reviews_per_listing']:.1f} reviews/listing
    - ${high_demand.iloc[1]['avg_price']:.2f}/night
    - Value Score: {high_demand.iloc[1]['value_score']:.2f}
    
    🥉 **{high_demand.iloc[2]['neighbourhood']}** 
    - {high_demand.iloc[2]['reviews_per_listing']:.1f} reviews/listing
    - ${high_demand.iloc[2]['avg_price']:.2f}/night
    - Value Score: {high_demand.iloc[2]['value_score']:.2f}
    
    💡 **Insight:** High review counts indicate strong guest satisfaction and consistent booking frequency.
    
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Recommendations
    st.subheader("🎬 Strategic Recommendations & Action Items")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 👨‍💼 FOR INVESTORS & HOSTS
        
        ✅ **Priority Neighbourhoods**
        - Focus on high value score areas for ROI
        - Balance between availability and pricing
        - Monitor guest reviews as demand indicator
        
        ✅ **Risk Management**
        - Diversify across multiple boroughs
        - Avoid low-demand areas without strong value
        - Consider seasonal patterns
        
        ✅ **Revenue Optimization**
        - Competitive pricing in high-demand areas
        - Premium pricing justified by location
        - Dynamic pricing based on availability
        """)
    
    with col2:
        st.markdown("""
        #### 🔬 FOR ANALYSTS & RESEARCHERS
        
        📊 **Analysis Focus Areas**
        - Seasonal availability patterns
        - Room type preferences by neighbourhood
        - Price correlation with amenities
        
        🔍 **Deep Dives**
        - Minimum night requirements impact
        - Long-term price trends
        - Cross-borough comparisons
        
        📈 **Predictions**
        - Emerging neighbourhood identification
        - Market saturation indicators
        - Guest satisfaction metrics
        """)
    
    st.markdown("")
    
    # Performance Dashboard
    st.subheader("📊 Market Performance Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_value = filtered["value_score"].mean()
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="font-size: 1.8em; margin: 0;">⭐</h3>
        <p style="font-size: 1.1em; margin: 10px 0 0 0;">{avg_value:.2f}</p>
        <p style="font-size: 0.85em; margin: 5px 0 0 0;">Avg Value Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_value_count = len(filtered[filtered["value_score"] > filtered["value_score"].quantile(0.75)])
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="font-size: 1.8em; margin: 0;">🌟</h3>
        <p style="font-size: 1.1em; margin: 10px 0 0 0;">{high_value_count}</p>
        <p style="font-size: 0.85em; margin: 5px 0 0 0;">Premium Value Areas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        price_var = (filtered["avg_price"].std() / filtered["avg_price"].mean()) * 100
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="font-size: 1.8em; margin: 0;">📈</h3>
        <p style="font-size: 1.1em; margin: 10px 0 0 0;">{price_var:.1f}%</p>
        <p style="font-size: 0.85em; margin: 5px 0 0 0;">Price Variation</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Data Export Section
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.subheader("📥 Export Analysis Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = filtered.sort_values("value_score", ascending=False).to_csv(index=False)
        st.download_button(
            label="📊 Download Rankings (CSV)",
            data=csv,
            file_name="neighbourhood_rankings.csv",
            mime="text/csv"
        )
    
    with col2:
        summary_csv = filtered.groupby("neighbourhood_group").agg({
            "listings": "sum",
            "avg_price": "mean",
            "value_score": "mean"
        }).to_csv()
        st.download_button(
            label="📈 Download Summary (CSV)",
            data=summary_csv,
            file_name="borough_summary.csv",
            mime="text/csv"
        )
    
    with col3:
        st.markdown("💡 **Tip:** Export data for further analysis in Excel or Python")
