#🏙️ NYC Airbnb Neighbourhood Value Analysis
📌 Project Overview

This project analyzes Airbnb listings in New York City to identify undervalued 🟢 and overpriced 🔴 neighbourhoods.
The goal is to understand where people get better value for money 💰 using data.

🎯 Objective

To find neighbourhoods that offer:

✅ Better value (lower price + higher demand)

❌ Poor value (high price without matching demand)

📊 Dataset

📍 Source: NYC Airbnb Open Data (Kaggle)
📦 Size: ~48,000 listings

🔑 Key Data Used

💵 Price per night

📅 Availability (days per year)

⭐ Number of reviews (demand indicator)

🏘️ Neighbourhood & borough

🧹 Data Cleaning

✔ Removed duplicate listings
✔ Removed invalid prices
✔ Handled missing values

🛠 Feature Engineering

Created neighbourhood-level metrics:

📈 Average price

📆 Average availability

🔥 Reviews per listing (popularity)

📐 Value Score Formula

To compare neighbourhoods fairly, a Value Score was created:

Value Score = (Availability × Popularity) / Price

🔹 High score → 🟢 Good value
🔹 Low score → 🔴 Potentially overpriced

📈 Visualizations

The project includes:

🗺️ Heatmap (borough vs value score)

📉 Price vs availability scatter plot

🏆 Top 10 undervalued neighbourhoods bar chart

🖥️ Interactive Streamlit dashboard

🔍 Key Findings

🟢 Undervalued Areas

Brooklyn

Queens

🔴 Overpriced Areas

Central Manhattan

📌 High prices do not always mean high demand.

🧰 Tools Used

🐍 Python

📊 Pandas

📉 Matplotlib & Seaborn

🚀 Streamlit

✅ Conclusion

This project shows how data can be used to:

📊 Compare neighbourhoods objectively

🧠 Create meaningful performance metrics

💼 Support data-driven decisions
