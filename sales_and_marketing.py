import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from plotly.subplots import make_subplots
plt.style.use('ggplot')

# Set Streamlit page config
st.set_page_config(
    page_title="Enhanced Marketing Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #E2E8F0, #CBD5E1, #E2E8F0);
        padding: 1rem;
        border-radius: 10px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
        text-align: center;
        margin: 0.8rem 0.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-top: 4px solid #3B82F6;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 1rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
        background: linear-gradient(90deg, #3B82F6, #1E40AF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .tab-subheader {
        font-size: 1.5rem;
        font-weight: 600;
        color: #334155;
        margin: 1rem 0;
    }
    .small-text {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F1F5F9;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #BFDBFE !important;
        color: #1E40AF;
    }
    .chart-container {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #EFF6FF, #F8FAFC);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 10px rgba(59, 130, 246, 0.1);
        margin: 0.8rem 0;
        border-left: 4px solid #3B82F6;
        transition: all 0.3s ease;
    }
    .kpi-card:hover {
        box-shadow: 0 6px 14px rgba(59, 130, 246, 0.15);
        transform: translateY(-2px);
    }
    .footer {
        text-align: center;
        padding: 1rem;
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 2rem;
        border-top: 1px solid #E2E8F0;
    }
    /* Column gap for multi-column layouts */
    .st-emotion-cache-ocqkz7 {
        gap: 1rem !important;
    }
    .st-emotion-cache-1r6slb0 {
        gap: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Create a sidebar with company logo/dashboard title
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(180deg, #1E3A8A, #2563EB); border-radius: 10px; margin-bottom: 2rem;">
    <h2 style="color: white; margin: 0;">Marketing Analytics</h2>
    <p style="color: #BFDBFE; margin: 0;">Performance Dashboard</p>
</div>
""", unsafe_allow_html=True)

# Load and clean data
@st.cache_data
def load_data():
    sales_data = pd.read_csv('sales_data_sample.csv', encoding='latin1')
    sales_data = sales_data.drop(columns=['ADDRESSLINE2', 'STATE', 'YEAR_ID', 'MONTH_ID', 'TERRITORY'])
    sales_data = sales_data.dropna()
    sales_data['ORDERDATE'] = pd.to_datetime(sales_data['ORDERDATE'])
    
    # Simulate CRM & campaign attributes
    np.random.seed(42)
    campaigns = ['Email Blast', 'Holiday Promo', 'SEO Push', 'Social Buzz', 'Referral Drive']
    channels = ['Email', 'Online Ads', 'SEO', 'Social Media', 'Word of Mouth']
    costs = [1500, 2500, 1800, 3000, 1000]
    campaign_probs = [0.10, 0.31, 0.15, 0.37, 0.07]
    sales_data['CAMPAIGN'] = np.random.choice(campaigns, len(sales_data), p=campaign_probs)
    sales_data['CHANNEL'] = sales_data['CAMPAIGN'].map(dict(zip(campaigns, channels)))
    sales_data['COST'] = sales_data['CAMPAIGN'].map(dict(zip(campaigns, costs)))
    sales_data['customer_type'] = np.where(np.random.rand(len(sales_data)) > 0.7, 'New', 'Returning')
    sales_data['LEADS'] = sales_data.groupby('CAMPAIGN')['CUSTOMERNAME'].transform('nunique') + np.random.randint(20, 200, len(sales_data))
    
    # Extract month and year from ORDERDATE
    sales_data['MONTH_ID'] = sales_data['ORDERDATE'].dt.month
    sales_data['YEAR_ID'] = sales_data['ORDERDATE'].dt.year
    sales_data['MONTH_NAME'] = sales_data['ORDERDATE'].dt.strftime('%b')
    sales_data['MONTH_YEAR'] = sales_data['ORDERDATE'].dt.strftime('%b %Y')
    
    return sales_data

sales_data = load_data()

# Sidebar filters
st.sidebar.markdown("## 🔍 Filters")

# Date range filter
min_date = sales_data['ORDERDATE'].min().date()
max_date = sales_data['ORDERDATE'].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (sales_data['ORDERDATE'].dt.date >= start_date) & (sales_data['ORDERDATE'].dt.date <= end_date)
    filtered_data = sales_data[mask]
else:
    filtered_data = sales_data

# Campaign filter
selected_campaign = st.sidebar.selectbox(
    "Campaign",
    ['All'] + list(sales_data['CAMPAIGN'].unique()),
    index=0
)

# Channel filter
selected_channel = st.sidebar.selectbox(
    "Marketing Channel",
    ['All'] + list(sales_data['CHANNEL'].unique()),
    index=0
)

# Apply filters
if selected_campaign != 'All':
    filtered_data = filtered_data[filtered_data['CAMPAIGN'] == selected_campaign]
if selected_channel != 'All':
    filtered_data = filtered_data[filtered_data['CHANNEL'] == selected_channel]

# Product line filter
selected_product_line = st.sidebar.selectbox(
    "Product Line",
    ['All'] + list(sales_data['PRODUCTLINE'].unique()),
    index=0
)

if selected_product_line != 'All':
    filtered_data = filtered_data[filtered_data['PRODUCTLINE'] == selected_product_line]

# Customer type filter
selected_customer_type = st.sidebar.selectbox(
    "Customer Type",
    ['All', 'New', 'Returning'],
    index=0
)

if selected_customer_type != 'All':
    filtered_data = filtered_data[filtered_data['customer_type'] == selected_customer_type]

# Main dashboard title
st.markdown("<h1 class='main-header'>Sales and Marketing Dashboard</h1>", unsafe_allow_html=True)

# Filter summary
filter_summary = f"📅 {start_date} to {end_date}"
if selected_campaign != 'All':
    filter_summary += f" | 🎯 Campaign: {selected_campaign}"
if selected_channel != 'All':
    filter_summary += f" | 📣 Channel: {selected_channel}"
if selected_product_line != 'All':
    filter_summary += f" | 📦 Product: {selected_product_line}"
if selected_customer_type != 'All':
    filter_summary += f" | 👤 Customer: {selected_customer_type}"

st.markdown(f"<p style='text-align: center; color: #64748B;'>{filter_summary}</p>", unsafe_allow_html=True)

# Create tabs
tabs = st.tabs(["📊 Overview", "📈 Trends", "🔍 Correlations", "🔮 Predictive Analysis", "📑 Campaign Details"])

# OVERVIEW TAB
with tabs[0]:
    # Key Performance Indicators in a grid layout
    st.markdown("<h2 class='tab-subheader'>Key Performance Indicators</h2>", unsafe_allow_html=True)
    
    total_sales = filtered_data['SALES'].sum()
    total_orders = filtered_data['ORDERNUMBER'].nunique()
    total_customers = filtered_data['CUSTOMERNAME'].nunique()
    avg_order_value = total_sales / total_orders if total_orders > 0 else 0
    
    # Calculate period-over-period changes
    previous_period_data = sales_data[
        (sales_data['ORDERDATE'].dt.date >= (start_date - pd.Timedelta(days=(end_date-start_date).days))) &
        (sales_data['ORDERDATE'].dt.date < start_date)
    ]
    prev_sales = previous_period_data['SALES'].sum()
    sales_change_pct = ((total_sales - prev_sales) / prev_sales * 100) if prev_sales > 0 else 0
    
    prev_orders = previous_period_data['ORDERNUMBER'].nunique()
    orders_change_pct = ((total_orders - prev_orders) / prev_orders * 100) if prev_orders > 0 else 0
    
    prev_customers = previous_period_data['CUSTOMERNAME'].nunique()
    customers_change_pct = ((total_customers - prev_customers) / prev_customers * 100) if prev_customers > 0 else 0
    
    # Primary KPIs row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card kpi-card">
            <p class="metric-label">Total Sales</p>
            <p class="metric-value">${:,.2f}</p>
            
        </div>
        """.format(total_sales, sales_change_pct), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card kpi-card">
            <p class="metric-label">Total Orders</p>
            <p class="metric-value">{:,}</p>
           
        </div>
        """.format(total_orders, orders_change_pct), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card kpi-card">
            <p class="metric-label">Unique Customers</p>
            <p class="metric-value">{:,}</p>
            
        </div>
        """.format(total_customers, customers_change_pct), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card kpi-card">
            <p class="metric-label">Avg. Order Value</p>
            <p class="metric-value">${:,.2f}</p>
            
        </div>
        """.format(avg_order_value), unsafe_allow_html=True)
    
    # Secondary KPIs row
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate additional KPIs
    conversion_rate = (filtered_data['ORDERNUMBER'].nunique() / filtered_data['LEADS'].mean()) * 100 if filtered_data['LEADS'].mean() > 0 else 0
    total_campaign_cost = filtered_data['COST'].sum()
    roi = ((total_sales - total_campaign_cost) / total_campaign_cost) * 100 if total_campaign_cost > 0 else 0
    new_customers = filtered_data[filtered_data['customer_type'] == 'New']['CUSTOMERNAME'].nunique()
    
    with col1:
        st.markdown("""
        <div class="metric-card kpi-card" style="border-left-color: #10B981;">
            <p class="metric-label">Conversion Rate</p>
            <p class="metric-value">{:.1f}%</p>
            <p class="small-text">Leads to sales</p>
        </div>
        """.format(conversion_rate), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card kpi-card" style="border-left-color: #10B981;">
            <p class="metric-label">Marketing ROI</p>
            <p class="metric-value">{:.1f}%</p>
            <p class="small-text">Return on investment</p>
        </div>
        """.format(roi), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card kpi-card" style="border-left-color: #10B981;">
            <p class="metric-label">New Customers</p>
            <p class="metric-value">{:,}</p>
            <p class="small-text">{:.1f}% of total</p>
        </div>
        """.format(new_customers, (new_customers/total_customers*100) if total_customers > 0 else 0), unsafe_allow_html=True)
    
    with col4:
        cost_per_acquisition = total_campaign_cost / new_customers if new_customers > 0 else 0
        st.markdown("""
        <div class="metric-card kpi-card" style="border-left-color: #10B981;">
            <p class="metric-label">Cost per Acquisition</p>
            <p class="metric-value">${:,.2f}</p>
            <p class="small-text">Marketing cost per new customer</p>
        </div>
        """.format(cost_per_acquisition), unsafe_allow_html=True)
    
    # Visual analytics section
    st.markdown("<h2 class='tab-subheader'>Visual Analytics</h2>", unsafe_allow_html=True)
    
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Marketing Funnel
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("🧭 Marketing Funnel")
        stages = ['Leads', 'Opportunities', 'Quotes Sent', 'Sales Closed']
        values = [
            filtered_data['LEADS'].mean() * 5,  # Scale for better visualization
            filtered_data['LEADS'].mean() * 2,
            filtered_data['ORDERNUMBER'].nunique() * 1.5,
            filtered_data['ORDERNUMBER'].nunique()
        ]
        funnel_fig = go.Figure(go.Funnel(
            y=stages,
            x=values,
            textinfo="value+percent initial",
            marker=dict(color=["#4285F4", "#34A853", "#FBBC05", "#EA4335"]),
            textfont=dict(size=14)
        ))
        funnel_fig.update_layout(
            title={"text": "Lead to Sale Conversion", "y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
            margin=dict(l=20, r=20, t=60, b=20),
            height=350
        )
        st.plotly_chart(funnel_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Donut Chart
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("🌟 Campaign Performance")
        campaign_sales = filtered_data.groupby('CAMPAIGN')['SALES'].sum().reset_index()
        donut_fig = px.pie(
            campaign_sales,
            values='SALES',
            names='CAMPAIGN',
            hole=0.6,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        donut_fig.update_traces(
            textposition='outside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Sales: $%{value:,.2f}<br>%{percent}'
        )
        donut_fig.update_layout(
            title={"text": "Sales by Campaign", "y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=60, b=60),
            height=350
        )
        st.plotly_chart(donut_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Product Line Bar Chart
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("📦 Product Line Performance")
        productline_sales = filtered_data.groupby('PRODUCTLINE')['SALES'].sum().sort_values(ascending=False).reset_index()
        bar_fig = px.bar(
            productline_sales,
            x='PRODUCTLINE',
            y='SALES',
            text_auto='.2s',
            color='SALES',
            color_continuous_scale='Viridis',
            title="Sales by Product Line"
        )
        bar_fig.update_traces(
            texttemplate='$%{text}',
            textposition='inside',
            hovertemplate='<b>%{x}</b><br>Sales: $%{y:,.2f}'
        )
        bar_fig.update_layout(
            xaxis_title="Product Line",
            yaxis_title="Sales ($)",
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=60, b=60),
            height=350
        )
        st.plotly_chart(bar_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        # Customer Type Acquisition
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("👤 Customer Acquisition")
        customer_data = filtered_data.groupby('customer_type')['CUSTOMERNAME'].nunique().reset_index()
        customer_data.columns = ['Type', 'Count']
        
        fig = px.pie(
            customer_data,
            values='Count',
            names='Type',
            color='Type',
            color_discrete_map={'New': '#3B82F6', 'Returning': '#10B981'},
            title="New vs Returning Customers"
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Customers: %{value:,}<br>%{percent}'
        )
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=60, b=60),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Campaign Performance Metrics Table
    st.markdown("<h2 class='tab-subheader'>Campaign Performance Metrics</h2>", unsafe_allow_html=True)
    
    campaign_summary = filtered_data.groupby('CAMPAIGN').agg(
        Total_Sales=('SALES', 'sum'),
        Orders=('ORDERNUMBER', 'nunique'),
        Customers=('CUSTOMERNAME', 'nunique'),
        Cost=('COST', 'first'),
        Leads=('LEADS', 'mean')
    )
    campaign_summary['Conversions'] = campaign_summary['Orders']
    campaign_summary['Conversion_Rate'] = (campaign_summary['Conversions'] / campaign_summary['Leads']) * 100
    campaign_summary['ROI'] = ((campaign_summary['Total_Sales'] - campaign_summary['Cost']) / campaign_summary['Cost']) * 100
    campaign_summary['Cost_per_Order'] = campaign_summary['Cost'] / campaign_summary['Orders']
    campaign_summary['Avg_Order_Value'] = campaign_summary['Total_Sales'] / campaign_summary['Orders']
    
    # Style the dataframe
    styled_summary = campaign_summary.style.format({
        "Total_Sales": "${:,.2f}",
        "Cost": "${:,.2f}",
        "ROI": "{:,.1f}%",
        "Cost_per_Order": "${:,.2f}",
        "Conversion_Rate": "{:,.1f}%",
        "Avg_Order_Value": "${:,.2f}"
    }).background_gradient(subset=['ROI'], cmap='RdYlGn').background_gradient(subset=['Conversion_Rate'], cmap='Blues')
    
    st.dataframe(styled_summary, use_container_width=True)

# TRENDS TAB
with tabs[1]:
    st.markdown("<h2 class='tab-subheader'>Sales & Performance Trends</h2>", unsafe_allow_html=True)
    
    # Monthly Sales Trend
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.subheader("📈 Monthly Sales Trend")
    
    # Group sales by date
    monthly_trend = filtered_data.groupby(filtered_data['ORDERDATE'].dt.to_period("M")).agg({
        'SALES': 'sum',
        'ORDERNUMBER': 'nunique',
        'COST': 'sum'
    }).reset_index()
    monthly_trend['ORDERDATE'] = monthly_trend['ORDERDATE'].dt.to_timestamp()
    monthly_trend['Profit'] = monthly_trend['SALES'] - monthly_trend['COST']
    
    # Create the line chart
    fig = px.line(
        monthly_trend,
        x='ORDERDATE',
        y=['SALES', 'Profit'],
        title='Sales and Profit Trend Over Time',
        labels={'value': 'Amount ($)', 'ORDERDATE': 'Date', 'variable': 'Metric'},
        color_discrete_map={'SALES': '#3B82F6', 'Profit': '#10B981'}
    )
    
    # Customize the layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Amount ($)',
        legend_title='Metric',
        xaxis=dict(
            tickformat="%b %Y",
            tickangle=45,
            showgrid=True
        ),
        yaxis=dict(
            tickprefix='$',
            showgrid=True
        ),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=60),
        height=500
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Sales by Channel and Campaign
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("📣 Sales by Marketing Channel")
        
        channel_sales = filtered_data.groupby(['CHANNEL', 'MONTH_NAME', 'MONTH_ID'])['SALES'].sum().reset_index()
        channel_sales = channel_sales.sort_values('MONTH_ID')
        
        fig = px.line(
            channel_sales,
            x='MONTH_NAME',
            y='SALES',
            color='CHANNEL',
            title='Monthly Sales by Channel',
            markers=True,
            category_orders={"MONTH_NAME": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]}
        )
        
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Sales ($)',
            yaxis=dict(tickprefix='$'),
            legend_title='Channel',
            margin=dict(l=40, r=40, t=60, b=40),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("🎯 Campaign Performance Over Time")
        
        campaign_time = filtered_data.groupby(['CAMPAIGN', 'MONTH_NAME', 'MONTH_ID'])['SALES'].sum().reset_index()
        campaign_time = campaign_time.sort_values('MONTH_ID')
        
        fig = px.bar(
            campaign_time,
            x='MONTH_NAME',
            y='SALES',
            color='CAMPAIGN',
            title='Monthly Sales by Campaign',
            category_orders={"MONTH_NAME": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]}
        )
        
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Sales ($)',
            yaxis=dict(tickprefix='$'),
            legend_title='Campaign',
            margin=dict(l=40, r=40, t=60, b=40),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Monthly Sales Heatmap
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.subheader("🗓️ Monthly Sales Heatmap")
    
    # Create the pivot table for the heatmap
    heatmap_data = filtered_data.pivot_table(values='SALES', index='MONTH_ID', columns='YEAR_ID', aggfunc='sum')
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        colorscale='YlGnBu',
        colorbar=dict(title='Sales ($)', tickprefix='$', tickformat=',.0f'),
        hovertemplate='Year: %{x}<br>Month: %{y}<br>Sales: $%{z:,.2f}<extra></extra>',
    ))
    
    # Customize the layout
    fig.update_layout(
        title='Monthly Sales Performance by Year',
        xaxis_title='Year',
        yaxis_title='Month',
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )
    
    # Display the heatmap
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Customer Acquisition Over Time
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.subheader("👥 Customer Acquisition Over Time")
    
    acq = filtered_data.groupby([filtered_data['ORDERDATE'].dt.to_period("M"), 'customer_type'])['CUSTOMERNAME'].nunique().unstack().fillna(0)
    acq.index = acq.index.to_timestamp()
    acq = acq.reset_index()
    acq['Month'] = acq['ORDERDATE'].dt.strftime('%b %Y')
    
    # Transform data for stacked bar chart
    acq_melted = pd.melt(
        acq,
        id_vars=['ORDERDATE', 'Month'],
        value_vars=['New', 'Returning'],
        var_name='Customer Type',
        value_name='Count'
    )
    
    fig = px.bar(
        acq_melted,
        x='Month',
        y='Count',
        color='Customer Type',
        title='New vs Returning Customers by Month',
        barmode='stack',
        color_discrete_map={'New': '#3B82F6', 'Returning': '#10B981'}
    )
    
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Number of Customers',
        legend_title='Customer Type',
        margin=dict(l=40, r=40, t=60, b=60),
        xaxis=dict(tickangle=45),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# CORRELATIONS TAB
with tabs[2]:
    st.markdown("<h2 class='tab-subheader'>Sales Metric Correlations</h2>", unsafe_allow_html=True)
    
    # First heatmap - Sales Metrics Correlation
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.subheader("📊 Sales Metrics Correlation")
    
    # Prepare the correlation data
    corr_data = filtered_data[['QUANTITYORDERED', 'PRICEEACH', 'SALES', 'MSRP']]
    corr = corr_data.corr()
    
    # Create heatmap using plotly
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        aspect="auto",
        title="Sales Metrics Correlation Matrix"
    )
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        margin=dict(l=40, r=40, t=60, b=40),
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Second row - two more heatmaps
    col1, col2 = st.columns(2)
    
    with col1:
        # Second heatmap - Campaign Performance Correlation
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("🎯 Campaign Performance Correlation")
        
        # Prepare campaign data
        campaign_data = filtered_data.groupby('CAMPAIGN').agg({
            'SALES': 'sum',
            'COST': 'mean',
            'LEADS': 'mean',
            'QUANTITYORDERED': 'sum'
        }).reset_index()
        
        # Add derived metrics
        campaign_data['ROI'] = (campaign_data['SALES'] - campaign_data['COST']) / campaign_data['COST'] * 100
        campaign_data['CONVERSION'] = campaign_data['SALES'] / campaign_data['LEADS']
        
        # Create correlation matrix
        camp_corr = campaign_data[['SALES', 'COST', 'LEADS', 'QUANTITYORDERED', 'ROI', 'CONVERSION']].corr()
        
        # Create heatmap
        fig = px.imshow(
            camp_corr,
            text_auto=True,
            color_continuous_scale='Viridis',
            aspect="auto",
            title="Campaign Metrics Correlation"
        )
        
        fig.update_layout(
            xaxis_title="",
            yaxis_title="",
            margin=dict(l=20, r=20, t=60, b=20),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Third heatmap - Customer Behavior Correlation
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("👥 Customer Behavior Correlation")
        
        # Prepare customer data
        customer_data = filtered_data.groupby('CUSTOMERNAME').agg({
            'SALES': 'sum',
            'QUANTITYORDERED': 'sum',
            'ORDERNUMBER': pd.Series.nunique,
            'PRICEEACH': 'mean'
        }).reset_index()
        
        customer_data['AVG_ORDER_VALUE'] = customer_data['SALES'] / customer_data['ORDERNUMBER']
        customer_data['AVG_QUANTITY'] = customer_data['QUANTITYORDERED'] / customer_data['ORDERNUMBER']
        
        # Create correlation matrix
        cust_corr = customer_data[['SALES', 'QUANTITYORDERED', 'ORDERNUMBER', 'PRICEEACH', 'AVG_ORDER_VALUE', 'AVG_QUANTITY']].corr()
        
        # Create heatmap
        fig = px.imshow(
            cust_corr,
            text_auto=True,
            color_continuous_scale='Teal',
            aspect="auto",
            title="Customer Behavior Correlation"
        )
        
        fig.update_layout(
            xaxis_title="",
            yaxis_title="",
            margin=dict(l=20, r=20, t=60, b=20),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    

# PREDICTIVE ANALYSIS TAB
with tabs[3]:
    st.markdown("<h2 class='tab-subheader'>Predictive Analytics</h2>", unsafe_allow_html=True)
    
    # Linear Regression Section
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.subheader("📐 Linear Regression: Predicting Sales")
    
    # Prepare features
    features = ['QUANTITYORDERED', 'PRICEEACH', 'MSRP', 'PRODUCTLINE', 'DEALSIZE', 'STATUS', 'QTR_ID']
    sales_data_clean = filtered_data.dropna(subset=features + ['SALES'])
    sales_encoded = pd.get_dummies(sales_data_clean[features], drop_first=True)
    y = sales_data_clean['SALES']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(sales_encoded, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Display metrics in cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">Mean Squared Error</p>
            <p class="metric-value">{:.2f}</p>
            <p class="small-text">Lower is better</p>
        </div>
        """.format(mse), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">R-squared Score</p>
            <p class="metric-value">{:.2f}</p>
            <p class="small-text">Higher is better (1.0 is perfect)</p>
        </div>
        """.format(r2), unsafe_allow_html=True)
    
    with col3:
        rmse = np.sqrt(mse)
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">Root Mean Squared Error</p>
            <p class="metric-value">{:.2f}</p>
            <p class="small-text">Lower is better</p>
        </div>
        """.format(rmse), unsafe_allow_html=True)
    
    # Plot Actual vs Predicted Sales
    fig = px.scatter(
        x=y_test,
        y=y_pred,
        labels={'x': 'Actual Sales', 'y': 'Predicted Sales'},
        title="Actual vs Predicted Sales"
    )
    
    # Add identity line (perfect predictions)
    fig.add_trace(
        go.Scatter(
            x=[min(y_test), max(y_test)],
            y=[min(y_test), max(y_test)],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        )
    )
    
    fig.update_layout(
        xaxis_title="Actual Sales",
        yaxis_title="Predicted Sales",
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': sales_encoded.columns,
        'Importance': np.abs(model.coef_)
    }).sort_values('Importance', ascending=False)
    
    top_features = feature_importance.head(10)
    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Top 10 Feature Importance"
    )
    
    fig.update_layout(
        xaxis_title="Importance (Absolute Coefficient Value)",
        yaxis_title="Feature",
        margin=dict(l=40, r=40, t=60, b=40),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Sales Forecast Section
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.subheader("🔮 Sales Forecast")
    
    # Create a simple time-based forecast
    monthly_sales = filtered_data.groupby(filtered_data['ORDERDATE'].dt.to_period("M"))['SALES'].sum()
    monthly_sales.index = monthly_sales.index.to_timestamp()
    
    # Get the date range
    dates = monthly_sales.index
    values = monthly_sales.values
    
    # Create a numeric index for regression
    X = np.arange(len(dates)).reshape(-1, 1)
    y = values
    
    # Fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Forecast next 3 months
    last_date = dates[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=3, freq='M')
    forecast_indices = np.arange(len(dates), len(dates) + 3).reshape(-1, 1)
    forecast_values = model.predict(forecast_indices)
    
    # Create forecast plot
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='blue')
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    # Add confidence interval (simple approach)
    std_err = np.std(y - model.predict(X))
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values + 2*std_err,
        mode='lines',
        name='Upper 95% CI',
        line=dict(color='rgba(255, 0, 0, 0.2)'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values - 2*std_err,
        mode='lines',
        name='Lower 95% CI',
        line=dict(color='rgba(255, 0, 0, 0.2)'),
        fill='tonexty',
        showlegend=False
    ))
    
    fig.update_layout(
        title="Sales Forecast for the Next 3 Months",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        yaxis=dict(tickprefix=''),
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# CAMPAIGN DETAILS TAB
with tabs[4]:
    st.markdown("<h2 class='tab-subheader'>Campaign Performance Details</h2>", unsafe_allow_html=True)
    
    # Select campaign for detailed analysis
    detailed_campaign = st.selectbox(
        "Select Campaign for Detailed Analysis",
        options=sales_data['CAMPAIGN'].unique(),
        key="campaign_detail_selector"
    )
    
    campaign_data = filtered_data[filtered_data['CAMPAIGN'] == detailed_campaign]
    
    # Campaign KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    camp_sales = campaign_data['SALES'].sum()
    camp_orders = campaign_data['ORDERNUMBER'].nunique()
    camp_customers = campaign_data['CUSTOMERNAME'].nunique()
    camp_cost = campaign_data['COST'].iloc[0] if not campaign_data.empty else 0
    camp_roi = ((camp_sales - camp_cost) / camp_cost * 100) if camp_cost > 0 else 0
    camp_leads = campaign_data['LEADS'].mean()
    camp_conversion = (camp_orders / camp_leads * 100) if camp_leads > 0 else 0
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">Total Sales</p>
            <p class="metric-value">${:,.2f}</p>
        </div>
        """.format(camp_sales), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">Total Orders</p>
            <p class="metric-value">{:,}</p>
        </div>
        """.format(camp_orders), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">Campaign ROI</p>
            <p class="metric-value">{:.1f}%</p>
        </div>
        """.format(camp_roi), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">Conversion Rate</p>
            <p class="metric-value">{:.1f}%</p>
        </div>
        """.format(camp_conversion), unsafe_allow_html=True)
    
    # Campaign Performance Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Campaign Performance by Product Line
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("📦 Campaign Performance by Product Line")
        
        product_sales = campaign_data.groupby('PRODUCTLINE')['SALES'].sum().sort_values(ascending=False).reset_index()
        
        fig = px.bar(
            product_sales,
            x='PRODUCTLINE',
            y='SALES',
            color='SALES',
            text_auto=True,
            title=f"{detailed_campaign} - Sales by Product Line"
        )
        
        fig.update_traces(
            texttemplate='$%{text:.2s}',
            textposition='outside'
        )
        
        fig.update_layout(
            xaxis_title="Product Line",
            yaxis_title="Sales ($)",
            yaxis=dict(tickprefix=''),
            coloraxis_showscale=False,
            margin=dict(l=40, r=40, t=60, b=60),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Campaign Performance by Deal Size
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("🏷️ Campaign Performance by Deal Size")
        
        deal_sales = campaign_data.groupby('DEALSIZE')['SALES'].sum().reset_index()
        
        fig = px.pie(
            deal_sales,
            values='SALES',
            names='DEALSIZE',
            title=f"{detailed_campaign} - Sales by Deal Size",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        
        fig.update_layout(
            margin=dict(l=40, r=40, t=60, b=40),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Campaign Time Performance
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.subheader("📅 Campaign Performance Over Time")
    
    # Monthly performance
    monthly_perf = campaign_data.groupby(campaign_data['ORDERDATE'].dt.to_period("M")).agg({
        'SALES': 'sum',
        'ORDERNUMBER': 'nunique',
        'CUSTOMERNAME': 'nunique'
    }).reset_index()
    
    monthly_perf['ORDERDATE'] = monthly_perf['ORDERDATE'].dt.to_timestamp()
    monthly_perf['Month'] = monthly_perf['ORDERDATE'].dt.strftime('%b %Y')
    
    # Create multi-line chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add Sales line
    fig.add_trace(
        go.Scatter(
            x=monthly_perf['Month'],
            y=monthly_perf['SALES'],
            name='Sales',
            line=dict(color='blue', width=3)
        ),
        secondary_y=False
    )
    
    # Add Orders line
    fig.add_trace(
        go.Scatter(
            x=monthly_perf['Month'],
            y=monthly_perf['ORDERNUMBER'],
            name='Orders',
            line=dict(color='red', width=2, dash='dot')
        ),
        secondary_y=True
    )
    
    # Add Customers line
    fig.add_trace(
        go.Scatter(
            x=monthly_perf['Month'],
            y=monthly_perf['CUSTOMERNAME'],
            name='Customers',
            line=dict(color='green', width=2, dash='dash')
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=f"{detailed_campaign} - Monthly Performance Metrics",
        xaxis_title="Month",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Sales ($)", secondary_y=False, tickprefix='')
    fig.update_yaxes(title_text="Count", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    
    # Campaign Data Table
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.subheader("📋 Campaign Data Details")
    
    # Prepare data for display
    display_data = campaign_data[[
        'ORDERNUMBER', 'CUSTOMERNAME', 'ORDERDATE', 'PRODUCTLINE', 
        'QUANTITYORDERED', 'PRICEEACH', 'SALES', 'DEALSIZE'
    ]].copy()
    
    display_data['ORDERDATE'] = display_data['ORDERDATE'].dt.strftime('%Y-%m-%d')
    
    # Show the data table
    st.dataframe(display_data, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>📊 Enhanced Marketing Analytics Dashboard | Created for demonstration purposes | Data is simulated</div>", unsafe_allow_html=True)