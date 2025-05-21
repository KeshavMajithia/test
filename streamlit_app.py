# streamlit_app.py
import streamlit as st
import pandas as pd
import datetime
import plotly.express as px

st.set_page_config(page_title="Auto Dashboard Generator", layout="wide")

st.title("Auto Dashboard Generator")
st.markdown("Upload your sales file (CSV), click 'Preprocess', and get an interactive dashboard automatically!")

uploaded_file = st.file_uploader("Upload your file (CSV or Excel)", type=['csv', 'xlsx'])

if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

if uploaded_file:

    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y", errors='coerce').dt.date
        df.dropna(subset=['Date'], inplace=True)

    st.subheader("Raw File Preview")
    st.dataframe(df.head())

    if st.button("Preprocess"):
        if 'Direct Sales' in df.columns and 'Indirect Sales' in df.columns:
            df['Total Sales'] = df['Direct Sales'] + df['Indirect Sales']
            st.session_state.processed_df = df
            st.success("Preprocessing Complete. 'Total Sales' column added.")
        else:
            st.error("'Direct Sales' and 'Indirect Sales' columns are missing in the uploaded file.")

if st.session_state.processed_df is not None:
    df = st.session_state.processed_df

    st.subheader("Processed Data Preview")
    st.dataframe(df.head())

    st.header("Auto-Generated Dashboard")

    with st.sidebar:
        st.header("Filter Options")

        # Date filter
        min_date = min(df['Date'])
        max_date = max(df['Date'])
        selected_date = st.date_input("Select Date", min_value=min_date, max_value=max_date, value=min_date, key="date")

        # Campaign filter
        campaign_names = df['Campaign Name'].dropna().unique().tolist()
        selected_campaign = st.selectbox("Select Campaign Name", options=["All"] + campaign_names, key="campaign")

        # Targeting Type filter
        targeting_types = df['Targeting Type'].dropna().unique().tolist()
        selected_type = st.selectbox("Select Targeting Type", options=["All"] + targeting_types, key="type")

        # Targeting Value filter
        targeting_values = df['Targeting Value'].dropna().unique().tolist()
        selected_value = st.selectbox("Select Targeting Value", options=["All"] + targeting_values, key="value")

    # Apply filters 
    filtered_df = df[df['Date'] == selected_date]

    if selected_campaign != "All":
        filtered_df = filtered_df[filtered_df['Campaign Name'] == selected_campaign]

    if selected_type != "All":
        filtered_df = filtered_df[filtered_df['Targeting Type'] == selected_type]

    if selected_value != "All":
        filtered_df = filtered_df[filtered_df['Targeting Value'] == selected_value]

    # DYNAMIC SUMS & ROI CALCULATION
    if not filtered_df.empty:
        total_budget_consumed = filtered_df['Estimated Budget Consumed'].sum()
        total_sales_sum = filtered_df['Total Sales'].sum()

        # Display metrics side by side
        col1, col2, col3 = st.columns(3)

        col1.metric(label="Total Estimated Budget Consumed", value=f"{total_budget_consumed:,.2f}")
        col2.metric(label="Total Sales", value=f"{total_sales_sum:,.2f}")

        if total_budget_consumed > 0:
            roi = (total_sales_sum ) / total_budget_consumed
            roi_percentage = round(roi, 2)
            col3.metric(label="ROI (Return on Investment)", value=f"{roi_percentage}")
        else:
            col3.warning("Budget Consumed is 0, cannot calculate ROI.")
    else:
        st.warning("Filtered data is empty or missing required columns for ROI calculation.")
        
    # Total Sales Over Time
    st.subheader("Total Sales Over Time")
    time_df = df.groupby('Date')['Total Sales'].sum().reset_index()
    fig_area = px.area(time_df, x='Date', y='Total Sales', title='Total Sales Trend Over Time', template='plotly_white')
    st.plotly_chart(fig_area, use_container_width=True)

    # Sales by Campaign Name
    st.subheader("Campaign Performance")
    campaign_df = filtered_df.groupby('Campaign Name')['Total Sales'].sum().reset_index().sort_values(by='Total Sales', ascending=False)
    if not campaign_df.empty:
        fig_campaign = px.bar(campaign_df, x='Campaign Name', y='Total Sales',
                            title='Sales by Campaign',
                            labels={'Total Sales': 'Total Sales Amount'},
                            color='Total Sales')
        st.plotly_chart(fig_campaign, use_container_width=True)
    else:
        st.info("No campaign data available for the selected filters.")
        
    #Estimated budget consumed by campaign performance
        
    if not filtered_df.empty:
        st.subheader("Estimated Budget Consumed by Campaign Name")
        budget_campaign_df = filtered_df.groupby('Campaign Name').agg({
            'Estimated Budget Consumed': 'sum'
        }).reset_index()

        fig_budget = px.bar(budget_campaign_df,
                            x='Campaign Name',
                            y='Estimated Budget Consumed',
                            title='Estimated Budget Consumed per Campaign',
                            color='Estimated Budget Consumed',
                            labels={'Estimated Budget Consumed': 'Estimated Budget Consumed'},
                            color_continuous_scale='Blues')
        st.plotly_chart(fig_budget, use_container_width=True)

    # ROI by Targeting Type
    st.subheader(" ROI by Targeting Type")
    roi_df = filtered_df.groupby('Targeting Type').agg({
        'Estimated Budget Consumed': 'sum',
        'Total Sales': 'sum'
    }).reset_index()
    roi_df = roi_df[roi_df['Estimated Budget Consumed'] > 0]
    if not roi_df.empty:
        roi_df['ROI'] = roi_df['Total Sales'] / roi_df['Estimated Budget Consumed']
        fig_roi = px.bar(roi_df, x='Targeting Type', y='ROI',
                        title='ROI by Targeting Type',
                        color='ROI')
        st.plotly_chart(fig_roi, use_container_width=True)
    else:
        st.info("ROI data not available for the selected filters.")

    # HIDE UNWANTED COLUMNS
    columns_to_hide = [
        'Most Viewed Position',
        'Pacing Type',
        'Direct Quantities Sold',
        'Indirect Quantities Sold',
        'Direct ATC',
        'Indirect ATC'
    ]
    display_df = filtered_df.drop(columns=[col for col in columns_to_hide if col in filtered_df.columns])

    st.subheader("Filtered Data Table")
    st.dataframe(display_df)
