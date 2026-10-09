import streamlit as st
import pandas as pd
import plotly.express as px

st.title("EC2 Instance EDA Dashboard")

df = pd.read_csv("ec2dataset.csv")

# ── Data Cleaning ─────────────────────────────────────────────────────────────

df["Memory_GiB"] = (
    df["Instance Memory"]
    .str.extract(r"([\d.]+)")
    .astype(float)
)

df["vCPU_Count"] = (
    df["vCPUs"]
    .str.extract(r"(\d+)")
    .astype(float)
)

def clean_price(value):
    if pd.isna(value):
        return None
    value = str(value)
    if "unavailable" in value.lower():
        return None
    return float(value.replace("$", "").replace(" hourly", "").strip())

price_columns = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost",
]
for col in price_columns:
    df[col + "_USD"] = df[col].apply(clean_price)

df["Monthly_On_Demand"] = df["On Demand_USD"] * 730
df["Cost_Per_GiB"] = df["On Demand_USD"] / df["Memory_GiB"]
df["Cost_Per_vCPU"] = df["On Demand_USD"] / df["vCPU_Count"]
df["Memory_per_vCPU"] = df["Memory_GiB"] / df["vCPU_Count"]

def classify_storage(value):
    v = str(value).lower()
    if "ebs" in v:
        return "EBS"
    if "nvme" in v:
        return "NVMe SSD"
    if "ssd" in v:
        return "SSD"
    if "hdd" in v:
        return "HDD"
    return "Other"

df["Storage_Type"] = df["Instance Storage"].apply(classify_storage)

# ── Dataset Overview ──────────────────────────────────────────────────────────

st.subheader("Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Number of Instances", len(df))
col2.metric("Number of Columns", len(df.columns))
col3.metric("Missing Values", int(df.isna().sum().sum()))

with st.expander("Dataset Preview"):
    st.dataframe(df)

with st.expander("Dataset Structure"):
    st.write("**Columns:**", df.columns.tolist())
    st.write("**Data Types:**")
    st.write(df.dtypes)

# ── Sidebar Filters ───────────────────────────────────────────────────────────

st.sidebar.header("Filters")

max_memory = float(df["Memory_GiB"].max())
memory_filter = st.sidebar.slider(
    "Maximum Memory (GiB)",
    min_value=0.5,
    max_value=max_memory,
    value=max_memory,
)

cpu_values = sorted(df["vCPU_Count"].dropna().unique())
selected_cpu = st.sidebar.multiselect(
    "vCPU Count",
    options=cpu_values,
    default=cpu_values,
)

network_options = sorted(df["Network Performance"].dropna().unique())
selected_network = st.sidebar.multiselect(
    "Network Performance",
    options=network_options,
    default=network_options,
)

storage_options = sorted(df["Storage_Type"].unique())
selected_storage = st.sidebar.multiselect(
    "Storage Type",
    options=storage_options,
    default=storage_options,
)

max_price = float(df["On Demand_USD"].max(skipna=True))
price_filter = st.sidebar.slider(
    "Maximum Hourly Price ($)",
    min_value=0.0,
    max_value=max_price,
    value=max_price,
    step=0.01,
)

max_monthly = float(df["Monthly_On_Demand"].max(skipna=True))
monthly_filter = st.sidebar.slider(
    "Maximum Monthly Cost ($)",
    min_value=0.0,
    max_value=max_monthly,
    value=max_monthly,
    step=1.0,
)

filtered_df = df[
    (df["Memory_GiB"] <= memory_filter)
    & (df["vCPU_Count"].isin(selected_cpu))
    & (df["Network Performance"].isin(selected_network))
    & (df["Storage_Type"].isin(selected_storage))
    & (df["On Demand_USD"].fillna(0) <= price_filter)
    & (df["Monthly_On_Demand"].fillna(0) <= monthly_filter)
]

st.write(f"**{len(filtered_df)} instances** match the current filters.")

# ── KPI Cards ─────────────────────────────────────────────────────────────────

st.subheader("EC2 Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Instances", len(filtered_df))
c2.metric("Avg Memory", f"{filtered_df['Memory_GiB'].mean():.2f} GiB")
c3.metric("Avg vCPUs", f"{filtered_df['vCPU_Count'].mean():.1f}")
c4.metric("Avg Hourly Cost", f"${filtered_df['On Demand_USD'].mean():.4f}")

# ── EDA Visualizations ────────────────────────────────────────────────────────

st.subheader("Memory Distribution")
fig = px.histogram(filtered_df, x="Memory_GiB", nbins=30,
                   title="Distribution of EC2 Memory")
st.plotly_chart(fig, use_container_width=True)

st.subheader("vCPU Distribution")
fig = px.histogram(filtered_df, x="vCPU_Count",
                   title="Distribution of vCPUs")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Memory vs vCPUs")
fig = px.scatter(filtered_df, x="vCPU_Count", y="Memory_GiB",
                 hover_name="API Name", hover_data=["On Demand_USD"],
                 title="EC2 Memory vs vCPU")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Memory vs On-Demand Cost")
fig = px.scatter(filtered_df, x="Memory_GiB", y="On Demand_USD",
                 hover_name="API Name", size="vCPU_Count",
                 title="Memory vs EC2 On-Demand Cost")
st.plotly_chart(fig, use_container_width=True)

# ── Cost Analysis ─────────────────────────────────────────────────────────────

st.subheader("Monthly On-Demand Cost")
st.dataframe(
    filtered_df[["Name", "API Name", "Memory_GiB", "vCPU_Count",
                 "On Demand_USD", "Monthly_On_Demand"]]
)

st.subheader("Lowest-Cost EC2 Instances")
cheapest = (
    filtered_df
    .sort_values("On Demand_USD")
    [["Name", "API Name", "Memory_GiB", "vCPU_Count",
      "On Demand_USD", "Monthly_On_Demand"]]
    .head(10)
)
st.dataframe(cheapest)

st.subheader("Highest-Cost EC2 Instances")
most_expensive = (
    filtered_df
    .sort_values("On Demand_USD", ascending=False)
    [["Name", "API Name", "Memory_GiB", "vCPU_Count",
      "On Demand_USD", "Monthly_On_Demand"]]
    .head(10)
)
st.dataframe(most_expensive)

st.subheader("Cost per GiB of Memory")
efficiency = (
    filtered_df
    .sort_values("Cost_Per_GiB")
    [["Name", "API Name", "Memory_GiB", "vCPU_Count",
      "On Demand_USD", "Cost_Per_GiB", "Cost_Per_vCPU"]]
    .head(15)
)
st.dataframe(efficiency)

st.subheader("Memory per vCPU")
mem_per_cpu = (
    filtered_df
    .sort_values("Memory_per_vCPU", ascending=False)
    [["Name", "API Name", "Memory_GiB", "vCPU_Count", "Memory_per_vCPU"]]
    .head(15)
)
st.dataframe(mem_per_cpu)

# ── Pricing Comparison ────────────────────────────────────────────────────────

st.subheader("EC2 Pricing Comparison")
pricing = filtered_df[[
    "Name", "API Name",
    "On Demand_USD",
    "Linux Reserved cost_USD",
    "Linux Spot Minimum cost_USD",
    "Windows On Demand cost_USD",
    "Windows Reserved cost_USD",
]]
st.dataframe(pricing)

st.subheader("Pricing Model — Instance Detail")
selected_instance_name = st.selectbox(
    "Select an EC2 Instance",
    filtered_df["API Name"].unique(),
)
instance = filtered_df[filtered_df["API Name"] == selected_instance_name].iloc[0]

pricing_data = pd.DataFrame({
    "Pricing Model": ["On Demand", "Linux Reserved", "Linux Spot"],
    "Hourly Cost": [
        instance["On Demand_USD"],
        instance["Linux Reserved cost_USD"],
        instance["Linux Spot Minimum cost_USD"],
    ],
}).dropna()

pricing_data["Monthly Cost"] = pricing_data["Hourly Cost"] * 730

fig = px.bar(pricing_data, x="Pricing Model", y="Hourly Cost",
             title=f"Pricing Comparison: {selected_instance_name}")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(pricing_data)

# ── Data Export ───────────────────────────────────────────────────────────────

csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download Filtered Dataset",
    data=csv,
    file_name="filtered_ec2_instances.csv",
    mime="text/csv",
)
