import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
from itertools import cycle

# --- Config ---
st.set_page_config(page_title="SmartRoute AI", layout="wide")
st.title("🛣️ SmartRoute AI - Road Health-Aware Routing")

# --- Load Model ---
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load("road_health_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
classes = ['Good', 'Satisfactory', 'Poor', 'Very Poor']

def predict_image(img):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
    return classes[pred.item()]

# --- Sidebar Inputs ---
st.sidebar.header("📍 Route Planning")

# Session state
if "selecting" not in st.session_state:
    st.session_state.selecting = None
if "start_coords" not in st.session_state:
    st.session_state.start_coords = None
if "end_coords" not in st.session_state:
    st.session_state.end_coords = None
if "route_coords" not in st.session_state:
    st.session_state.route_coords = []

# Prediction
uploaded_files = st.sidebar.file_uploader("📸 Upload Road Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

predictions = []
if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        pred = predict_image(img)
        predictions.append(pred)
    st.sidebar.success("🧠 Road Conditions: " + ", ".join(predictions))

# Button flow for point selection
st.sidebar.markdown("### 🖱️ Click-based Selection")
col1, col2 = st.sidebar.columns(2)
if col1.button("Select Start"):
    st.session_state.selecting = "start"
if col2.button("Select Destination"):
    st.session_state.selecting = "end"

if st.sidebar.button("Reset Selection"):
    st.session_state.start_coords = None
    st.session_state.end_coords = None
    st.session_state.route_coords = []
    st.session_state.selecting = None

# --- Map Rendering ---
default_location = (28.6139, 77.2090)  # New Delhi center
map_center = st.session_state.start_coords or st.session_state.end_coords or default_location
route_map = folium.Map(location=map_center, zoom_start=13, tiles="cartodb positron")

# Marker preview
if st.session_state.start_coords:
    folium.Marker(st.session_state.start_coords, popup="Start", icon=folium.Icon(color="green")).add_to(route_map)
if st.session_state.end_coords:
    folium.Marker(st.session_state.end_coords, popup="Destination", icon=folium.Icon(color="red")).add_to(route_map)

# Detect map click
click_data = st_folium(route_map, width=1200, height=600)

if click_data and click_data.get("last_clicked"):
    lat, lon = click_data["last_clicked"]["lat"], click_data["last_clicked"]["lng"]
    if st.session_state.selecting == "start":
        st.session_state.start_coords = (lat, lon)
        st.session_state.selecting = None
        st.success(f"✅ Start location set: {lat:.5f}, {lon:.5f}")
    elif st.session_state.selecting == "end":
        st.session_state.end_coords = (lat, lon)
        st.session_state.selecting = None
        st.success(f"✅ Destination set: {lat:.5f}, {lon:.5f}")

# --- Route Calculation ---
if st.sidebar.button("🚀 Generate Route"):
    if not st.session_state.start_coords or not st.session_state.end_coords:
        st.error("❌ Please select both start and destination points first.")
        st.stop()
    try:
        with st.spinner("Calculating route..."):
            start_coords = st.session_state.start_coords
            end_coords = st.session_state.end_coords
            center = ((start_coords[0] + end_coords[0]) / 2,
                      (start_coords[1] + end_coords[1]) / 2)
            G = ox.graph.graph_from_point(center, dist=1500, network_type="drive")

            orig_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
            dest_node = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])
            route = nx.shortest_path(G, orig_node, dest_node, weight="length")

            coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]
            st.session_state.route_coords = coords
            st.success("✅ Route generated!")

    except Exception as e:
        st.error(f"❌ Could not calculate route: {e}")

# --- Draw Route if Available ---
if st.session_state.route_coords:
    pred_cycle = cycle(predictions if predictions else ["Good"])
    for i in range(len(st.session_state.route_coords) - 1):
        color = {
            "Good": "green",
            "Satisfactory": "orange",
            "Poor": "red",
            "Very Poor": "black"
        }.get(next(pred_cycle), "blue")

        folium.PolyLine(
            [st.session_state.route_coords[i], st.session_state.route_coords[i + 1]],
            color=color,
            weight=6,
            opacity=0.7
        ).add_to(route_map)

    folium.Marker(st.session_state.route_coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(route_map)
    folium.Marker(st.session_state.route_coords[-1], popup="End", icon=folium.Icon(color="red")).add_to(route_map)

# --- Display Final Map ---
st_folium(route_map, width=1200, height=600)
