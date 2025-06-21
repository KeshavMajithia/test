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
st.sidebar.header("📍 Route Selection")

use_click = st.sidebar.toggle("Use map clicks to select start/end")

# Inputs (only used if not using clicks)
start_point = st.sidebar.text_input("Start Location", value="Connaught Place, New Delhi")
end_point = st.sidebar.text_input("End Location", value="Rajouri Garden, New Delhi")

uploaded_files = st.sidebar.file_uploader("📸 Upload Road Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# --- Predict Road Conditions ---
predictions = []
if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        pred = predict_image(img)
        predictions.append(pred)
    st.sidebar.success("🧠 Road Conditions: " + ", ".join(predictions))

# --- Session State to Store Map + Coords ---
if "start_coords" not in st.session_state:
    st.session_state.start_coords = None
if "end_coords" not in st.session_state:
    st.session_state.end_coords = None
if "route_coords" not in st.session_state:
    st.session_state.route_coords = []

# --- Interactive Map for Selecting Points ---
default_location = (28.6139, 77.2090)  # New Delhi center
route_map = folium.Map(location=default_location, zoom_start=12, tiles="cartodb positron")

# Add click instructions
if use_click:
    folium.Marker(default_location, popup="Click to select start/end points", icon=folium.Icon(color="blue")).add_to(route_map)

    # Display the map and capture clicks
    click_data = st_folium(route_map, width=1200, height=600)
    if click_data and click_data.get("last_clicked"):
        lat, lon = click_data["last_clicked"]["lat"], click_data["last_clicked"]["lng"]
        if st.session_state.start_coords is None:
            st.session_state.start_coords = (lat, lon)
            st.success(f"✅ Start point set at: {lat:.5f}, {lon:.5f}")
        elif st.session_state.end_coords is None:
            st.session_state.end_coords = (lat, lon)
            st.success(f"✅ End point set at: {lat:.5f}, {lon:.5f}")
        else:
            st.warning("❗ Start and End already selected. Refresh to reset or click Generate Route.")
else:
    st.session_state.start_coords = None
    st.session_state.end_coords = None
    click_data = None

# --- Generate Route ---
if st.sidebar.button("🚀 Generate Route"):
    try:
        with st.spinner("Calculating route..."):
            # Use clicked coords or geocode from input
            if use_click:
                start_coords = st.session_state.start_coords
                end_coords = st.session_state.end_coords
                if not start_coords or not end_coords:
                    st.error("❌ Please click on the map to select both start and end points.")
                    st.stop()
            else:
                try:
                    start_coords = ox.geocode(start_point)
                    end_coords = ox.geocode(end_point)
                except Exception:
                    st.error("❌ Could not geocode one of the entered addresses.")
                    st.stop()

            center = ((start_coords[0] + end_coords[0]) / 2,
                      (start_coords[1] + end_coords[1]) / 2)
            G = ox.graph.graph_from_point(center, dist=1000, network_type="drive")

            orig_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
            dest_node = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])
            route = nx.shortest_path(G, orig_node, dest_node, weight="length")

            coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]
            st.session_state.route_coords = coords

            # Reset map with route
            route_map = folium.Map(location=center, zoom_start=13, tiles="cartodb positron")
            pred_cycle = cycle(predictions if predictions else ["Good"])

            for i in range(len(coords) - 1):
                color = {
                    "Good": "green",
                    "Satisfactory": "orange",
                    "Poor": "red",
                    "Very Poor": "black"
                }.get(next(pred_cycle), "blue")

                folium.PolyLine(
                    [coords[i], coords[i + 1]],
                    color=color,
                    weight=6,
                    opacity=0.7
                ).add_to(route_map)

            folium.Marker(coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(route_map)
            folium.Marker(coords[-1], popup="End", icon=folium.Icon(color="red")).add_to(route_map)

            st.success("✅ Route generated!")

    except Exception as e:
        st.error(f"❌ Could not calculate route: {e}")

# --- If route already exists, draw it on map ---
if st.session_state.route_coords:
    for i in range(len(st.session_state.route_coords) - 1):
        folium.PolyLine(
            [st.session_state.route_coords[i], st.session_state.route_coords[i + 1]],
            color="blue", weight=5, opacity=0.6
        ).add_to(route_map)

    folium.Marker(st.session_state.route_coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(route_map)
    folium.Marker(st.session_state.route_coords[-1], popup="End", icon=folium.Icon(color="red")).add_to(route_map)

# --- Always Show Map at End ---
st_folium(route_map, width=1200, height=600)
