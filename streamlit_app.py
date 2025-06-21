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

# --- Page Config ---
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

# --- Prediction Function ---
def predict_image(img):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
    return classes[pred.item()]

# --- Sidebar Inputs ---
st.sidebar.header("📍 Route Controls")
start_point = st.sidebar.text_input("Start Location", value="Connaught Place, New Delhi")
end_point = st.sidebar.text_input("End Location", value="Rajouri Garden, New Delhi")

uploaded_files = st.sidebar.file_uploader(
    "📸 Upload Multiple Road Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

predictions = []
if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        pred = predict_image(img)
        predictions.append(pred)

if uploaded_files:
    st.sidebar.success("🧠 Road Conditions: " + ", ".join(predictions))

# --- Default Base Map (always shown) ---
default_location = (28.6139, 77.2090)  # New Delhi
route_map = folium.Map(location=default_location, zoom_start=12, tiles="cartodb positron")

# --- Generate Route and Overlay ---
if st.sidebar.button("Generate Route"):
    try:
        with st.spinner("Calculating route..."):
            start_coords = ox.geocode(start_point)
            end_coords = ox.geocode(end_point)

            if not start_coords or not end_coords:
                st.error("❌ Invalid location inputs.")
                st.stop()

            center = ((start_coords[0] + end_coords[0]) / 2,
                      (start_coords[1] + end_coords[1]) / 2)
            G = ox.graph.graph_from_point(center, dist=1000, network_type="drive")

            orig_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
            dest_node = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])
            route = nx.shortest_path(G, orig_node, dest_node, weight="length")

            coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]

            if len(coords) < 2:
                st.error("❌ Route too short to display.")
                st.stop()

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

    except Exception as e:
        st.error(f"❌ Could not calculate route: {e}")

# --- Always Display the Map ---
st_folium(route_map, width=1200, height=600)
