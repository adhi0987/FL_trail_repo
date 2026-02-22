from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Allow React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FL State ---
# Simulating a simple model's weights (e.g., a single Dense layer for demo purposes)
# In a real scenario, this would be the exact shape of your TF.js model layers
global_weights = [np.random.rand(10, 3).tolist(), np.random.rand(3).tolist()]
client_updates = []
current_round = 0
MIN_CLIENTS_FOR_FEDAVG = 3

class WeightUpdate(BaseModel):
    client_id: str
    weights: list
    local_fpr: float

def aggregate_weights():
    """Federated Averaging (FedAvg) Algorithm"""
    global global_weights, client_updates, current_round
    print(f"Aggregating weights for Round {current_round + 1}...")
    
    # Average the weights
    new_weights = []
    num_layers = len(global_weights)
    
    for layer_idx in range(num_layers):
        layer_weights = [client[layer_idx] for client in client_updates]
        averaged_layer = np.mean(layer_weights, axis=0).tolist()
        new_weights.append(averaged_layer)
        
    global_weights = new_weights
    client_updates.clear()
    current_round += 1
    print(f"Round {current_round} Complete.")

@app.get("/api/model/global")
def get_global_model():
    return {
        "round": current_round,
        "weights": global_weights
    }

@app.post("/api/model/update")
def submit_local_update(update: WeightUpdate, background_tasks: BackgroundTasks):
    global client_updates
    client_updates.append(update.weights)
    
    # If we have enough updates, trigger FedAvg in the background
    if len(client_updates) >= MIN_CLIENTS_FOR_FEDAVG:
        background_tasks.add_task(aggregate_weights)
        
    return {"status": "Update received", "pending_aggregations": len(client_updates)}