from fastapi import FastAPI

# Create a FastAPI instance
app = FastAPI()

# Define a path operation 
@app.get("/")
def read_root():
    return{"Hello": "World"}