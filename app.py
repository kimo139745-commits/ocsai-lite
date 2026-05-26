from fastapi import FastAPI

print("APP IMPORTED")

app = FastAPI()

print("FASTAPI APP CREATED")


@app.get("/")
def home():
    return {
        "message": "OCSAI-lite server is running",
        "status": "ok"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }
