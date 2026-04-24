from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pathlib

STATIC_DIR = pathlib.Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def serve_dashboard():
        return FileResponse(str(STATIC_DIR / "index.html"))
