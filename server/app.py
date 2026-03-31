"""
Data Cleaning OpenEnv Environment - FastAPI Server

Creates the FastAPI app via OpenEnv helper.
"""
import uvicorn
from openenv.core.env_server import create_fastapi_app

from data_cleaning_env.server.environment import DataCleaningEnvironment
from data_cleaning_env.models import DataCleaningAction, DataCleaningObservation

app = create_fastapi_app(
    DataCleaningEnvironment,   # factory callable — called once per session
    DataCleaningAction,
    DataCleaningObservation,
)


def main():
    """Entry point for the server."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

