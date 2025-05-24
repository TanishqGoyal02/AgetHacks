from pydantic import BaseModel, Field
from typing import Optional
from zep_cloud.client import Zep, AsyncZep
from datetime import datetime

class HealthData(BaseModel):
    user_id: str = Field(..., description="User identifier")
    date: str = Field(..., description="Date of the health data (YYYY-MM-DD)")
    sleep_score: int = Field(..., description="Sleep score out of 100")
    hydration_liters: float = Field(..., description="Hydration in liters")
    stress_level_heart_rate: int = Field(..., description="Average heart rate as stress level")
    steps_taken: int = Field(..., description="Number of steps taken")
    calorie_burn: int = Field(..., description="Calories burned")

class HealthDataTool:
    def __init__(self, zep_client):
        self.client = zep_client

    def add_health_data_to_kg(self, health_data: HealthData):
        group_id = f"health_{health_data.user_id}_{health_data.date}"
        data = health_data.dict()
        # Add health data as a node in the knowledge graph
        result = self.client.graph.add(
            group_id=group_id,
            type="json",
            data=data
        )
        return result

# Example usage:
"""
async def main():
    client = Zep(api_key="your-api-key")
    health_tool = HealthDataTool(client)
    health_data = HealthData(
        user_id="user_1",
        date="2024-06-01",
        sleep_score=85,
        hydration_liters=2.3,
        stress_level_heart_rate=78,
        steps_taken=10432,
        calorie_burn=2350
    )
    await health_tool.add_health_data_to_kg(health_data)
""" 