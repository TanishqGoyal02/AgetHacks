from typing import List, Optional
from pydantic import BaseModel, Field
from zep_cloud import ZepClient
from datetime import datetime

class MemoryQuery(BaseModel):
    """Query parameters for memory search"""
    query: str = Field(..., description="The search query to find relevant memories")
    limit: int = Field(default=5, description="Maximum number of memories to retrieve")
    collection_name: str = Field(default="default", description="Name of the memory collection to search")

class Memory(BaseModel):
    """Represents a single memory entry"""
    content: str = Field(..., description="The content of the memory")
    metadata: dict = Field(default_factory=dict, description="Additional metadata about the memory")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the memory was created")

class MemoryTool:
    """Tool for accessing and querying the memory layer"""
    
    def __init__(self, zep_client: ZepClient):
        self.client = zep_client
    
    async def search_memories(self, query: MemoryQuery) -> List[Memory]:
        """Search for relevant memories based on the query"""
        try:
            # Search the memory collection
            results = await self.client.search(
                collection_name=query.collection_name,
                query=query.query,
                limit=query.limit
            )
            
            # Convert results to Memory objects
            memories = []
            for result in results:
                memory = Memory(
                    content=result.text,
                    metadata=result.metadata,
                    timestamp=result.created_at
                )
                memories.append(memory)
            
            return memories
        except Exception as e:
            raise Exception(f"Error searching memories: {str(e)}")
    
    async def add_memory(self, memory: Memory, collection_name: str = "default") -> str:
        """Add a new memory to the collection"""
        try:
            # Add the memory to the collection
            result = await self.client.add(
                collection_name=collection_name,
                text=memory.content,
                metadata=memory.metadata
            )
            return result.id
        except Exception as e:
            raise Exception(f"Error adding memory: {str(e)}")

# Example usage:
"""
async def main():
    # Initialize the Zep client
    client = ZepClient(api_key="your-api-key")
    
    # Create the memory tool
    memory_tool = MemoryTool(client)
    
    # Search for memories
    query = MemoryQuery(
        query="What was discussed about project X?",
        limit=3
    )
    memories = await memory_tool.search_memories(query)
    
    # Add a new memory
    new_memory = Memory(
        content="Discussed project timeline and deliverables",
        metadata={"project": "X", "topic": "timeline"}
    )
    memory_id = await memory_tool.add_memory(new_memory)
""" 