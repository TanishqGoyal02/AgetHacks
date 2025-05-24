from typing import List, Optional
from pydantic import BaseModel, Field
from zep_cloud.client import Zep, AsyncZep
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
    """Tool for accessing and querying the memory layer (low-level KG search)"""
    
    def __init__(self, zep_client):  # Accept either Zep or AsyncZep
        self.client = zep_client
    
    def search_memories(self, query: MemoryQuery, user_id: str = None, group_id: str = None) -> List[Memory]:
        """Search for relevant memories based on the query using the knowledge graph (low-level)."""
        try:
            # Use the graph search method, require user_id or group_id
            search_kwargs = {
                'query': query.query,
                'limit': query.limit
            }
            if user_id:
                search_kwargs['user_id'] = user_id
            if group_id:
                search_kwargs['group_id'] = group_id
            results = self.client.graph.search(**search_kwargs)
            # Convert results to Memory objects
            memories = []
            for result in results:
                memory = Memory(
                    content=getattr(result, 'text', ''),
                    metadata=getattr(result, 'metadata', {}),
                    timestamp=getattr(result, 'created_at', datetime.now())
                )
                memories.append(memory)
            return memories
        except Exception as e:
            raise Exception(f"Error searching memories: {str(e)}")
    
    def add_memory(self, memory: Memory, collection_name: str = "default") -> str:
        """Add a new memory to the collection (low-level KG add)."""
        try:
            # Add the memory to the collection
            result = self.client.graph.add(
                group_id=collection_name,
                type="text",
                data=memory.content
            )
            return getattr(result, 'id', None)
        except Exception as e:
            raise Exception(f"Error adding memory: {str(e)}")

# Example usage:
"""
async def main():
    client = Zep(api_key="your-api-key")
    memory_tool = MemoryTool(client)
    query = MemoryQuery(query="What was discussed about project X?", limit=3)
    memories = await memory_tool.search_memories(query)
    new_memory = Memory(
        content="Discussed project timeline and deliverables",
        metadata={"project": "X", "topic": "timeline"}
    )
    memory_id = await memory_tool.add_memory(new_memory)
""" 