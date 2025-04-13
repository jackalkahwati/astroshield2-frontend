import logging
import asyncio
from typing import Dict, Any, List, Callable, Awaitable, Optional, Union
import uuid
import time
import inspect

logger = logging.getLogger(__name__)

# Type for event handler functions
EventHandler = Callable[[Dict[str, Any]], Union[None, Awaitable[None]]]

class EventBus:
    """
    Implementation of an event bus for event-driven architecture.
    Supports synchronous and asynchronous event handlers.
    """
    
    def __init__(self):
        """Initialize the event bus"""
        self.subscribers: Dict[str, List[Dict[str, Any]]] = {}
        self.history: List[Dict[str, Any]] = []
        self.max_history = 100  # Maximum number of events to keep in history
    
    def subscribe(self, event_type: str, handler: EventHandler, subscriber_id: Optional[str] = None) -> str:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Function to call when event occurs
            subscriber_id: Optional ID for the subscriber (auto-generated if None)
            
        Returns:
            Subscriber ID
        """
        if not subscriber_id:
            subscriber_id = str(uuid.uuid4())
        
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        # Check if handler is coroutine function
        is_async = inspect.iscoroutinefunction(handler)
        
        self.subscribers[event_type].append({
            "id": subscriber_id,
            "handler": handler,
            "is_async": is_async,
            "created_at": time.time()
        })
        
        logger.debug(f"Subscribed {subscriber_id} to {event_type} events")
        return subscriber_id
    
    def unsubscribe(self, event_type: str, subscriber_id: str) -> bool:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Type of event
            subscriber_id: Subscriber ID
            
        Returns:
            True if unsubscribed, False if not found
        """
        if event_type not in self.subscribers:
            return False
        
        original_length = len(self.subscribers[event_type])
        self.subscribers[event_type] = [
            sub for sub in self.subscribers[event_type] if sub["id"] != subscriber_id
        ]
        
        if len(self.subscribers[event_type]) < original_length:
            logger.debug(f"Unsubscribed {subscriber_id} from {event_type} events")
            return True
        
        return False
    
    def publish(self, event_type: str, data: Dict[str, Any] = None) -> None:
        """
        Publish a synchronous event.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        if data is None:
            data = {}
        
        event = {
            "id": str(uuid.uuid4()),
            "type": event_type,
            "timestamp": time.time(),
            "data": data
        }
        
        # Add to history
        self._add_to_history(event)
        
        if event_type not in self.subscribers:
            logger.debug(f"No subscribers for event type {event_type}")
            return
        
        subscribers = self.subscribers[event_type]
        
        for subscriber in subscribers:
            try:
                if subscriber["is_async"]:
                    # For async handlers, create a task to run it
                    asyncio.create_task(self._call_async_handler(subscriber["handler"], event))
                else:
                    # For sync handlers, call directly
                    subscriber["handler"](event)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {str(e)}")
    
    async def publish_async(self, event_type: str, data: Dict[str, Any] = None) -> None:
        """
        Publish an event and wait for all async handlers to complete.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        if data is None:
            data = {}
        
        event = {
            "id": str(uuid.uuid4()),
            "type": event_type,
            "timestamp": time.time(),
            "data": data
        }
        
        # Add to history
        self._add_to_history(event)
        
        if event_type not in self.subscribers:
            logger.debug(f"No subscribers for event type {event_type}")
            return
        
        subscribers = self.subscribers[event_type]
        tasks = []
        
        for subscriber in subscribers:
            try:
                if subscriber["is_async"]:
                    # For async handlers, add to task list
                    tasks.append(self._call_async_handler(subscriber["handler"], event))
                else:
                    # For sync handlers, call directly
                    subscriber["handler"](event)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {str(e)}")
        
        # Wait for all async handlers to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _call_async_handler(self, handler: EventHandler, event: Dict[str, Any]) -> None:
        """
        Call an async event handler.
        
        Args:
            handler: Async handler function
            event: Event data
        """
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"Error in async event handler for {event['type']}: {str(e)}")
    
    def _add_to_history(self, event: Dict[str, Any]) -> None:
        """
        Add an event to history, maintaining max size.
        
        Args:
            event: Event to add
        """
        self.history.append(event)
        
        # Trim history if it exceeds max size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_history(self, event_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent events from history.
        
        Args:
            event_type: Optional filter by event type
            limit: Maximum number of events to return
            
        Returns:
            List of events
        """
        if event_type:
            filtered = [e for e in self.history if e["type"] == event_type]
            return filtered[-limit:]
        else:
            return self.history[-limit:]
    
    def clear_subscribers(self, event_type: Optional[str] = None) -> None:
        """
        Clear subscribers.
        
        Args:
            event_type: Optional event type to clear, or all if None
        """
        if event_type:
            if event_type in self.subscribers:
                self.subscribers[event_type] = []
                logger.debug(f"Cleared all subscribers for {event_type}")
        else:
            self.subscribers = {}
            logger.debug("Cleared all subscribers")
    
    def get_subscriber_count(self, event_type: Optional[str] = None) -> int:
        """
        Get count of subscribers.
        
        Args:
            event_type: Optional event type to count, or all if None
            
        Returns:
            Number of subscribers
        """
        if event_type:
            return len(self.subscribers.get(event_type, []))
        else:
            return sum(len(subs) for subs in self.subscribers.values()) 