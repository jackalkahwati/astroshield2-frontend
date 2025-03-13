"""
Message Headers Module

This module provides standardized message header structures for all messages in the AstroShield platform.
It ensures consistent handling of metadata and traceability information across all subsystems.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Union


class MessageHeader:
    """
    Standardized message header with traceability support.
    
    This class handles message IDs, timestamps, source information, and traceability
    with trace IDs and parent message IDs. Use this header class for all internal
    and external messages to ensure consistent traceability.
    """
    
    def __init__(
        self,
        message_type: str,
        source: str,
        trace_id: Optional[str] = None,
        parent_message_ids: Optional[List[str]] = None,
        priority: str = "NORMAL",
        additional_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a message header with traceability information.
        
        Args:
            message_type: Type of the message (e.g., "state.vector", "ccdm.detection")
            source: Source system or component generating the message
            trace_id: Unique trace ID for tracking message lineage (generated if None)
            parent_message_ids: List of parent message IDs (empty list if None)
            priority: Message priority (default: "NORMAL")
            additional_metadata: Additional header metadata (optional)
        """
        self.message_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow().isoformat()
        self.message_type = message_type
        self.source = source
        self.priority = priority
        
        # Traceability
        self.trace_id = trace_id if trace_id else str(uuid.uuid4())
        self.parent_message_ids = parent_message_ids if parent_message_ids else []
        
        # Additional metadata
        self.metadata = additional_metadata if additional_metadata else {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert header to dictionary format for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the header
        """
        return {
            "messageId": self.message_id,
            "timestamp": self.timestamp,
            "messageType": self.message_type,
            "source": self.source,
            "priority": self.priority,
            "traceId": self.trace_id,
            "parentMessageIds": self.parent_message_ids,
            **self.metadata
        }
    
    @classmethod
    def from_dict(cls, header_dict: Dict[str, Any]) -> 'MessageHeader':
        """
        Create a MessageHeader instance from a dictionary.
        
        Args:
            header_dict: Dictionary containing header information
            
        Returns:
            MessageHeader: New instance with values from the dictionary
        """
        # Extract known fields
        message_type = header_dict.get("messageType", "unknown")
        source = header_dict.get("source", "unknown")
        trace_id = header_dict.get("traceId")
        parent_message_ids = header_dict.get("parentMessageIds", [])
        priority = header_dict.get("priority", "NORMAL")
        
        # Create new header
        header = cls(
            message_type=message_type,
            source=source,
            trace_id=trace_id,
            parent_message_ids=parent_message_ids,
            priority=priority
        )
        
        # Set fields that are directly copied
        header.message_id = header_dict.get("messageId", header.message_id)
        header.timestamp = header_dict.get("timestamp", header.timestamp)
        
        # Add any additional fields to metadata
        standard_fields = {
            "messageId", "timestamp", "messageType", "source", 
            "priority", "traceId", "parentMessageIds"
        }
        header.metadata = {
            k: v for k, v in header_dict.items() 
            if k not in standard_fields
        }
        
        return header
    
    def derive_child_header(self, message_type: str, source: str) -> 'MessageHeader':
        """
        Create a new header that maintains the trace chain for a derived message.
        
        Use this when creating a new message based on an existing message,
        to maintain the traceability chain.
        
        Args:
            message_type: Type of the new message
            source: Source system or component generating the new message
            
        Returns:
            MessageHeader: New header with the same trace ID and this message's ID as parent
        """
        return MessageHeader(
            message_type=message_type,
            source=source,
            trace_id=self.trace_id,
            parent_message_ids=[self.message_id] + self.parent_message_ids
        )
    
    def __str__(self) -> str:
        """String representation of the header for logging purposes."""
        return (f"MessageHeader(id={self.message_id}, type={self.message_type}, "
                f"source={self.source}, trace_id={self.trace_id})")


class MessageFactory:
    """
    Factory class for creating standardized messages with proper headers.
    
    This class provides methods to create new messages with standardized headers,
    maintaining traceability through the processing chain.
    """
    
    @staticmethod
    def create_message(
        message_type: str,
        source: str,
        payload: Dict[str, Any],
        trace_id: Optional[str] = None,
        parent_message_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a complete message with proper header and payload.
        
        Args:
            message_type: Type of the message
            source: Source system or component
            payload: Message payload data
            trace_id: Optional trace ID (generated if None)
            parent_message_ids: Optional list of parent message IDs
            
        Returns:
            Dict[str, Any]: Complete message with header and payload
        """
        header = MessageHeader(
            message_type=message_type,
            source=source,
            trace_id=trace_id,
            parent_message_ids=parent_message_ids
        )
        
        return {
            "header": header.to_dict(),
            "payload": payload
        }
    
    @staticmethod
    def derive_message(
        parent_message: Dict[str, Any],
        message_type: str,
        source: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a new message derived from a parent message, maintaining traceability.
        
        Args:
            parent_message: The original message being processed
            message_type: Type of the new message
            source: Source system or component for the new message
            payload: Payload for the new message
            
        Returns:
            Dict[str, Any]: New message with proper traceability to parent
        """
        parent_header = MessageHeader.from_dict(parent_message["header"])
        child_header = parent_header.derive_child_header(
            message_type=message_type,
            source=source
        )
        
        return {
            "header": child_header.to_dict(),
            "payload": payload
        } 