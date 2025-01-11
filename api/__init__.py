"""API package initialization"""
from fastapi import FastAPI
from .endpoints import app

__all__ = ['app']
