#!/usr/bin/env python3
"""Smoke-test the officialdom-v15 Space chat with one short question.

Uses gradio_client against the Space. Prints the model reply (truncated).
Only one question, to spare the 5 min/day ZeroGPU budget.
"""
import sys

from gradio_client import Client

SPACE = "unimaster/officialdom-v15"
QUESTION = "2026年赤字率比2025年高还是低？高多少个百分点？"

client = Client(SPACE, httpx_kwargs={"timeout": 300})
result = client.predict(message=QUESTION, api_name="/answer")
print("REPLY:")
print(result if isinstance(result, str) else str(result)[:2000])
