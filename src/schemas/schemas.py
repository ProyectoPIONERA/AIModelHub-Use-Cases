"""Pydantic schemas used by the API server and service layers.

This module defines request payloads and response models for flares
linguistic predictions.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class TextRequest(BaseModel):
    Id: int
    Text: str


class Tag(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    Tag_Start: int
    Tag_End: int
    Label_5W1H: str = Field(..., alias="5W1H_Label")
    Tag_Text: str


class ReliabilitySample(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    Id: int
    Text: str
    Label_5W1H: str = Field(..., alias="5W1H_Label")
    Tag_Text: str
    Tag_Start: int
    Tag_End: int


class ReliabilityPrediction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    Id: int
    Text: str
    Label_5W1H: str = Field(..., alias="5W1H_Label")
    Tag_Text: str
    Tag_Start: int
    Tag_End: int
    Reliability_Label: str

