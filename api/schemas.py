import pydantic as pd


class PredictRequestModel(pd.BaseModel):
    input: str = ''


class PredictResponseItem(pd.BaseModel):
    start_index: int
    end_index: int
    entity: str = ''
