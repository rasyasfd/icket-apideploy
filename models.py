from pydantic import BaseModel

class place(BaseModel):
    place: str
    
    class Config:
        schema_extra = {
            'example': {
                'place': 'Situ Patenggang'
            }
        }