from pydantic import BaseModel

class LinkedInPosts(BaseModel):
    post_idx: int
    angle: str
    content: str

class GeneratedPosts(BaseModel):
    posts: list[LinkedInPosts]


