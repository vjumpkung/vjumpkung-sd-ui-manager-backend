import uuid


def generate_uuid(url: str):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, url))
