from mcp.server.fastmcp import FastMCP, Context
import os
import requests
from typing import Any, List

mcp = FastMCP("obsidian-notes", description="MCP server for interacting with Obsidian")
mcp = FastMCP("obsidian-notes", dependencies=["requests"])


def get_base_url() -> str:
    return 'https://127.0.0.1:27124'
    
def _get_headers() -> dict:
    headers = {
        'Authorization': f'Bearer 59cac049ae10f5986a266b901faf17d34cf813bff9103c0093a924cfa2ab5261'
    }
    return headers

def _safe_call(f) -> Any:
    try:
        return f()
    except requests.HTTPError as e:
        error_data = e.response.json() if e.response.content else {}
        code = error_data.get('errorCode', -1) 
        message = error_data.get('message', '<unknown>')
        raise Exception(f"Error {code}: {message}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")

@mcp.tool()
def files_in_vault() -> Any:
    """
    List all files in the Obsidian vault.
    
    Returns a list of files in the current Obsidian vault.
    """
    url = f"{get_base_url()}/vault/"
        
    def call_fn():
        response = requests.get(url, headers=_get_headers(), verify=False)
        response.raise_for_status()
            
        return response.json()['files']

    return _safe_call(call_fn)

@mcp.tool()
def add_content(note_topic: str, content: str) -> str:
    """
    Append content to a new or existing note in the vault.

    Args:
        note_topic (str): The topic of the note to add.
        content (str): The content of the note to add.

    Returns:
        str: A message indicating that the note was added.
    """
    url = f"{get_base_url()}/vault/{note_topic}.md"

    def call_fn():
        response = requests.post(url,
            headers= _get_headers() | {
                'Content-Type': 'text/markdown',
                "accept" : "*/*"
            },
            data=content,
            verify=False
        )
        response.raise_for_status()
        return f"Content added in {note_topic}"
    
    return _safe_call(call_fn)


@mcp.tool()
def update_note(note_name: str, content: str) -> str:
    """
    Update the content of a note in the vault.

    Args:
        note_name (str): The name of the note to update.
        content (str): The content of the note to update.

    Returns:
        str: A message indicating that the note was updated.
    """
    url = f"{get_base_url()}/vault/{note_name}"
    

    def call_fn():
        response = requests.put(url,
            headers=_get_headers() | {'Content-Type': 'text/markdown'},
            data=content,
            verify=False
        )
        response.raise_for_status()
        return f"Content updated in {note_name}"
    
    return _safe_call(call_fn)


@mcp.tool()
def create_folder(folder_name: str) -> str:
    """
    Create a new folder in the vault.

    Args:
        folder_name (str): The name of the folder to create.

    Returns:
        str: A message indicating that the folder was created.
    """

    path = f"{os.getenv('OBSIDIAN_VAULT_PATH')}/{folder_name}"
    os.makedirs(path, exist_ok=True)

    return f"Folder created: {folder_name}"


@mcp.tool()
def read_note(note_name: str) -> str:

    """
    Read the content of a note in the vault.

    Args:
        note_name (str): The name of the note to read.

    Returns:
        str: The content of the note.
    """

    url = f"{get_base_url()}/vault/{note_name}"

    def call_fn():
        response = requests.get(url, 
            headers= _get_headers() | {
                "accept": "application/vnd.olrapi.note+json"
            },
            verify=False
        )
        response.raise_for_status()
        return response.json()
    
    return _safe_call(call_fn)


@mcp.tool()
def search(query: str, context_len: int = 100) -> Any:

    """
    Search for notes in the vault using a simple string query.

    Args:
        query (str): The query to search for.
        context_len (int): The number of characters to return from the note (optional).

    """

    url = f"{get_base_url()}/search/simple"
    params = {
        "query": query,
        "contextLength": context_len
    }

    def call_fn():
        response = requests.post(url, params=params, headers=_get_headers() | {"accept": "application/json"}, verify=False)
        response.raise_for_status()

        return response.json()
    
    return _safe_call(call_fn)

