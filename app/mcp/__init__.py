"""
MCP package — Model Context Protocol server for DocuMind.

Exposes DocuMind's document Q&A pipeline to MCP clients
(Claude Desktop, Cursor, etc.) via two tools:
  - list_documents  : returns all uploaded documents
  - ask_document    : runs the agentic RAG loop against one document
"""
