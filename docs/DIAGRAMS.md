# System Architecture Diagrams

This document contains information about the architectural diagrams in the README.md file.

## Viewing the Diagrams

The README.md file contains Mermaid diagrams that might not render properly in all Markdown viewers. Here are some options for viewing them:

1. **GitHub**: GitHub natively supports Mermaid diagrams in Markdown files
2. **VS Code**: Install the "Markdown Preview Mermaid Support" extension
3. **Mermaid Live Editor**: Copy the Mermaid code blocks to [Mermaid Live Editor](https://mermaid.live/)
4. **Markdown Viewers**: Some markdown viewers like Typora also support Mermaid diagrams

## Diagram Descriptions

### Architecture Diagram

The architecture diagram shows:
- The main system components (modules)
- How data flows between them
- The external dependencies (APIs)
- The output artifacts produced by each module

Key points:
- The Runner orchestrates the entire pipeline
- Each module has a clear responsibility
- Data flows sequentially through the pipeline
- External API dependencies are clearly marked
- Configuration controls the behavior of all modules

### Module Interaction Flow

The sequence diagram shows:
- The temporal order of operations
- How modules interact during execution
- Data transfer between modules
- The complete end-to-end workflow

Key points:
- The User initiates the process
- The Runner coordinates all interactions
- Each module processes and passes data to the next
- The LLM only works with a sample of reviews
- Final results are returned to the User