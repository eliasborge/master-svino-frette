# Master Thesis

This repository contains the source code and documentation associated with the Master's Thesis "Words Are Loaded Pistols: Classifying Threats of Violence in Extremist Social Media Using Quantized Large Language Models" by Thomas Addisu Sæve Frette and Elias Borge Svinø at the Norwegian University of Science and Technology (NTNU), supervised by Björn Gambäck.


## Setup

Follow these steps to set up your local environment:

1. **Clone the repository**

   ```bash
   git clone https://github.com/eliasborge/master-svino-frette.git
   ```

2. **Install Python 3.13.x**

Ensure that Python version **3.13.x** is installed on your system.

You can check your current Python version with:

```bash
python --version
```

3. **Install Dependencies**

```bash
pip install pydantic
```

4. **Install Ollama**

Ollama is required to run quantized LLMs locally.

1. Download and install Ollama from: [https://ollama.com](https://ollama.com)
2. Follow the installation instructions for your operating system.
3. Start the Ollama service by running:

    ```bash
    Pip install Ollama
    Ollama serve 
    ````
All experiments were run on [IDUN](https://www.hpc.ntnu.no/idun/)

## Code Structure


The core logic for classification is located in the `src` directory. It is organized into two main components:

- **Agents**: All agent classes are defined in `src/agents/`. These include:
  - `framing_agent.py`
  - `intent_agent.py`
  - `context_agent.py`
  - `classification_agent.py`
  - `SinglePrompt_agent.py`
  - `batch_agent.py`
  - `content_filter_agent.py`

- **Pipelines**: The different classification strategies are implemented in `src/` and include:
  - `IAB_pipeline.py` – Individual Agent-Based pipeline
  - `CAAB_pipeline.py` – Context-Aware Agent-Based pipeline
  - `CHAB_pipeline.py` – Conversation History Agent-Based pipeline
  - `SP_pipeline.py` – Single-Prompt classification pipeline
  - `model_config.py` – configuration for models used in pipelines
    
