"""Parallel agents in LangGraph: map-reduce with ``Send``, sectioning and voting."""

from parallel_agents.config import Settings
from parallel_agents.corpus import Corpus
from parallel_agents.fake import FakeDeskModel
from parallel_agents.isolation import BranchFailure
from parallel_agents.llm import make_model
from parallel_agents.research import Finding
from parallel_agents.research import build_research_desk
from parallel_agents.research import build_researcher
from parallel_agents.sectioning import build_sectioning
from parallel_agents.voting import build_voting

__version__ = "0.1.0"

__all__ = [
    "BranchFailure",
    "Corpus",
    "FakeDeskModel",
    "Finding",
    "Settings",
    "__version__",
    "build_research_desk",
    "build_researcher",
    "build_sectioning",
    "build_voting",
    "make_model",
]
