# pipeline/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TypeVar, Generic
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Type variables for input/output types
T = TypeVar("T")
U = TypeVar("U")


class NodeStatus(str, Enum):
    """Status of a pipeline node's execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeResult(Generic[T]):
    """Result of a node's execution."""

    status: NodeStatus
    data: Optional[T] = None
    error: Optional[Exception] = None

    @property
    def is_success(self) -> bool:
        """Check if the node executed successfully."""
        return self.status == NodeStatus.COMPLETED


class Node(Generic[T, U]):
    """Base class for pipeline nodes."""

    def __init__(self, name: str, inputs: Set[str], outputs: Set[str]):
        """
        Initialize a pipeline node.

        Args:
            name: Unique name of the node
            inputs: Set of input data keys required by this node
            outputs: Set of output data keys produced by this node
        """
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self._status = NodeStatus.PENDING

    @abstractmethod
    async def execute(self, inputs: Dict[str, T]) -> NodeResult[U]:
        """
        Execute the node's processing logic.

        Args:
            inputs: Dictionary mapping input names to their values

        Returns:
            NodeResult containing the execution status and output data
        """
        pass

    @property
    def status(self) -> NodeStatus:
        """Get the current execution status of the node."""
        return self._status

    @status.setter
    def status(self, value: NodeStatus):
        """Set the execution status of the node."""
        self._status = value


class Pipeline:
    """Represents a processing pipeline composed of connected nodes."""

    def __init__(self, name: str, nodes: List[Node], *, expected_inputs: Set[str]):
        """
        Initialize a pipeline.

        Args:
            name: Unique name of the pipeline
            nodes: List of nodes in execution order
            expected_inputs: Set of input keys expected from external sources

        Note:
            expected_inputs is a keyword-only argument to make its purpose clear
            at call sites.
        """
        self.name = name
        self.nodes = nodes
        self.expected_inputs = expected_inputs
        self._validate_pipeline()

    def _validate_pipeline(self):
        """
        Validate pipeline connectivity and data dependencies.

        Takes into account expected external inputs when validating.

        Raises:
            ValueError: If pipeline configuration is invalid
        """
        # Start with expected external inputs
        available_outputs = self.expected_inputs.copy()

        # Track inputs and outputs for cycle detection
        for node in self.nodes:
            # Check if required inputs are available
            missing_inputs = node.inputs - available_outputs
            if missing_inputs:
                raise ValueError(
                    f"Node {node.name} requires inputs {missing_inputs} "
                    f"which are not provided by previous nodes or external inputs"
                )

            # Add this node's outputs to available outputs
            available_outputs.update(node.outputs)

            # Check for duplicate outputs (not allowed)
            duplicate_outputs = node.outputs & (available_outputs - node.outputs)
            if duplicate_outputs:
                raise ValueError(
                    f"Node {node.name} produces outputs {duplicate_outputs} "
                    f"which are already produced by previous nodes"
                )

    async def execute(self, initial_inputs: Dict[str, Any]) -> Dict[str, NodeResult]:
        """
        Execute the pipeline with given inputs.

        Args:
            initial_inputs: Initial input data for the pipeline

        Returns:
            Dictionary mapping output names to their NodeResults

        Raises:
            ValueError: If required inputs are missing
        """
        # Validate that all expected inputs are provided
        missing_inputs = self.expected_inputs - set(initial_inputs.keys())
        if missing_inputs:
            raise ValueError(
                f"Missing required inputs for pipeline {self.name}: {missing_inputs}"
            )

        # Initialize data dictionary with inputs
        current_data = initial_inputs.copy()
        results = {}

        # Execute nodes in sequence
        for node in self.nodes:
            try:
                logger.info(f"Executing node: {node.name}")
                node.status = NodeStatus.RUNNING

                # Prepare inputs for this node
                node_inputs = {
                    key: current_data[key] for key in node.inputs if key in current_data
                }

                # Execute node
                result = await node.execute(node_inputs)
                node.status = (
                    NodeStatus.COMPLETED if result.is_success else NodeStatus.FAILED
                )

                if result.is_success and result.data:
                    # Update available data with node outputs
                    current_data.update(result.data)
                    # Store results
                    results.update({output: result for output in node.outputs})
                else:
                    logger.error(f"Node {node.name} failed: {result.error}")
                    node.status = NodeStatus.FAILED
                    break

            except Exception as e:
                logger.exception(f"Error executing node {node.name}")
                node.status = NodeStatus.FAILED
                results[node.name] = NodeResult(status=NodeStatus.FAILED, error=e)
                break

        return results


class PipelineRegistry:
    """Registry for managing multiple pipelines."""

    def __init__(self):
        """Initialize an empty pipeline registry."""
        self.pipelines: Dict[str, Pipeline] = {}

    def register_pipeline(self, pipeline: Pipeline):
        """
        Register a pipeline.

        Args:
            pipeline: Pipeline instance to register
        """
        if pipeline.name in self.pipelines:
            logger.warning(f"Overwriting existing pipeline: {pipeline.name}")
        self.pipelines[pipeline.name] = pipeline

    def get_pipeline(self, name: str) -> Optional[Pipeline]:
        """
        Get a registered pipeline by name.

        Args:
            name: Name of the pipeline to retrieve

        Returns:
            Pipeline instance if found, None otherwise
        """
        return self.pipelines.get(name)

    async def execute_pipeline(
        self, name: str, inputs: Dict[str, Any]
    ) -> Dict[str, NodeResult]:
        """
        Execute a registered pipeline.

        Args:
            name: Name of the pipeline to execute
            inputs: Input data for the pipeline

        Returns:
            Dictionary mapping output names to their NodeResults

        Raises:
            ValueError: If pipeline not found
        """
        pipeline = self.get_pipeline(name)
        if not pipeline:
            raise ValueError(f"Pipeline not found: {name}")

        return await pipeline.execute(inputs)
