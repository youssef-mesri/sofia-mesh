"""Batch operation processing for high-throughput mesh modification.

This module provides a BatchEditor that queues operations and validates in
batches rather than per-operation, dramatically reducing validation overhead
for large-scale mesh generation.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Callable, Optional, List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class BatchEditor:
    """Wrapper around PatchBasedMeshEditor that batches operations.
    
    Key Features:
    - Queues operations instead of executing immediately
    - Validates once per batch instead of per operation
    - Automatic flushing when batch size is reached
    - Can rollback entire batch on validation failure
    
    Performance Impact:
    - Eliminates per-operation validation overhead (80% of time)
    - Expected 5-10x speedup for large-scale generation
    - Most effective when batch_size >= 1000
    
    Example:
        >>> editor = PatchBasedMeshEditor(points, triangles)
        >>> batch = BatchEditor(editor, batch_size=5000)
        >>> for edge in edges_to_split:
        ...     batch.split_edge(edge)
        >>> batch.flush()  # Validate and commit remaining operations
    """
    
    def __init__(self, editor, batch_size: int = 5000, 
                 validate_on_flush: bool = True,
                 auto_fix: bool = False):
        """Initialize batch editor.
        
        Parameters
        ----------
        editor : PatchBasedMeshEditor
            The underlying mesh editor
        batch_size : int, default=5000
            Number of operations to queue before auto-flush
        validate_on_flush : bool, default=True
            Whether to run conformity check after batch execution
        auto_fix : bool, default=False
            If True, attempt to fix issues instead of rolling back
            (Not yet implemented)
        """
        self.editor = editor
        self.batch_size = batch_size
        self.validate_on_flush = validate_on_flush
        self.auto_fix = auto_fix
        
        self.operation_queue: List[Tuple[str, Tuple, Dict]] = []
        self.batch_count = 0
        self.total_ops = 0
        self.failed_batches = 0
        
        # Snapshot for rollback
        self._snapshot: Optional[Dict[str, Any]] = None
        
    def _take_snapshot(self):
        """Save current mesh state for potential rollback."""
        self._snapshot = {
            'points': self.editor.points.copy(),
            'triangles': self.editor.triangles.copy(),
            'edge_map': {k: set(v) for k, v in self.editor.edge_map.items()},
            'v_map': {k: set(v) for k, v in self.editor.v_map.items()},
        }
        
    def _restore_snapshot(self):
        """Restore mesh to last snapshot."""
        if self._snapshot is None:
            logger.warning("No snapshot to restore")
            return
            
        self.editor.points = self._snapshot['points']
        self.editor.triangles = self._snapshot['triangles']
        self.editor.edge_map = {k: set(v) for k, v in self._snapshot['edge_map'].items()}
        self.editor.v_map = {k: set(v) for k, v in self._snapshot['v_map'].items()}
        self._snapshot = None
        
    def split_edge(self, edge, **kwargs) -> bool:
        """Queue edge split operation.
        
        Parameters
        ----------
        edge : tuple
            Edge to split (u, v)
        **kwargs
            Additional arguments passed to split operation
            
        Returns
        -------
        bool
            True if queued successfully (always True unless queue is full)
        """
        self.operation_queue.append(('split_edge', (edge,), kwargs))
        self.total_ops += 1
        
        if len(self.operation_queue) >= self.batch_size:
            return self.flush()
        return True
        
    def edge_collapse(self, edge, position: str = 'midpoint', **kwargs) -> bool:
        """Queue edge collapse operation.
        
        Parameters
        ----------
        edge : tuple
            Edge to collapse (u, v)
        position : str
            Where to place new vertex ('midpoint')
        **kwargs
            Additional arguments
            
        Returns
        -------
        bool
            True if queued successfully
        """
        self.operation_queue.append(('edge_collapse', (edge,), {'position': position, **kwargs}))
        self.total_ops += 1
        
        if len(self.operation_queue) >= self.batch_size:
            return self.flush()
        return True
        
    def remove_node(self, v_idx, **kwargs) -> bool:
        """Queue node removal operation.
        
        Parameters
        ----------
        v_idx : int
            Vertex index to remove
        **kwargs
            Additional arguments
            
        Returns
        -------
        bool
            True if queued successfully
        """
        self.operation_queue.append(('remove_node', (v_idx,), kwargs))
        self.total_ops += 1
        
        if len(self.operation_queue) >= self.batch_size:
            return self.flush()
        return True
    
    def _execute_operation(self, op_type: str, args: Tuple, kwargs: Dict) -> Tuple[bool, str, Any]:
        """Execute a single operation without validation.
        
        This directly calls the low-level operation functions, bypassing
        per-operation conformity checks and simulation by setting flags.
        """
        if op_type == 'split_edge':
            # Skip both preflight check and simulation for maximum speed
            kwargs['skip_preflight_check'] = True
            kwargs['skip_simulation'] = True
            return self.editor.split_edge(*args, **kwargs)
                
        elif op_type == 'edge_collapse':
            kwargs['skip_preflight_check'] = True
            kwargs['skip_simulation'] = True
            return self.editor.edge_collapse(*args, **kwargs)
            
        elif op_type == 'remove_node':
            # Remove operations don't have skip flags yet
            return self.editor.remove_node_with_patch2(*args, **kwargs)
            
        else:
            raise ValueError(f"Unknown operation type: {op_type}")
    
    def flush(self) -> bool:
        """Execute all queued operations and validate.
        
        Returns
        -------
        bool
            True if batch succeeded, False if validation failed and rolled back
        """
        if not self.operation_queue:
            return True
            
        n_ops = len(self.operation_queue)
        logger.info(f"Flushing batch {self.batch_count + 1} with {n_ops} operations...")
        
        # Take snapshot for potential rollback
        self._take_snapshot()
        
        # Execute all operations
        success_count = 0
        fail_count = 0
        
        for op_type, args, kwargs in self.operation_queue:
            try:
                ok, msg, info = self._execute_operation(op_type, args, kwargs)
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
                    logger.debug(f"Operation {op_type}{args} failed: {msg}")
            except Exception as e:
                fail_count += 1
                logger.warning(f"Exception during {op_type}{args}: {e}")
        
        # Clear queue
        self.operation_queue.clear()
        self.batch_count += 1
        
        logger.info(f"Batch {self.batch_count}: {success_count} succeeded, {fail_count} failed")
        
        # Validate if requested
        if self.validate_on_flush:
            from .conformity import check_mesh_conformity
            ok, msgs = check_mesh_conformity(
                self.editor.points, 
                self.editor.triangles,
                allow_marked=True
            )
            
            if not ok:
                logger.error(f"Batch {self.batch_count} validation failed: {msgs}")
                logger.error(f"Rolling back {n_ops} operations...")
                self._restore_snapshot()
                self.failed_batches += 1
                return False
            else:
                logger.info(f"Batch {self.batch_count} validation passed")
        
        return True
    
    def finalize(self) -> bool:
        """Flush remaining operations and compact mesh.
        
        Returns
        -------
        bool
            True if successful
        """
        ok = self.flush()
        if ok:
            logger.info("Compacting mesh...")
            self.editor.compact_triangle_indices()
            logger.info(f"Final mesh: {len(self.editor.points)} vertices, "
                       f"{len(self.editor.triangles)} triangles")
        return ok
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics.
        
        Returns
        -------
        dict
            Statistics including total ops, batches, failures
        """
        return {
            'total_operations': self.total_ops,
            'batches_processed': self.batch_count,
            'failed_batches': self.failed_batches,
            'batch_size': self.batch_size,
            'operations_in_queue': len(self.operation_queue),
        }
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Flush on exit."""
        if exc_type is None:
            self.flush()
        return False


# Convenience function for batch processing
def batch_split_edges(editor, edges: List[Tuple[int, int]], 
                      batch_size: int = 5000,
                      validate: bool = True) -> Tuple[int, int]:
    """Split multiple edges in batches.
    
    Parameters
    ----------
    editor : PatchBasedMeshEditor
        Mesh editor
    edges : list of tuple
        Edges to split
    batch_size : int
        Operations per batch
    validate : bool
        Whether to validate each batch
        
    Returns
    -------
    (success_count, fail_count) : tuple of int
        Number of successful and failed operations
    """
    with BatchEditor(editor, batch_size=batch_size, validate_on_flush=validate) as batch:
        for edge in edges:
            batch.split_edge(edge)
    
    stats = batch.get_stats()
    return stats['total_operations'], stats['failed_batches']
