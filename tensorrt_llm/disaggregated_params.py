from dataclasses import dataclass
from typing import List, Optional

from tensorrt_llm.bindings import executor as tllme
from tensorrt_llm.logger import logger


@dataclass(slots=True, kw_only=True)
class DisaggregatedParams:
    """Disaggregated serving parameters.

    Args:
        request_type (str): The type of request ("context_only" | "generation_only" | "context_and_generation")
        first_gen_tokens (List[int]): The first tokens of the generation request
        ctx_request_id (int): The context request id
        opaque_state(bytes): Any additional state needing to be exchanged between context and gen instances
    """

    request_type: Optional[str] = None
    first_gen_tokens: Optional[List[int]] = None
    ctx_request_id: Optional[int] = None
    opaque_state: Optional[bytes] = None
    draft_tokens: Optional[List[int]] = None

    def get_context_phase_params(self) -> tllme.ContextPhaseParams:
        logger.debug(f"[trace - disaggregated_params.py] get_context_phase_params called with:")
        logger.debug(f"[trace - disaggregated_params.py]   first_gen_tokens: {self.first_gen_tokens} (type: {type(self.first_gen_tokens)})")
        logger.debug(f"[trace - disaggregated_params.py]   ctx_request_id: {self.ctx_request_id} (type: {type(self.ctx_request_id)})")
        logger.debug(f"[trace - disaggregated_params.py]   opaque_state: {self.opaque_state} (type: {type(self.opaque_state)})")
        logger.debug(f"[trace - disaggregated_params.py]   draft_tokens: {self.draft_tokens} (type: {type(self.draft_tokens)})")
        
        # Check if required parameters are None for generation_only requests
        if self.request_type == "generation_only":
            if self.first_gen_tokens is None:
                logger.error(f"[trace - disaggregated_params.py] ERROR: first_gen_tokens is None for generation_only request!")
            if self.ctx_request_id is None:
                logger.error(f"[trace - disaggregated_params.py] ERROR: ctx_request_id is None for generation_only request!")
        
        logger.debug(f"[trace - disaggregated_params.py] About to call ContextPhaseParams constructor...")
        try:
            result = tllme.ContextPhaseParams(
                self.first_gen_tokens, self.ctx_request_id, self.opaque_state, self.draft_tokens
            )
            logger.debug(f"[trace - disaggregated_params.py] ContextPhaseParams constructor succeeded")
            return result
        except Exception as e:
            logger.error(f"[trace - disaggregated_params.py] ContextPhaseParams constructor failed: {e}")
            logger.error(f"[trace - disaggregated_params.py] Constructor signature expects: arg0: list[int], arg1: int, arg2: Optional[bytes], arg3: Optional[list[int]]")
            logger.error(f"[trace - disaggregated_params.py] But got: {self.first_gen_tokens}, {self.ctx_request_id}, {self.opaque_state}, {self.draft_tokens}")
            raise

    def get_request_type(self) -> tllme.RequestType:
        logger.debug(f"[trace - disaggregated_params.py] get_request_type called with request_type: {self.request_type}")
        if self.request_type == "context_only":
            return tllme.RequestType.REQUEST_TYPE_CONTEXT_ONLY
        elif self.request_type == "generation_only":
            return tllme.RequestType.REQUEST_TYPE_GENERATION_ONLY
        elif self.request_type == "context_and_generation":
            return tllme.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION
        else:
            raise ValueError(
                f"Unknown request type: {self.request_type}. Must be context_only, generation_only or "
                "context_and_generation"
            )
