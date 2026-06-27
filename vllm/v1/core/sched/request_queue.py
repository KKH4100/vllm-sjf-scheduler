# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    SJF = "sjf"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")


class SJFRequestQueue(RequestQueue):
    """
    A Shortest-Job-First queue using a heap ordered by OPT predictor scores.
    Lower score = shorter predicted output = processed first.
    Aging prevents starvation of long requests.
    """

    def __init__(
        self,
        predictor: "OPTPredictor | None" = None,  # noqa: F821
        precomputed_scores: dict | None = None,
    ) -> None:
        self._heap: list["SJFSorter"] = []  # noqa: F821
        self._predictor = predictor
        self._precomputed_scores = precomputed_scores or {}

    def add_request(self, request: Request) -> None:
        import hashlib
        from vllm.v1.core.sched.policy.shortest_job_first import SJFSorter
        assert request.prompt_token_ids is not None

        # precomputed score lookup 먼저 시도
        if self._precomputed_scores:
            key = "tid_" + hashlib.md5(
                str(request.prompt_token_ids).encode()
            ).hexdigest()
            base_score = self._precomputed_scores.get(key, None)
        else:
            base_score = None

        # fallback: predictor 또는 prompt 길이 기반
        if base_score is None:
            if self._predictor is not None:
                base_score = self._predictor.predict_score(
                    request.prompt_token_ids
                )
            else:
                base_score = -len(request.prompt_token_ids) / 1000.0

        sorter = SJFSorter(request, base_score)
        heapq.heappush(self._heap, sorter)

    # 대기 시간 threshold (초) — 이 이상 기다린 요청은 강제 최우선 처리
    MAX_WAIT_SECONDS = 10.0

    def pop_request(self) -> Request:
        if not self._heap:
            raise IndexError("pop from empty heap")

        # 대기 시간이 MAX_WAIT_SECONDS 초과한 요청이 있으면 강제 우선 처리
        now = time.time()
        for sorter in self._heap:
            if now - sorter._arrival_time > self.MAX_WAIT_SECONDS:
                self._heap.remove(sorter)
                heapq.heapify(self._heap)
                return sorter.request

        # 없으면 정상 SJF 정렬
        heapq.heapify(self._heap)
        return heapq.heappop(self._heap).request

    def peek_request(self) -> Request:
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0].request

    def prepend_request(self, request: Request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        self._heap = [s for s in self._heap
                      if s.request_id != request.request_id]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        ids = {r.request_id for r in requests}
        self._heap = [s for s in self._heap if s.request_id not in ids]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy).request


def create_sjf_queue(
    model_path: str | None = None,
    device: str = "cpu",
    precomputed_scores_path: str | None = None,
) -> SJFRequestQueue:
    """Create SJF queue with precomputed scores or OPT predictor."""
    import json
    precomputed_scores = {}
    if precomputed_scores_path:
        with open(precomputed_scores_path) as f:
            precomputed_scores = json.load(f)
        from vllm.logger import init_logger
        logger = init_logger(__name__)
        logger.info("Loaded %d precomputed scores from %s",
                    len(precomputed_scores), precomputed_scores_path)

    predictor = None
    if model_path and not precomputed_scores:
        from vllm.v1.core.sched.policy.shortest_job_first import OPTPredictor
        predictor = OPTPredictor(model_path=model_path, device=device)

    return SJFRequestQueue(
        predictor=predictor,
        precomputed_scores=precomputed_scores,
    )
