# vLLM SJF Scheduler

vLLM에 Shortest Job First(SJF) 스케줄링 정책을 구현한 프로젝트입니다.
기반 코드: [vllm-project/vllm](https://github.com/vllm-project/vllm)
[vllm-ltr](https://github.com/hao-ai-lab/vllm-ltr)

[구현 결과](./vLLM_SJF.pdf)
## 수정/추가한 파일

| 파일 | 변경 내용 |
|------|-----------|
| `vllm/config/scheduler.py` | SJF 관련 스케줄러 설정 추가 |
| `vllm/v1/core/sched/scheduler.py` | SJF 정책 연동 |
| `vllm/v1/core/sched/request_queue.py` | 요청 정렬 로직 수정 |
| `vllm/v1/core/sched/policy/__init__.py` | 정책 모듈 추가 |
| `vllm/v1/core/sched/policy/shortest_job_first.py` | SJF 구현 |

## 실험 환경
- 모델: Llama3-8B
- GPU: H100
- 비교: FCFS vs SJF (TTFT/TPOT 기준)
