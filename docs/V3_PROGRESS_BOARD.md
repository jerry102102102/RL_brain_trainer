# V3 Progress Board (online-memory)

## TODO
- [ ] Replace offline data-collection+supervised fit with true online loop (act->step->store->update)
- [ ] Add replay transitions schema for recurrent batches
- [ ] Add episodic memory retrieval + confidence gate into policy input
- [ ] Run ablation B (online no-memory) vs C (online + memory)
- [ ] Report timeout distance bins (near/mid/far)

## DOING
- [x] Create dedicated V3 script/config scaffold (`train_rl_brainer_v3_online.py`, `train_rl_brainer_v3_online_quick.yaml`)
- [ ] Refactor `train_rl_brainer_v3_online.py` training core to online updates

## DONE
- [x] Created V3 branch: `v3-online-memory`
- [x] Added architecture/causal plan: `docs/V3_ONLINE_MEMORY_LSTM_PLAN.md`
- [x] Verified CUDA bootstrap run pipeline
