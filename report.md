# Cambricon PyTorch Model Migration Report
## Cambricon PyTorch Changes
| No. |  File  |  Description  |
| 1 | cli_demo.py:7 | change "model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()" to "model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().mlu() " |
| 2 | web_demo_old.py:5 | change "model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()" to "model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().mlu() " |
| 3 | api.py:4 | add "import torch_mlu" |
| 4 | api.py:6 | change "DEVICE = "cuda"" to "DEVICE = "mlu" " |
| 5 | api.py:12 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 6 | api.py:13 | change "with torch.cuda.device(CUDA_DEVICE):" to "with torch.mlu.device(CUDA_DEVICE): " |
| 7 | api.py:14 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 8 | api.py:15 | change "torch.cuda.ipc_collect()" to "torch.mlu.ipc_collect() " |
| 9 | api.py:54 | change "model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()" to "model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().mlu() " |
| 10 | web_demo.py:6 | change "model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()" to "model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().mlu() " |
| 11 | web_demo2.py:15 | change "model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()" to "model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().mlu() " |
| 12 | utils.py:41 | change "model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()" to "model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half().mlu() " |
| 13 | ptuning/web_demo.py:6 | add "import torch_mlu" |
| 14 | ptuning/web_demo.py:157 | change "model = model.half().cuda()" to "model = model.half().mlu() " |
| 15 | ptuning/web_demo.py:158 | change "model.transformer.prefix_encoder.float().cuda()" to "model.transformer.prefix_encoder.float().mlu() " |
| 16 | ptuning/trainer.py:59 | add "import torch_mlu" |
| 17 | ptuning/trainer.py:455 | change "# postpone switching model to cuda when:" to "# postpone switching model to mlu when: " |
| 18 | ptuning/trainer.py:561 | change "self.use_cuda_amp = False" to "self.use_mlu_amp = False " |
| 19 | ptuning/trainer.py:597 | change "args.half_precision_backend = "cuda_amp"" to "args.half_precision_backend = "mlu_amp" " |
| 20 | ptuning/trainer.py:604 | change "if args.half_precision_backend == "cuda_amp":" to "if args.half_precision_backend == "mlu_amp": " |
| 21 | ptuning/trainer.py:605 | change "self.use_cuda_amp = True" to "self.use_mlu_amp = True " |
| 22 | ptuning/trainer.py:623 | change "self.scaler = torch.cuda.amp.GradScaler()" to "self.scaler = torch.mlu.amp.GradScaler() " |
| 23 | ptuning/trainer.py:638 | change "and self.use_cuda_amp" to "and self.use_mlu_amp " |
| 24 | ptuning/trainer.py:1335 | change "self.use_cuda_amp = False" to "self.use_mlu_amp = False " |
| 25 | ptuning/trainer.py:2272 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 26 | ptuning/trainer.py:2274 | change "torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])" to "torch.mlu.random.set_rng_state(checkpoint_rng_state["mlu"]) " |
| 27 | ptuning/trainer.py:2277 | change "torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])" to "torch.mlu.random.set_rng_state_all(checkpoint_rng_state["mlu"]) " |
| 28 | ptuning/trainer.py:2366 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 29 | ptuning/trainer.py:2369 | change "rng_states["cuda"] = torch.cuda.random.get_rng_state_all()" to "rng_states["mlu"] = torch.mlu.random.get_rng_state_all() " |
| 30 | ptuning/trainer.py:2371 | change "rng_states["cuda"] = torch.cuda.random.get_rng_state()" to "rng_states["mlu"] = torch.mlu.random.get_rng_state() " |
| 31 | ptuning/trainer.py:2607 | change "if self.use_cuda_amp or self.use_cpu_amp:" to "if self.use_mlu_amp or self.use_cpu_amp: " |
| 32 | ptuning/trainer.py:2612 | change "else torch.cuda.amp.autocast(cache_enabled=cache_enabled, dtype=self.amp_dtype)" to "else torch.mlu.amp.autocast(cache_enabled=cache_enabled, dtype=self.amp_dtype) " |
| 33 | ptuning/trainer.py:2615 | change "ctx_manager = torch.cuda.amp.autocast()" to "ctx_manager = torch.mlu.amp.autocast() " |
| 34 | ptuning/trainer_seq2seq.py:17 | add "import torch_mlu" |
| 35 | ptuning/main.py:31 | add "import torch_mlu" |
