# P3 CUDA Evaluation Device-Mismatch Fix

Status: **COMPLETE**

The P3 Colab environment reached the free-running evaluation smoke before
optimizer step 1. `scaled_duration_plan` constructs a CPU integer duration plan
while its `AlignmentUnitBatch` masks live with the CUDA model. The plan was
passed to `DurationPredictor.validate_plan` before the existing
`prepare_generation` device transfer, so CUDA boolean masks indexed a CPU
duration tensor.

The fix is limited to `DurationPredictor.validate_plan`: both masks are copied
to the duration tensor's device as `torch.bool` before boolean indexing. No
duration values, limits, validation rules, model parameters, initialization,
loss, optimizer, or schedule changed.

The immediate P2/P3 evaluation path was audited. Other duration indexing in
`duration_metrics`, `duration_row`, and teacher-forced evaluation uses masks and
tensors created together on the model device. Supplied plans entering
`SwaraSpeechPoCV1.prepare_generation` already move to the linguistic-state
device before expansion. No additional confirmed mismatch was changed.

## Verification

- Focused Gate C/D suite: 35 passed, 1 CUDA-only skip, 0 failed.
- Full available unittest suite: 85 passed, 2 optional skips, 0 failed.
- Frozen model config SHA256:
  `8c2414f838899e112975ed4fdd61215f59f3c03b059149fd4b1ce36e95f1c24c`
- Frozen P3 config SHA256:
  `2c365f070046593fdaf3670cf0c9d1de05acb8a6b72f5dd70842fe1e38387816`
- Trainable parameters: 13,393,283.
- Patched source SHA256:
  `700705947071378d790c99e8680b9583b73d3ec2cda63a64fe6c49ddda42765a`

The expected rerun initialization SHA is the authoritative Colab value
`2ba0277bc6e7172d8a1d9a9e0bf115d48d26827d1efb0b19cfa1f1b1b17c7553`.
The initialization digest hashes model state, not Python source, so this fix
must not change it in the same Colab runtime/software environment. A different
digest is a stop condition requiring explanation.

Optimizer steps during diagnosis/fix: **0**  
Recovery required: **NO**  
Architecture changed: **NO**  
Numerical training behavior changed: **NO**  
Commit/push: **NO**
